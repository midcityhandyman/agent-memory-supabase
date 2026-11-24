"""In memory backend: the whole system with no database.

InMemoryDatabase plays the part of the Postgres tables. as_user(owner_id)
returns a backend scoped to one owner, which plays the part of a signed in
Supabase session under row level security: it can only see and change rows
it owns. recall() follows the same steps as the SQL function
public.recall_memories, using the reference ranking in ranking.py.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence

from .embeddings import cosine_similarity
from .models import MEMORY_KINDS, OUTCOMES, Deposit, Memory, Outcome, RecallHit, Rule
from .ranking import DEFAULT_CONFIG, Candidate, RankingConfig, rank_candidates
from .textsearch import text_score


class NotFound(LookupError):
    """Row does not exist or is not visible to the current owner."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class InMemoryDatabase:
    """Shared storage for every owner. Pass clock to control time in tests."""

    def __init__(self, clock: Callable[[], datetime] = _utcnow,
                 id_factory: Optional[Callable[[], str]] = None):
        self.clock = clock
        self._id_factory = id_factory or (lambda: str(uuid.uuid4()))
        self.rules: list[Rule] = []
        self.memories: list[Memory] = []
        self.deposits: list[Deposit] = []
        self.outcomes: list[Outcome] = []

    def new_id(self) -> str:
        return self._id_factory()

    def as_user(self, owner_id: str) -> "InMemoryBackend":
        if not owner_id:
            raise ValueError("owner_id is required")
        return InMemoryBackend(self, owner_id)


class InMemoryBackend:
    """One owner's view of an InMemoryDatabase."""

    def __init__(self, db: InMemoryDatabase, owner_id: str):
        self.db = db
        self.owner_id = owner_id

    # ---- rules -----------------------------------------------------------

    def add_rule(self, agent_id: str, rule: str, priority: int = 100,
                 source: Optional[str] = None) -> Rule:
        if not rule.strip():
            raise ValueError("rule must not be empty")
        row = Rule(id=self.db.new_id(), owner_id=self.owner_id, agent_id=agent_id,
                   rule=rule, priority=priority, active=True,
                   created_at=self.db.clock(), source=source)
        self.db.rules.append(row)
        return row

    def active_rules(self, agent_id: str) -> list[Rule]:
        rows = [r for r in self.db.rules
                if r.owner_id == self.owner_id and r.agent_id == agent_id and r.active]
        return sorted(rows, key=lambda r: (r.priority, r.created_at, r.id))

    def deactivate_rule(self, rule_id: str) -> bool:
        for row in self.db.rules:
            if row.id == rule_id and row.owner_id == self.owner_id and row.active:
                row.active = False
                return True
        return False

    # ---- memories --------------------------------------------------------

    def add_memory(self, agent_id: str, content: str, title: str = "", kind: str = "fact",
                   tags: Sequence[str] = (), embedding: Optional[Sequence[float]] = None,
                   created_at: Optional[datetime] = None) -> Memory:
        if not content.strip():
            raise ValueError("content must not be empty")
        if kind not in MEMORY_KINDS:
            raise ValueError(f"kind must be one of {MEMORY_KINDS}")
        row = Memory(id=self.db.new_id(), owner_id=self.owner_id, agent_id=agent_id,
                     kind=kind, title=title, content=content, tags=list(tags),
                     embedding=list(embedding) if embedding is not None else None,
                     created_at=created_at or self.db.clock())
        self.db.memories.append(row)
        return row

    def _visible_memory(self, memory_id: str) -> Memory:
        for row in self.db.memories:
            if row.id == memory_id and row.owner_id == self.owner_id:
                return row
        raise NotFound(f"memory {memory_id} not found")

    def archive_memory(self, memory_id: str) -> bool:
        try:
            row = self._visible_memory(memory_id)
        except NotFound:
            return False
        if row.archived_at is not None:
            return False
        row.archived_at = self.db.clock()
        return True

    # ---- deposits --------------------------------------------------------

    def add_deposit(self, agent_id: str, content: str, source: Optional[str] = None) -> Deposit:
        if not content.strip():
            raise ValueError("content must not be empty")
        row = Deposit(id=self.db.new_id(), owner_id=self.owner_id, agent_id=agent_id,
                      content=content, status="pending", created_at=self.db.clock(),
                      source=source)
        self.db.deposits.append(row)
        return row

    def pending_deposits(self, agent_id: str) -> list[Deposit]:
        rows = [d for d in self.db.deposits
                if d.owner_id == self.owner_id and d.agent_id == agent_id
                and d.status == "pending"]
        return sorted(rows, key=lambda d: (d.created_at, d.id))

    def _visible_deposit(self, deposit_id: str) -> Deposit:
        for row in self.db.deposits:
            if row.id == deposit_id and row.owner_id == self.owner_id:
                return row
        raise NotFound(f"deposit {deposit_id} not found")

    def promote_deposit(self, deposit_id: str, title: str = "", kind: str = "fact",
                        tags: Sequence[str] = (),
                        embedding: Optional[Sequence[float]] = None) -> str:
        deposit = self._visible_deposit(deposit_id)
        if deposit.status != "pending":
            raise ValueError(f"deposit {deposit_id} is already {deposit.status}")
        memory = self.add_memory(deposit.agent_id, deposit.content, title=title, kind=kind,
                                 tags=tags, embedding=embedding)
        deposit.status = "promoted"
        deposit.promoted_memory_id = memory.id
        deposit.resolved_at = self.db.clock()
        return memory.id

    def reject_deposit(self, deposit_id: str) -> bool:
        try:
            deposit = self._visible_deposit(deposit_id)
        except NotFound:
            return False
        if deposit.status != "pending":
            return False
        deposit.status = "rejected"
        deposit.resolved_at = self.db.clock()
        return True

    # ---- outcomes --------------------------------------------------------

    def record_outcome(self, memory_id: str, outcome: str, query: Optional[str] = None,
                       note: Optional[str] = None) -> str:
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}")
        # Same effect as the RLS insert policy: you can only vote on a
        # memory you can see.
        self._visible_memory(memory_id)
        row = Outcome(id=self.db.new_id(), owner_id=self.owner_id, memory_id=memory_id,
                      outcome=outcome, created_at=self.db.clock(), query=query, note=note)
        self.db.outcomes.append(row)
        return row.id

    def outcome_tally(self, memory_id: str) -> tuple[int, int]:
        """(helped, not_helped) counts for one memory, as this owner sees them."""
        helped = not_helped = 0
        for row in self.db.outcomes:
            if row.memory_id != memory_id or row.owner_id != self.owner_id:
                continue
            if row.outcome == "helped":
                helped += 1
            elif row.outcome == "not_helped":
                not_helped += 1
        return helped, not_helped

    # ---- recall ----------------------------------------------------------

    def recall(self, query: str, query_embedding: Optional[Sequence[float]] = None,
               agent_id: str = "default", match_count: int = 8, candidate_pool: int = 50,
               config: RankingConfig = DEFAULT_CONFIG) -> list[RecallHit]:
        pool = max(candidate_pool, 1)
        visible = [m for m in self.db.memories
                   if m.owner_id == self.owner_id and m.agent_id == agent_id
                   and m.archived_at is None]
        by_id = {m.id: m for m in visible}

        # text_hits CTE
        text_hits = [(m.id, text_score(query, m.title, m.content)) for m in visible]
        text_hits = sorted((h for h in text_hits if h[1] > 0), key=lambda h: -h[1])[:pool]

        # vector_hits CTE
        vector_hits: list[tuple[str, float]] = []
        if query_embedding is not None:
            scored = [(m.id, cosine_similarity(m.embedding, query_embedding))
                      for m in visible if m.embedding is not None]
            scored.sort(key=lambda h: -h[1])
            vector_hits = [(mid, max(sim, 0.0)) for mid, sim in scored[:pool]]

        # candidates CTE: full outer join of the two hit lists
        text_by_id = dict(text_hits)
        vector_by_id = dict(vector_hits)
        candidates = []
        for mid in {**text_by_id, **vector_by_id}:
            helped, not_helped = self.outcome_tally(mid)
            candidates.append(Candidate(
                id=mid, created_at=by_id[mid].created_at,
                text_score=text_by_id.get(mid, 0.0),
                vector_score=vector_by_id.get(mid, 0.0),
                helped=helped, not_helped=not_helped))

        effective = config if query_embedding is not None else config.without_vector()
        ranked = rank_candidates(candidates, now=self.db.clock(), config=effective,
                                 limit=match_count)
        hits = []
        for item in ranked:
            cand, mem = item.candidate, by_id[item.candidate.id]
            hits.append(RecallHit(
                id=mem.id, kind=mem.kind, title=mem.title, content=mem.content,
                tags=list(mem.tags), created_at=mem.created_at,
                text_score=cand.text_score, vector_score=cand.vector_score,
                helped=cand.helped, not_helped=cand.not_helped, score=item.score))
        return hits
