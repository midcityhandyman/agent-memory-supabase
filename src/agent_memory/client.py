"""MemoryClient: the small API an agent actually calls.

    boot()      load standing rules at the start of a session
    recall()    fetch the memories most likely to help with a query
    feedback()  say whether a recalled memory helped (this drives re-ranking)
    deposit()   drop a raw note in the inbox
    promote()   turn a deposit into a curated memory
    remember()  write a curated memory directly

The client is backend agnostic: InMemoryBackend for tests and demos,
SupabaseBackend for the real thing.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Sequence

from .embeddings import Embedder, HashEmbedder, embedder_from_env
from .memory_backend import InMemoryDatabase
from .models import Deposit, Memory, RecallHit, Rule
from .ranking import DEFAULT_CONFIG, RankingConfig
from .supabase_backend import SupabaseBackend


class MemoryClient:
    def __init__(self, backend: Any, embedder: Optional[Embedder] = None,
                 agent_id: str = "default", config: RankingConfig = DEFAULT_CONFIG):
        self.backend = backend
        self.embedder = embedder if embedder is not None else HashEmbedder()
        self.agent_id = agent_id
        self.config = config

    @classmethod
    def offline(cls, owner_id: str = "local-user", agent_id: str = "default",
                db: Optional[InMemoryDatabase] = None) -> "MemoryClient":
        """A client that needs no network, no keys and no database."""
        db = db or InMemoryDatabase()
        return cls(db.as_user(owner_id), HashEmbedder(), agent_id=agent_id)

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None,
                 agent_id: Optional[str] = None) -> "MemoryClient":
        """Supabase when SUPABASE_URL and SUPABASE_KEY are set, else offline."""
        env = os.environ if env is None else env
        agent = agent_id or env.get("AGENT_ID", "default")
        url, key = env.get("SUPABASE_URL", "").strip(), env.get("SUPABASE_KEY", "").strip()
        if not (url and key):
            return cls.offline(agent_id=agent)
        backend = SupabaseBackend(url, key, access_token=env.get("SUPABASE_ACCESS_TOKEN") or None)
        return cls(backend, embedder_from_env(env), agent_id=agent)

    # ---- rules layer -----------------------------------------------------

    def add_rule(self, rule: str, priority: int = 100, source: Optional[str] = None) -> Rule:
        return self.backend.add_rule(self.agent_id, rule, priority=priority, source=source)

    def boot(self) -> list[Rule]:
        """Standing rules for this agent, most important first."""
        return self.backend.active_rules(self.agent_id)

    # ---- memory layer ----------------------------------------------------

    def remember(self, content: str, title: str = "", kind: str = "fact",
                 tags: Sequence[str] = ()) -> Memory:
        embedding = self.embedder.embed((title + "\n" + content).strip())
        return self.backend.add_memory(self.agent_id, content, title=title, kind=kind,
                                       tags=tags, embedding=embedding)

    def recall(self, query: str, limit: int = 5, use_vectors: bool = True) -> list[RecallHit]:
        embedding = self.embedder.embed(query) if use_vectors else None
        return self.backend.recall(query, query_embedding=embedding, agent_id=self.agent_id,
                                   match_count=limit, config=self.config)

    def forget(self, memory_id: str) -> bool:
        """Archive a memory so recall stops returning it. Nothing is deleted."""
        return self.backend.archive_memory(memory_id)

    # ---- deposit layer ---------------------------------------------------

    def deposit(self, content: str, source: Optional[str] = None) -> Deposit:
        return self.backend.add_deposit(self.agent_id, content, source=source)

    def inbox(self) -> list[Deposit]:
        return self.backend.pending_deposits(self.agent_id)

    def promote(self, deposit: Deposit, title: str = "", kind: str = "fact",
                tags: Sequence[str] = ()) -> str:
        embedding = self.embedder.embed((title + "\n" + deposit.content).strip())
        return self.backend.promote_deposit(deposit.id, title=title, kind=kind, tags=tags,
                                            embedding=embedding)

    def reject(self, deposit: Deposit) -> bool:
        return self.backend.reject_deposit(deposit.id)

    # ---- outcomes layer --------------------------------------------------

    def feedback(self, memory_id: str, helped: Optional[bool], query: Optional[str] = None,
                 note: Optional[str] = None) -> str:
        """Record an outcome. helped=None records a neutral, uncounted outcome."""
        outcome = "neutral" if helped is None else ("helped" if helped else "not_helped")
        return self.backend.record_outcome(memory_id, outcome, query=query, note=note)
