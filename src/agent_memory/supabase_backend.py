"""Supabase backend: the same operations over PostgREST.

Uses only the standard library. Every call carries the caller's key or JWT,
so row level security on the server decides what is visible. Pass a custom
transport to test without a network (see tests/test_supabase_backend.py).

The SQL this talks to lives in sql/. Apply it to your project first.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import quote, urlencode

from .embeddings import Transport, urllib_transport
from .models import MEMORY_KINDS, OUTCOMES, Deposit, Memory, RecallHit, Rule
from .ranking import DEFAULT_CONFIG, RankingConfig

MEMORY_COLUMNS = "id,owner_id,agent_id,kind,title,content,tags,archived_at,created_at"


class SupabaseError(RuntimeError):
    """A non 2xx response from PostgREST."""

    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


def _ts(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class SupabaseBackend:
    def __init__(self, url: str, key: str, access_token: Optional[str] = None,
                 transport: Transport = urllib_transport, rest_path: str = "/rest/v1"):
        """url: project URL. key: anon or service key. access_token: user JWT.

        With only the anon key the policies grant nothing, by design. Pass a
        signed in user's access_token. The service key bypasses row level
        security: recall then reads every owner's rows for the agent_id, and
        inserts fail because owner_id defaults to auth.uid(), which is null
        for that key. This client has no owner_id argument, so use the
        service key only for reads you mean to be unscoped.

        rest_path is where PostgREST is mounted. Supabase serves it under
        /rest/v1. Pass "" for a bare PostgREST server.
        """
        if not url or not key:
            raise ValueError("url and key are required")
        self.base = url.rstrip("/") + rest_path
        self._key = key
        self._token = access_token or key
        self._transport = transport

    # ---- plumbing --------------------------------------------------------

    def _request(self, method: str, path: str, payload: Any = None,
                 params: Optional[Mapping[str, str]] = None,
                 prefer: Optional[str] = None) -> Any:
        url = self.base + path
        if params:
            url += "?" + urlencode(params, quote_via=quote)
        headers = {
            "apikey": self._key,
            "Authorization": "Bearer " + self._token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if prefer:
            headers["Prefer"] = prefer
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        status, raw = self._transport(method, url, headers, body)
        text = raw.decode("utf-8") if raw else ""
        if status >= 300:
            try:
                message = json.loads(text).get("message", text)
            except (ValueError, AttributeError):
                message = text
            raise SupabaseError(status, message)
        return json.loads(text) if text else None

    def _rpc(self, name: str, args: Mapping[str, Any]) -> Any:
        return self._request("POST", "/rpc/" + name, payload=dict(args))

    def _insert(self, table: str, row: Mapping[str, Any], select: str = "*") -> dict:
        rows = self._request("POST", "/" + table, payload=dict(row),
                             params={"select": select}, prefer="return=representation")
        return rows[0]

    # ---- rules -----------------------------------------------------------

    @staticmethod
    def _rule(row: Mapping[str, Any]) -> Rule:
        return Rule(id=row["id"], owner_id=row["owner_id"], agent_id=row["agent_id"],
                    rule=row["rule"], priority=row["priority"], active=row["active"],
                    created_at=_ts(row["created_at"]), source=row.get("source"))

    def add_rule(self, agent_id: str, rule: str, priority: int = 100,
                 source: Optional[str] = None) -> Rule:
        return self._rule(self._insert("agent_rules", {
            "agent_id": agent_id, "rule": rule, "priority": priority, "source": source}))

    def active_rules(self, agent_id: str) -> list[Rule]:
        rows = self._rpc("active_rules", {"p_agent_id": agent_id})
        return [self._rule(r) for r in rows or []]

    def deactivate_rule(self, rule_id: str) -> bool:
        rows = self._request("PATCH", "/agent_rules", payload={"active": False},
                             params={"id": "eq." + rule_id, "active": "is.true",
                                     "select": "id"},
                             prefer="return=representation")
        return bool(rows)

    # ---- memories --------------------------------------------------------

    def add_memory(self, agent_id: str, content: str, title: str = "", kind: str = "fact",
                   tags: Sequence[str] = (),
                   embedding: Optional[Sequence[float]] = None) -> Memory:
        if kind not in MEMORY_KINDS:
            raise ValueError(f"kind must be one of {MEMORY_KINDS}")
        row = self._insert("memories", {
            "agent_id": agent_id, "kind": kind, "title": title, "content": content,
            "tags": list(tags),
            "embedding": list(embedding) if embedding is not None else None,
        }, select=MEMORY_COLUMNS)
        return Memory(id=row["id"], owner_id=row["owner_id"], agent_id=row["agent_id"],
                      kind=row["kind"], title=row["title"], content=row["content"],
                      tags=list(row.get("tags") or []), created_at=_ts(row["created_at"]),
                      embedding=list(embedding) if embedding is not None else None,
                      archived_at=_ts(row.get("archived_at")))

    def archive_memory(self, memory_id: str) -> bool:
        rows = self._request("PATCH", "/memories", payload={"archived_at": "now"},
                             params={"id": "eq." + memory_id, "archived_at": "is.null",
                                     "select": "id"},
                             prefer="return=representation")
        return bool(rows)

    # ---- deposits --------------------------------------------------------

    @staticmethod
    def _deposit(row: Mapping[str, Any]) -> Deposit:
        return Deposit(id=row["id"], owner_id=row["owner_id"], agent_id=row["agent_id"],
                       content=row["content"], status=row["status"],
                       created_at=_ts(row["created_at"]), source=row.get("source"),
                       promoted_memory_id=row.get("promoted_memory_id"),
                       resolved_at=_ts(row.get("resolved_at")))

    def add_deposit(self, agent_id: str, content: str, source: Optional[str] = None) -> Deposit:
        return self._deposit(self._insert("memory_deposits", {
            "agent_id": agent_id, "content": content, "source": source}))

    def pending_deposits(self, agent_id: str) -> list[Deposit]:
        rows = self._request("GET", "/memory_deposits", params={
            "agent_id": "eq." + agent_id, "status": "eq.pending",
            "order": "created_at.asc,id.asc", "select": "*"})
        return [self._deposit(r) for r in rows or []]

    def promote_deposit(self, deposit_id: str, title: str = "", kind: str = "fact",
                        tags: Sequence[str] = (),
                        embedding: Optional[Sequence[float]] = None) -> str:
        return self._rpc("promote_deposit", {
            "p_deposit_id": deposit_id, "p_title": title, "p_kind": kind,
            "p_tags": list(tags),
            "p_embedding": list(embedding) if embedding is not None else None})

    def reject_deposit(self, deposit_id: str) -> bool:
        return bool(self._rpc("reject_deposit", {"p_deposit_id": deposit_id}))

    # ---- outcomes --------------------------------------------------------

    def record_outcome(self, memory_id: str, outcome: str, query: Optional[str] = None,
                       note: Optional[str] = None) -> str:
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}")
        return self._rpc("record_outcome", {
            "p_memory_id": memory_id, "p_outcome": outcome,
            "p_query": query, "p_note": note})

    # ---- recall ----------------------------------------------------------

    def recall(self, query: str, query_embedding: Optional[Sequence[float]] = None,
               agent_id: str = "default", match_count: int = 8, candidate_pool: int = 50,
               config: RankingConfig = DEFAULT_CONFIG) -> list[RecallHit]:
        rows = self._rpc("recall_memories", {
            "p_query": query,
            "p_query_embedding": list(query_embedding) if query_embedding is not None else None,
            "p_agent_id": agent_id,
            "p_match_count": match_count,
            "p_candidate_pool": candidate_pool,
            "p_w_text": config.w_text,
            "p_w_vector": config.w_vector,
            "p_outcome_weight": config.outcome_weight,
            "p_recency_weight": config.recency_weight,
            "p_half_life_days": config.half_life_days,
        })
        return [RecallHit(
            id=r["id"], kind=r["kind"], title=r["title"], content=r["content"],
            tags=list(r.get("tags") or []), created_at=_ts(r["created_at"]),
            text_score=float(r["text_score"]), vector_score=float(r["vector_score"]),
            helped=int(r["helped"]), not_helped=int(r["not_helped"]),
            score=float(r["score"])) for r in rows or []]
