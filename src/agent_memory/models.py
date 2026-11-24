"""Plain data records shared by every backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

MEMORY_KINDS = ("fact", "procedure", "preference", "lesson")
OUTCOMES = ("helped", "not_helped", "neutral")
DEPOSIT_STATUSES = ("pending", "promoted", "rejected")


@dataclass
class Rule:
    """A standing instruction. Always loaded at boot, never ranked."""

    id: str
    owner_id: str
    agent_id: str
    rule: str
    priority: int
    active: bool
    created_at: datetime
    source: Optional[str] = None


@dataclass
class Memory:
    """A curated long term memory. Ranked at recall time."""

    id: str
    owner_id: str
    agent_id: str
    kind: str
    title: str
    content: str
    created_at: datetime
    tags: list[str] = field(default_factory=list)
    embedding: Optional[list[float]] = None
    archived_at: Optional[datetime] = None


@dataclass
class Deposit:
    """A raw capture waiting to be promoted to a memory or rejected."""

    id: str
    owner_id: str
    agent_id: str
    content: str
    status: str
    created_at: datetime
    source: Optional[str] = None
    promoted_memory_id: Optional[str] = None
    resolved_at: Optional[datetime] = None


@dataclass
class Outcome:
    """One piece of feedback: did this memory help when it was recalled."""

    id: str
    owner_id: str
    memory_id: str
    outcome: str
    created_at: datetime
    query: Optional[str] = None
    note: Optional[str] = None


@dataclass
class RecallHit:
    """One ranked recall result, with the components that produced its score."""

    id: str
    kind: str
    title: str
    content: str
    tags: list[str]
    created_at: datetime
    text_score: float
    vector_score: float
    helped: int
    not_helped: int
    score: float
