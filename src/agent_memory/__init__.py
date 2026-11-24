"""Layered agent memory on Postgres and Supabase, with an offline reference."""

from .client import MemoryClient
from .embeddings import HashEmbedder, HttpEmbedder, embedder_from_env
from .memory_backend import InMemoryBackend, InMemoryDatabase
from .models import Deposit, Memory, Outcome, RecallHit, Rule
from .ranking import RankingConfig, blend_score, outcome_multiplier, recency_factor
from .supabase_backend import SupabaseBackend

__all__ = [
    "Deposit",
    "HashEmbedder",
    "HttpEmbedder",
    "InMemoryBackend",
    "InMemoryDatabase",
    "Memory",
    "MemoryClient",
    "Outcome",
    "RankingConfig",
    "RecallHit",
    "Rule",
    "SupabaseBackend",
    "blend_score",
    "embedder_from_env",
    "outcome_multiplier",
    "recency_factor",
]

__version__ = "0.1.0"
