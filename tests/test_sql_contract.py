"""Static checks that keep the SQL and the Python reference in step.

These run with no database. The live checks are in tests/integration.
"""

import re
from dataclasses import fields
from pathlib import Path

import pytest

from agent_memory.embeddings import EMBEDDING_DIM
from agent_memory.models import MEMORY_KINDS, OUTCOMES
from agent_memory.ranking import RankingConfig

ROOT = Path(__file__).resolve().parent.parent
SQL = {p.name: p.read_text(encoding="utf-8") for p in sorted((ROOT / "sql").glob("*.sql"))}
TABLES = ("agent_rules", "memories", "memory_deposits", "memory_outcomes")


def test_expected_migrations_exist():
    assert list(SQL) == ["001_tables.sql", "002_indexes.sql", "003_recall.sql",
                         "004_write_functions.sql", "005_rls_policies.sql"]


@pytest.mark.parametrize("field", [f.name for f in fields(RankingConfig)])
def test_blend_defaults_match_python(field):
    blend = SQL["003_recall.sql"].split("create or replace function public.recall_memories")[0]
    match = re.search(rf"p_{field}\s+double precision default ([0-9.]+)", blend)
    assert match, f"p_{field} missing from memory_blend_score"
    assert float(match.group(1)) == getattr(RankingConfig(), field)


@pytest.mark.parametrize("field", ["w_text", "w_vector", "outcome_weight",
                                   "recency_weight", "half_life_days"])
def test_recall_defaults_match_python(field):
    recall = SQL["003_recall.sql"].split("create or replace function public.recall_memories")[1]
    match = re.search(rf"p_{field}\s+double precision default ([0-9.]+)", recall)
    assert match, f"p_{field} missing from recall_memories"
    assert float(match.group(1)) == getattr(RankingConfig(), field)


def test_embedding_width_matches_python():
    widths = set(re.findall(r"vector\((\d+)\)", "\n".join(SQL.values())))
    assert widths == {str(EMBEDDING_DIM)}


def test_check_constraints_match_python_enums():
    tables = SQL["001_tables.sql"]
    kinds = re.search(r"kind in \(([^)]*)\)", tables).group(1)
    outcomes = re.search(r"outcome in \(([^)]*)\)", tables).group(1)
    assert tuple(re.findall(r"'(\w+)'", kinds)) == MEMORY_KINDS
    assert tuple(re.findall(r"'(\w+)'", outcomes)) == OUTCOMES


@pytest.mark.parametrize("table", TABLES)
def test_every_table_has_rls_and_owner_policies(table):
    rls = SQL["005_rls_policies.sql"]
    assert re.search(rf"alter table public\.{table}\s+enable row level security", rls)
    assert f"create policy {table}_owner_select on public.{table}" in rls
    assert f"create policy {table}_owner_insert on public.{table}" in rls
    assert re.search(rf"owner_id\s+uuid not null default auth\.uid\(\)",
                     SQL["001_tables.sql"].split(f"public.{table} (")[1].split(");")[0])


def test_every_policy_is_scoped_to_authenticated_and_owner():
    policies = re.findall(r"create policy .*?;", SQL["005_rls_policies.sql"], flags=re.S)
    assert len(policies) == 14
    for policy in policies:
        assert "to authenticated" in policy
        assert "owner_id = (select auth.uid())" in policy


def test_outcomes_are_append_only_for_users():
    rls = SQL["005_rls_policies.sql"]
    assert "grant select, insert on public.memory_outcomes to authenticated" in rls
    assert "memory_outcomes_owner_update" not in rls
    assert "memory_outcomes_owner_delete" not in rls


def test_functions_are_security_invoker():
    body = SQL["003_recall.sql"] + SQL["004_write_functions.sql"]
    assert "security definer" not in body.lower()
    assert body.count("security invoker") >= 5


def test_no_em_dashes_anywhere_in_the_repo():
    skip = {".git", ".deps", ".venv", "__pycache__", ".pytest_cache"}
    for path in ROOT.rglob("*"):
        if path.is_file() and not skip.intersection(path.parts):
            if path.suffix in {".py", ".sql", ".md", ".yml", ".toml", ".txt", ".example"}:
                assert chr(0x2014) not in path.read_text(encoding="utf-8"), path
