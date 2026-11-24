"""Live checks against Postgres with pgvector.

Skipped unless DATABASE_URL is set. The database user must be able to
create roles and extensions (the default superuser of a throwaway local or
CI Postgres is fine). Never point this at a Supabase project: it applies
sql/dev/000_local_shim.sql.

Every test runs in a transaction that is rolled back.
"""

import itertools
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

DATABASE_URL = os.environ.get("DATABASE_URL")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL is not set"),
]

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from agent_memory import HashEmbedder, InMemoryDatabase, RankingConfig  # noqa: E402
from agent_memory.ranking import blend_score  # noqa: E402

USER_A = str(uuid.UUID(int=0xA))
USER_B = str(uuid.UUID(int=0xB))
EMB = HashEmbedder()


def vec(text):
    return "[" + ",".join(f"{x:.8f}" for x in EMB.embed(text)) + "]"


@pytest.fixture(scope="module")
def migrated():
    import psycopg
    from apply_sql import migration_files

    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        for path in migration_files(with_shim=True):
            conn.execute(path.read_text(encoding="utf-8"))


@pytest.fixture
def conn(migrated):
    import psycopg

    with psycopg.connect(DATABASE_URL) as connection:
        yield connection
        connection.rollback()


def become(conn, user_id, role="authenticated"):
    conn.execute("reset role")
    conn.execute("select set_config('request.jwt.claim.sub', %s, true)", (user_id or "",))
    conn.execute(f"set local role {role}")


def add_memory(conn, content, title="", embed=True, agent="default"):
    row = conn.execute(
        "insert into memories (agent_id, title, content, embedding) "
        "values (%s, %s, %s, %s::extensions.vector) returning id",
        (agent, title, content, vec(title + "\n" + content) if embed else None),
    ).fetchone()
    return row[0]


def recall(conn, query, embed=True, **params):
    args = {"p_query": query, "p_query_embedding": vec(query) if embed else None, **params}
    names = ", ".join(
        f"{k} => %({k})s" + ("::extensions.vector" if k == "p_query_embedding" else "")
        for k in args)
    cur = conn.execute(
        f"select id, title, text_score, vector_score, helped, not_helped, score "
        f"from recall_memories({names})", args)
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def test_blend_score_matches_python_over_a_grid(conn):
    grid = itertools.product(
        [0.0, 0.3, 0.97, 1.4, -0.2],   # text, including out of range values
        [0.0, 0.55, 1.0],              # vector
        [0, 1, 12],                    # helped
        [0, 2, 40],                    # not helped
        [0.0, 30.0, 400.0, -5.0],      # age in days
    )
    configs = [RankingConfig(), RankingConfig(w_text=1.0, w_vector=0.0),
               RankingConfig(w_text=0.0, w_vector=0.0),
               RankingConfig(outcome_weight=0.9, prior_helped=1.0, prior_not_helped=3.0,
                             recency_weight=0.5, half_life_days=14.0)]
    checked = 0
    for (t, v, h, n, age), cfg in itertools.product(grid, configs):
        got = conn.execute(
            "select memory_blend_score(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (t, v, h, n, age, cfg.w_text, cfg.w_vector, cfg.outcome_weight,
             cfg.prior_helped, cfg.prior_not_helped, cfg.recency_weight,
             cfg.half_life_days)).fetchone()[0]
        assert got == pytest.approx(blend_score(t, v, h, n, age, cfg), abs=1e-12)
        checked += 1
    assert checked == 5 * 3 * 3 * 3 * 4 * 4


def test_blend_score_sql_defaults_match_python_defaults(conn):
    got = conn.execute("select memory_blend_score(0.6, 0.8, 3, 1, 45)").fetchone()[0]
    assert got == pytest.approx(blend_score(0.6, 0.8, 3, 1, 45), abs=1e-12)


NOTES = [
    ("Rain rule", "Exterior painting stops when rain is forecast within four hours"),
    ("Deposit policy", "Take a 30 percent deposit before ordering materials"),
    ("Old deposit policy", "Take a 50 percent deposit before ordering materials"),
    ("Arrival texts", "Send a text before the crew arrives at the property"),
    ("Invoice filing", "Supplier invoices are filed monthly in the finance folder"),
    ("Warranty", "Labour is covered by a twelve month warranty"),
]


def test_vector_recall_order_and_scores_match_python_reference(conn):
    become(conn, USER_A)
    now = datetime(2026, 1, 15, tzinfo=timezone.utc)
    ref = InMemoryDatabase(clock=lambda: now).as_user(USER_A)
    sql_to_ref = {}
    for title, content in NOTES:
        sql_id = add_memory(conn, content, title)
        ref_mem = ref.add_memory("default", content, title=title,
                                 embedding=EMB.embed(title + "\n" + content))
        sql_to_ref[sql_id] = ref_mem.id
    votes = {"Deposit policy": ["helped"] * 3, "Old deposit policy": ["not_helped"] * 4,
             "Warranty": ["helped", "not_helped", "neutral"]}
    for sql_id, ref_id in sql_to_ref.items():
        title = conn.execute("select title from memories where id = %s", (sql_id,)).fetchone()[0]
        for outcome in votes.get(title, []):
            conn.execute("select record_outcome(%s, %s)", (sql_id, outcome))
            ref.record_outcome(ref_id, outcome)

    # Vector only on both sides: the offline text scorer is an approximation
    # of ts_rank_cd, so only the vector path is expected to agree exactly.
    cfg = RankingConfig(w_text=0.0, w_vector=1.0)
    for query in ["what deposit do we take before ordering materials",
                  "can we paint outside if rain is coming",
                  "is labour under warranty"]:
        got = [r for r in recall(conn, query, p_w_text=0.0, p_w_vector=1.0, p_match_count=6)
               if r["vector_score"] > 0]
        want = [h for h in ref.recall(query, EMB.embed(query), match_count=6, config=cfg)
                if h.vector_score > 0]
        assert [sql_to_ref[r["id"]] for r in got] == [h.id for h in want]
        for r, h in zip(got, want):
            assert r["vector_score"] == pytest.approx(h.vector_score, abs=1e-5)
            assert (r["helped"], r["not_helped"]) == (h.helped, h.not_helped)
            assert r["score"] == pytest.approx(h.score, abs=1e-5)


def test_outcomes_flip_the_order_of_near_duplicates(conn):
    become(conn, USER_A)
    stale = add_memory(conn, "Take a 50 percent deposit before ordering materials",
                       "Deposit policy")
    current = add_memory(conn, "Take a 30 percent deposit before ordering materials",
                         "Deposit policy")
    query = "what deposit do we take before ordering materials"
    assert {r["id"] for r in recall(conn, query)} == {stale, current}
    for _ in range(3):
        conn.execute("select record_outcome(%s, 'not_helped', %s)", (stale, query))
        conn.execute("select record_outcome(%s, 'helped', %s)", (current, query))
    after = recall(conn, query)
    assert [r["id"] for r in after] == [current, stale]
    assert (after[0]["helped"], after[1]["not_helped"]) == (3, 3)
    assert after[0]["score"] > after[1]["score"] * 1.5


def test_text_search_is_stemmed_or_matched_and_bounded(conn):
    become(conn, USER_A)
    title_hit = add_memory(conn, "Twelve months on labour", "Warranty terms", embed=False)
    body_hit = add_memory(conn, "Twelve months warranty on labour", "Terms", embed=False)
    add_memory(conn, "Parking is behind the workshop", "Parking", embed=False)

    rows = recall(conn, "warranties for the roof", embed=False)
    assert [r["id"] for r in rows] == [title_hit, body_hit]  # weight A beats weight B
    for r in rows:
        assert 0 < r["text_score"] < 1 and r["vector_score"] == 0
        # no query embedding: the text score carries the whole relevance term
        assert r["score"] == pytest.approx(r["text_score"], rel=1e-6)


@pytest.mark.parametrize("query", ["", "the and of", "   ", "'); drop table memories; --",
                                   "what's the 'deposit' policy? (50%) & more | stuff:* \\ !"])
def test_awkward_queries_never_raise(conn, query):
    become(conn, USER_A)
    add_memory(conn, "Take a deposit before ordering", "Deposit policy", embed=False)
    rows = recall(conn, query, embed=False)
    assert len(rows) in (0, 1)
    assert conn.execute("select count(*) from memories").fetchone()[0] == 1


def test_hybrid_returns_text_only_and_vector_only_candidates(conn):
    become(conn, USER_A)
    text_only = add_memory(conn, "Ladder inspection is due every quarter", embed=False)
    with_vec = add_memory(conn, "Quarterly ladder checks keep the crew safe")
    rows = recall(conn, "ladder inspection")
    by_id = {r["id"]: r for r in rows}
    assert set(by_id) == {text_only, with_vec}
    assert by_id[text_only]["vector_score"] == 0 and by_id[text_only]["text_score"] > 0
    assert by_id[with_vec]["vector_score"] > 0


def test_archived_and_other_agent_memories_are_excluded(conn):
    become(conn, USER_A)
    keep = add_memory(conn, "Phone tree instructions for the office line")
    gone = add_memory(conn, "Phone tree instructions, older copy")
    add_memory(conn, "Phone tree instructions for another agent", agent="other")
    conn.execute("update memories set archived_at = now() where id = %s", (gone,))
    assert [r["id"] for r in recall(conn, "phone tree instructions")] == [keep]
    assert len(recall(conn, "phone tree instructions", p_agent_id="other")) == 1
    assert len(recall(conn, "phone tree instructions", p_match_count=0)) == 0


def test_rls_hides_everything_from_another_user(conn):
    import psycopg

    become(conn, USER_A)
    memory = add_memory(conn, "The gate code for the storage yard is kept in the vault")
    conn.execute("insert into agent_rules (rule) values ('Owner A rule')")
    deposit = conn.execute(
        "insert into memory_deposits (content) values ('raw note') returning id").fetchone()[0]
    conn.execute("select record_outcome(%s, 'helped')", (memory,))
    assert len(recall(conn, "gate code storage yard")) == 1

    become(conn, USER_B)
    assert recall(conn, "gate code storage yard") == []
    for table in ("memories", "agent_rules", "memory_deposits", "memory_outcomes",
                  "memory_outcome_stats"):
        assert conn.execute(f"select count(*) from {table}").fetchone()[0] == 0
    assert conn.execute("select count(*) from active_rules()").fetchone()[0] == 0
    assert conn.execute("update memories set content = 'x' returning id").fetchall() == []
    assert conn.execute("delete from memories returning id").fetchall() == []

    # cannot vote on a memory you cannot see
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with conn.transaction():
            conn.execute("select record_outcome(%s, 'not_helped')", (memory,))
    # cannot write a row on behalf of someone else
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with conn.transaction():
            conn.execute("insert into memories (owner_id, content) values (%s, 'spoof')",
                         (USER_A,))
    # cannot promote someone else's deposit
    with pytest.raises(psycopg.errors.NoDataFound):
        with conn.transaction():
            conn.execute("select promote_deposit(%s)", (deposit,))

    become(conn, USER_A)
    assert conn.execute("select helped from memory_outcome_stats").fetchone()[0] == 1


def test_outcomes_are_append_only_and_anon_gets_nothing(conn):
    import psycopg

    become(conn, USER_A)
    memory = add_memory(conn, "Crew start time is seven in the morning")
    conn.execute("select record_outcome(%s, 'helped')", (memory,))
    for statement in ("update memory_outcomes set outcome = 'not_helped'",
                      "delete from memory_outcomes"):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with conn.transaction():
                conn.execute(statement)
    with pytest.raises(psycopg.errors.CheckViolation):
        with conn.transaction():
            conn.execute("select record_outcome(%s, 'amazing')", (memory,))

    become(conn, None, role="anon")
    for statement in ("select count(*) from memories",
                      "select * from recall_memories('crew start time')",
                      "select * from active_rules()"):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with conn.transaction():
                conn.execute(statement)


def test_deposit_promotion_and_rules(conn):
    import psycopg

    become(conn, USER_A)
    deposit = conn.execute(
        "insert into memory_deposits (content, source) values (%s, 'call notes') returning id",
        ("Customer prefers a text before the crew arrives",)).fetchone()[0]
    memory = conn.execute(
        "select promote_deposit(%s, 'Arrival texts', 'preference', array['comms'], "
        "%s::extensions.vector)", (deposit, vec("arrival texts"))).fetchone()[0]
    status, linked = conn.execute(
        "select status, promoted_memory_id from memory_deposits where id = %s",
        (deposit,)).fetchone()
    assert (status, linked) == ("promoted", memory)
    assert [r["id"] for r in recall(conn, "text before crew arrives")] == [memory]
    with pytest.raises(psycopg.errors.RaiseException, match="already promoted"):
        with conn.transaction():
            conn.execute("select promote_deposit(%s)", (deposit,))

    other = conn.execute(
        "insert into memory_deposits (content) values ('noise') returning id").fetchone()[0]
    assert conn.execute("select reject_deposit(%s)", (other,)).fetchone()[0] is True
    assert conn.execute("select reject_deposit(%s)", (other,)).fetchone()[0] is False

    conn.execute("insert into agent_rules (rule, priority) values ('second', 20), "
                 "('first', 10), ('off', 1)")
    conn.execute("update agent_rules set active = false where rule = 'off'")
    rules = [r[0] for r in conn.execute("select rule from active_rules()").fetchall()]
    assert rules == ["first", "second"]
