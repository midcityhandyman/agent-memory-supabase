"""SupabaseBackend over a real PostgREST server, in front of the same SQL.

This is the request path Supabase uses: HTTP, a signed JWT, PostgREST
switching to the role in the token, row level security deciding the rest.
It is still not a hosted Supabase project (no GoTrue, no gateway).

Skipped unless DATABASE_URL, POSTGREST_URL and POSTGREST_JWT_SECRET are all
set. Unlike test_postgres.py these requests commit, so the test uses fresh
random owners and deletes their rows afterwards.
"""

import base64
import hashlib
import hmac
import json
import os
import sys
import time
import uuid
from pathlib import Path

import pytest

DATABASE_URL = os.environ.get("DATABASE_URL")
POSTGREST_URL = os.environ.get("POSTGREST_URL")
JWT_SECRET = os.environ.get("POSTGREST_JWT_SECRET")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (DATABASE_URL and POSTGREST_URL and JWT_SECRET),
                       reason="DATABASE_URL, POSTGREST_URL and POSTGREST_JWT_SECRET are not all set"),
]

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from agent_memory import HashEmbedder, MemoryClient, SupabaseBackend  # noqa: E402
from agent_memory.supabase_backend import SupabaseError  # noqa: E402

QUERY = "what deposit do we take before ordering materials"


def _b64(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=")


def sign(sub, role):
    """An HS256 token shaped like the ones Supabase issues: sub, role, exp."""
    head = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = _b64(json.dumps({"sub": sub, "role": role, "exp": int(time.time()) + 300}).encode())
    mac = hmac.new(JWT_SECRET.encode(), head + b"." + body, hashlib.sha256).digest()
    return (head + b"." + body + b"." + _b64(mac)).decode()


def client(sub, agent_id, role="authenticated"):
    backend = SupabaseBackend(POSTGREST_URL, "unused-by-bare-postgrest",
                              access_token=sign(sub, role), rest_path="")
    return MemoryClient(backend, HashEmbedder(), agent_id=agent_id)


@pytest.fixture(scope="module")
def owners():
    import psycopg
    from apply_sql import migration_files

    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        for path in migration_files(with_shim=True):
            conn.execute(path.read_text(encoding="utf-8"))
        conn.execute("notify pgrst, 'reload schema'")
    time.sleep(1)  # let PostgREST pick up the reload
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    yield ids
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        for table in ("memory_outcomes", "memory_deposits", "memories", "agent_rules"):
            conn.execute(f"delete from {table} where owner_id = any(%s::uuid[])", (ids,))


def test_full_session_over_http(owners):
    agent_id = "http-" + uuid.uuid4().hex[:8]
    a = client(owners[0], agent_id)

    rule = a.add_rule("Quote prices only from the current price sheet", priority=10)
    assert rule.owner_id == owners[0]          # filled in by auth.uid() on the server
    assert [r.rule for r in a.boot()] == [rule.rule]

    stale = a.remember("Take a 50 percent deposit before ordering materials for any job.",
                       title="Deposit policy (2024 sheet)", kind="procedure", tags=["billing"])
    current = a.remember("Take a 30 percent deposit before ordering materials. Jobs under "
                         "500 dollars need no deposit.",
                         title="Deposit policy (current)", kind="procedure", tags=["billing"])
    before = a.recall(QUERY, limit=3)
    assert {h.id for h in before} == {stale.id, current.id}
    assert all(h.text_score > 0 and h.vector_score > 0 and h.tags == ["billing"] for h in before)

    for _ in range(3):
        a.feedback(stale.id, helped=False, query=QUERY)
        a.feedback(current.id, helped=True, query=QUERY)
    after = a.recall(QUERY, limit=3)
    assert [h.id for h in after] == [current.id, stale.id]
    assert (after[0].helped, after[1].not_helped) == (3, 3)
    assert [h.id for h in a.recall(QUERY, use_vectors=False)] == [current.id, stale.id]

    note = a.deposit("Customer at the corner unit prefers a text before the crew arrives.",
                     source="call notes")
    noise = a.deposit("Printer was out of paper again.", source="chat")
    assert [d.id for d in a.inbox()] == [note.id, noise.id]
    promoted = a.promote(note, title="Arrival text preference", kind="preference")
    assert a.reject(noise) is True and a.reject(noise) is False
    assert a.inbox() == []
    assert a.recall("text before crew arrives", limit=1)[0].id == promoted
    with pytest.raises(SupabaseError, match="already promoted"):
        a.promote(note)

    assert a.forget(stale.id) is True and a.forget(stale.id) is False
    assert stale.id not in [h.id for h in a.recall(QUERY)]
    assert a.backend.deactivate_rule(rule.id) is True
    assert a.boot() == []

    with pytest.raises(SupabaseError) as blank:
        a.remember("   ")
    assert blank.value.status == 400


def test_another_user_and_anon_get_nothing_over_http(owners):
    agent_id = "http-" + uuid.uuid4().hex[:8]
    a = client(owners[0], agent_id)
    a.add_rule("Owner A rule")
    memory = a.remember("The gate code for the storage yard is kept in the vault")
    a.deposit("raw note")
    assert len(a.recall("gate code storage yard")) == 1

    b = client(owners[1], agent_id)
    assert b.boot() == [] and b.inbox() == []
    assert b.recall("gate code storage yard") == []
    assert b.forget(memory.id) is False
    with pytest.raises(SupabaseError) as vote:
        b.feedback(memory.id, helped=False)
    assert vote.value.status == 403 and "row-level security" in str(vote.value)

    anon = client(owners[0], agent_id, role="anon")
    for call in (lambda: anon.recall("gate code"), anon.boot, anon.inbox,
                 lambda: anon.remember("spoof")):
        with pytest.raises(SupabaseError) as denied:
            call()
        assert denied.value.status in (401, 403)
