"""SupabaseBackend against a fake transport. No network is touched."""

import json

import pytest

from agent_memory import HashEmbedder, MemoryClient, RankingConfig, SupabaseBackend
from agent_memory.supabase_backend import SupabaseError

TS = "2026-01-15T12:00:00.123456+00:00"


class FakeTransport:
    def __init__(self):
        self.calls = []
        self.replies = []

    def reply(self, status, payload):
        self.replies.append((status, json.dumps(payload).encode()))

    def __call__(self, method, url, headers, body):
        self.calls.append({"method": method, "url": url, "headers": dict(headers),
                           "json": json.loads(body) if body else None})
        return self.replies.pop(0)


@pytest.fixture
def transport():
    return FakeTransport()


@pytest.fixture
def backend(transport):
    return SupabaseBackend("https://project.example.com/", "anon-key-not-real",
                           access_token="user-jwt-not-real", transport=transport)


def test_headers_carry_key_and_user_token(backend, transport):
    transport.reply(200, [])
    backend.active_rules("default")
    call = transport.calls[0]
    assert call["url"] == "https://project.example.com/rest/v1/rpc/active_rules"
    assert call["headers"]["apikey"] == "anon-key-not-real"
    assert call["headers"]["Authorization"] == "Bearer user-jwt-not-real"
    assert call["json"] == {"p_agent_id": "default"}


def test_key_is_used_as_bearer_when_no_token(transport):
    backend = SupabaseBackend("https://project.example.com", "service-key-not-real",
                              transport=transport)
    transport.reply(200, [])
    backend.active_rules("default")
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer service-key-not-real"


def test_recall_sends_every_ranking_parameter(backend, transport):
    transport.reply(200, [{
        "id": "m1", "kind": "fact", "title": "T", "content": "C", "tags": ["x"],
        "created_at": TS, "text_score": 0.25, "vector_score": 0.75,
        "helped": 2, "not_helped": 1, "score": 0.61}])
    cfg = RankingConfig(w_text=0.3, w_vector=0.7, outcome_weight=0.4,
                        recency_weight=0.1, half_life_days=30.0)
    hits = backend.recall("q", query_embedding=[0.1, 0.2], agent_id="a1",
                          match_count=3, candidate_pool=20, config=cfg)
    assert transport.calls[0]["url"].endswith("/rest/v1/rpc/recall_memories")
    assert transport.calls[0]["json"] == {
        "p_query": "q", "p_query_embedding": [0.1, 0.2], "p_agent_id": "a1",
        "p_match_count": 3, "p_candidate_pool": 20, "p_w_text": 0.3, "p_w_vector": 0.7,
        "p_outcome_weight": 0.4, "p_recency_weight": 0.1, "p_half_life_days": 30.0}
    assert hits[0].id == "m1" and hits[0].score == 0.61
    assert hits[0].created_at.year == 2026 and hits[0].tags == ["x"]


def test_add_memory_never_sends_owner_id(backend, transport):
    transport.reply(201, [{"id": "m1", "owner_id": "u1", "agent_id": "default",
                           "kind": "lesson", "title": "T", "content": "C", "tags": [],
                           "archived_at": None, "created_at": TS}])
    mem = backend.add_memory("default", "C", title="T", kind="lesson", embedding=[1.0])
    call = transport.calls[0]
    assert call["method"] == "POST"
    assert call["url"].startswith("https://project.example.com/rest/v1/memories?select=id")
    assert call["headers"]["Prefer"] == "return=representation"
    assert "owner_id" not in call["json"]  # the database default auth.uid() fills it
    assert call["json"]["embedding"] == [1.0]
    assert mem.owner_id == "u1" and mem.kind == "lesson"


def test_record_outcome_validates_before_calling(backend, transport):
    with pytest.raises(ValueError):
        backend.record_outcome("m1", "great")
    assert transport.calls == []
    transport.reply(200, "o1")
    assert backend.record_outcome("m1", "helped", query="q") == "o1"
    assert transport.calls[0]["json"] == {"p_memory_id": "m1", "p_outcome": "helped",
                                          "p_query": "q", "p_note": None}


def test_deposit_flow_calls(backend, transport):
    row = {"id": "d1", "owner_id": "u1", "agent_id": "default", "content": "raw",
           "status": "pending", "created_at": TS, "source": None,
           "promoted_memory_id": None, "resolved_at": None}
    transport.reply(201, [row])
    transport.reply(200, [row])
    transport.reply(200, "m9")
    transport.reply(200, True)
    deposit = backend.add_deposit("default", "raw")
    assert [d.id for d in backend.pending_deposits("default")] == ["d1"]
    assert backend.promote_deposit(deposit.id, title="T", tags=["a"]) == "m9"
    assert backend.reject_deposit("d2") is True
    assert "status=eq.pending" in transport.calls[1]["url"]
    assert transport.calls[2]["json"]["p_deposit_id"] == "d1"
    assert transport.calls[2]["json"]["p_tags"] == ["a"]


def test_archive_and_deactivate_use_filtered_patch(backend, transport):
    transport.reply(200, [{"id": "m1"}])
    transport.reply(200, [])
    assert backend.archive_memory("m1") is True
    assert backend.deactivate_rule("r1") is False
    assert transport.calls[0]["method"] == "PATCH"
    assert "id=eq.m1" in transport.calls[0]["url"]
    assert "archived_at=is.null" in transport.calls[0]["url"]


def test_errors_surface_the_server_message(backend, transport):
    transport.reply(403, {"message": "new row violates row-level security policy"})
    with pytest.raises(SupabaseError, match="row-level security") as err:
        backend.add_rule("default", "x")
    assert err.value.status == 403


def test_client_works_end_to_end_over_the_fake_transport(backend, transport):
    client = MemoryClient(backend, HashEmbedder(), agent_id="a1")
    transport.reply(200, [])
    assert client.recall("anything") == []
    sent = transport.calls[0]["json"]
    assert len(sent["p_query_embedding"]) == 384 and sent["p_agent_id"] == "a1"


def test_rest_path_can_point_at_a_bare_postgrest(transport):
    backend = SupabaseBackend("http://localhost:3000/", "key-not-real",
                              transport=transport, rest_path="")
    transport.reply(200, [])
    backend.active_rules("default")
    assert transport.calls[0]["url"] == "http://localhost:3000/rpc/active_rules"
