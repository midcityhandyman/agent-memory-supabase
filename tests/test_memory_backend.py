from datetime import timedelta

import pytest

from agent_memory import HashEmbedder, MemoryClient, RankingConfig
from agent_memory.memory_backend import NotFound


def test_rules_come_back_in_priority_order(client):
    client.add_rule("Never quote a price without a site visit", priority=10)
    client.add_rule("Sign messages with the office line", priority=50)
    client.add_rule("Confirm appointments the day before", priority=20)
    assert [r.priority for r in client.boot()] == [10, 20, 50]


def test_deactivated_rule_is_not_loaded(client):
    rule = client.add_rule("Temporary holiday hours apply")
    assert client.backend.deactivate_rule(rule.id) is True
    assert client.backend.deactivate_rule(rule.id) is False
    assert client.boot() == []


def test_recall_finds_the_relevant_memory(client):
    client.remember("Exterior painting stops when rain is forecast within four hours",
                    title="Rain rule for exterior work", kind="procedure")
    client.remember("Supplier invoices are filed monthly", title="Invoice filing")
    hits = client.recall("can we paint outside if rain is coming")
    assert hits[0].title == "Rain rule for exterior work"
    assert hits[0].text_score > 0 and hits[0].vector_score > 0
    assert hits[0].score > 0


def test_recall_respects_limit_and_agent_namespace(db):
    a = MemoryClient(db.as_user("owner-a"), HashEmbedder(), agent_id="scheduler")
    b = MemoryClient(db.as_user("owner-a"), HashEmbedder(), agent_id="estimator")
    for i in range(6):
        a.remember(f"Scheduling note number {i} about the calendar")
    b.remember("Estimating note about the calendar")
    assert len(a.recall("calendar note", limit=4)) == 4
    assert len(a.recall("calendar note", limit=50)) == 6
    assert len(b.recall("calendar note", limit=50)) == 1


def test_owners_are_isolated_like_row_level_security(db):
    alice = MemoryClient(db.as_user("owner-a"), HashEmbedder())
    bob = MemoryClient(db.as_user("owner-b"), HashEmbedder())
    secret = alice.remember("The gate code for the storage yard is kept in the vault")
    alice.add_rule("Owner A rule")
    alice.deposit("Owner A raw note")

    assert bob.recall("gate code storage yard") == []
    assert bob.boot() == []
    assert bob.inbox() == []
    with pytest.raises(NotFound):
        bob.feedback(secret.id, helped=False)
    assert bob.forget(secret.id) is False
    assert len(alice.recall("gate code storage yard")) == 1


def test_feedback_reranks_two_near_duplicates(client, clock):
    stale = client.remember("Deposit policy: take 50 percent deposit before ordering materials",
                            title="Deposit policy")
    clock.advance(seconds=1)
    current = client.remember("Deposit policy: take 30 percent deposit before ordering materials",
                              title="Deposit policy")
    query = "what deposit do we take before ordering materials"

    before = client.recall(query)
    assert {h.id for h in before[:2]} == {stale.id, current.id}

    for _ in range(3):
        client.feedback(stale.id, helped=False, query=query)
        client.feedback(current.id, helped=True, query=query)

    after = client.recall(query)
    assert after[0].id == current.id
    assert after[1].id == stale.id
    assert (after[0].helped, after[0].not_helped) == (3, 0)
    assert (after[1].helped, after[1].not_helped) == (0, 3)
    assert after[0].score > after[1].score * 1.5


def test_neutral_feedback_does_not_change_scores(client):
    mem = client.remember("Crew start time is seven in the morning", title="Start time")
    before = client.recall("crew start time")[0].score
    client.feedback(mem.id, helped=None)
    assert client.recall("crew start time")[0].score == pytest.approx(before)


def test_invalid_outcome_and_kind_are_rejected(client):
    mem = client.remember("Valid memory")
    with pytest.raises(ValueError):
        client.backend.record_outcome(mem.id, "amazing")
    with pytest.raises(ValueError):
        client.remember("Bad kind", kind="rumour")
    with pytest.raises(ValueError):
        client.remember("   ")


def test_archived_memory_is_not_recalled(client):
    mem = client.remember("Old phone tree instructions", title="Phone tree")
    assert len(client.recall("phone tree")) == 1
    assert client.forget(mem.id) is True
    assert client.forget(mem.id) is False
    assert client.recall("phone tree") == []


def test_deposit_promote_and_reject_flow(client):
    keep = client.deposit("Customer prefers a text before the crew arrives", source="call notes")
    drop = client.deposit("Weather was nice today")
    assert [d.id for d in client.inbox()] == [keep.id, drop.id]
    assert client.recall("text before crew arrives") == []

    memory_id = client.promote(keep, title="Arrival texts", kind="preference", tags=["comms"])
    assert client.reject(drop) is True
    assert client.reject(drop) is False
    assert client.inbox() == []

    hits = client.recall("text before crew arrives")
    assert [h.id for h in hits] == [memory_id]
    assert hits[0].kind == "preference" and hits[0].tags == ["comms"]
    with pytest.raises(ValueError, match="already promoted"):
        client.promote(keep)


def test_text_only_recall_ignores_vectors(client):
    client.remember("Warranty covers labour for twelve months", title="Warranty")
    client.remember("Completely unrelated note about parking")
    hits = client.recall("warranty labour", use_vectors=False)
    assert len(hits) == 1
    assert hits[0].vector_score == 0.0
    # with no query embedding the text score carries the whole relevance term
    assert hits[0].score == pytest.approx(hits[0].text_score)


def test_memory_without_embedding_is_still_found_by_text(client):
    client.backend.add_memory("default", "Ladder inspection is due every quarter",
                              title="Ladder inspection")
    hits = client.recall("ladder inspection")
    assert len(hits) == 1 and hits[0].vector_score == 0.0 and hits[0].text_score > 0


def test_older_memory_scores_lower_all_else_equal(db, clock):
    client = MemoryClient(db.as_user("owner-a"), HashEmbedder())
    old = client.remember("Van service interval is every ten thousand miles")
    clock.advance(days=365)
    new = client.remember("Van service interval is every ten thousand miles")
    hits = client.recall("van service interval")
    assert [h.id for h in hits] == [new.id, old.id]
    assert hits[0].text_score == hits[1].text_score
    assert hits[0].score > hits[1].score


def test_custom_config_can_switch_off_outcomes(db):
    client = MemoryClient(db.as_user("owner-a"), HashEmbedder(),
                          config=RankingConfig(outcome_weight=0.0))
    mem = client.remember("Dump runs happen on Fridays")
    before = client.recall("dump runs")[0].score
    for _ in range(5):
        client.feedback(mem.id, helped=False)
    assert client.recall("dump runs")[0].score == pytest.approx(before)


def test_offline_and_from_env_constructors():
    offline = MemoryClient.offline()
    offline.remember("Works with no setup")
    assert len(offline.recall("works setup")) == 1
    from_env = MemoryClient.from_env({"AGENT_ID": "helper"})
    assert from_env.agent_id == "helper"
    assert type(from_env.backend).__name__ == "InMemoryBackend"
    remote = MemoryClient.from_env({"SUPABASE_URL": "https://project.example.com",
                                    "SUPABASE_KEY": "test-key-not-real"})
    assert type(remote.backend).__name__ == "SupabaseBackend"
