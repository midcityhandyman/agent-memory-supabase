import json
import math

import pytest

from agent_memory.embeddings import (HashEmbedder, HttpEmbedder, cosine_similarity,
                                     embedder_from_env)


def test_hash_embedder_is_deterministic_and_normalised():
    emb = HashEmbedder()
    a = emb.embed("Confirm the appointment window by text the day before")
    b = emb.embed("Confirm the appointment window by text the day before")
    assert a == b
    assert len(a) == 384
    assert math.sqrt(sum(x * x for x in a)) == pytest.approx(1.0)


def test_empty_text_gives_zero_vector():
    vec = HashEmbedder().embed("")
    assert vec == [0.0] * 384
    assert cosine_similarity(vec, vec) == 0.0


def test_related_text_is_closer_than_unrelated_text():
    emb = HashEmbedder()
    query = emb.embed("how do we handle a rain delay on an exterior job")
    related = emb.embed("Exterior jobs: on a rain delay, reschedule within two days")
    unrelated = emb.embed("Supplier invoices are filed monthly in the finance folder")
    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated) + 0.1


def test_cosine_rejects_mismatched_dimensions():
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 0.0], [1.0])


def test_env_falls_back_to_hash_embedder_without_a_key():
    assert isinstance(embedder_from_env({}), HashEmbedder)
    assert isinstance(embedder_from_env({"EMBEDDINGS_API_KEY": "  "}), HashEmbedder)


def test_env_picks_http_embedder_with_a_key():
    emb = embedder_from_env({"EMBEDDINGS_API_KEY": "test-key-not-real",
                             "EMBEDDINGS_BASE_URL": "https://embeddings.example.com/v1/"})
    assert isinstance(emb, HttpEmbedder)
    assert emb.base_url == "https://embeddings.example.com/v1"


def test_http_embedder_request_shape_and_parsing():
    calls = []

    def transport(method, url, headers, body):
        calls.append((method, url, headers, json.loads(body)))
        return 200, json.dumps({"data": [{"embedding": [0.5] * 384}]}).encode()

    emb = HttpEmbedder("test-key-not-real", base_url="https://embeddings.example.com/v1",
                       transport=transport)
    assert emb.embed("hello") == [0.5] * 384
    method, url, headers, payload = calls[0]
    assert (method, url) == ("POST", "https://embeddings.example.com/v1/embeddings")
    assert headers["Authorization"] == "Bearer test-key-not-real"
    assert payload == {"model": "text-embedding-3-small", "input": "hello", "dimensions": 384}


def test_http_embedder_raises_on_error_and_wrong_width():
    bad_status = HttpEmbedder("k", transport=lambda *a: (500, b"{}"))
    with pytest.raises(RuntimeError, match="HTTP 500"):
        bad_status.embed("x")
    wrong_width = HttpEmbedder(
        "k", transport=lambda *a: (200, json.dumps({"data": [{"embedding": [1.0]}]}).encode()))
    with pytest.raises(RuntimeError, match="expected 384"):
        wrong_width.embed("x")
