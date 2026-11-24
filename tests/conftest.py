from datetime import datetime, timedelta, timezone

import pytest

from agent_memory import HashEmbedder, InMemoryDatabase, MemoryClient

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)


class FakeClock:
    def __init__(self, now=NOW):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, **kwargs):
        self.now += timedelta(**kwargs)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def db(clock):
    counter = iter(range(1, 100000))
    return InMemoryDatabase(clock=clock, id_factory=lambda: f"id-{next(counter):05d}")


@pytest.fixture
def client(db):
    return MemoryClient(db.as_user("owner-a"), HashEmbedder(), agent_id="default")
