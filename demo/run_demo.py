"""Offline demo: one agent session across all four memory layers.

Runs against the in memory backend with the hash embedder, a fixed clock and
counter ids, so the output is identical on every run. No network, no keys.
All data is invented.

    python demo/run_demo.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_memory import HashEmbedder, InMemoryDatabase, MemoryClient  # noqa: E402
from agent_memory.ranking import outcome_multiplier  # noqa: E402


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 15, 9, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


def heading(text: str) -> None:
    print()
    print(text)
    print("-" * len(text))


def show(hits) -> None:
    print(f"  {'#':<2} {'score':>6}  {'text':>5}  {'vec':>5}  {'helped':>6}  {'not':>3}  title")
    for rank, hit in enumerate(hits, start=1):
        print(f"  {rank:<2} {hit.score:>6.3f}  {hit.text_score:>5.2f}  {hit.vector_score:>5.2f}"
              f"  {hit.helped:>6}  {hit.not_helped:>3}  {hit.title}")


def main() -> None:
    clock = Clock()
    counter = iter(range(1, 10_000))
    db = InMemoryDatabase(clock=clock, id_factory=lambda: f"row-{next(counter):04d}")
    agent = MemoryClient(db.as_user("owner-a"), HashEmbedder(), agent_id="office-assistant")

    print("agent-memory-supabase demo (offline, in memory backend, synthetic data)")

    heading("1. Rules layer: standing instructions, loaded at boot")
    agent.add_rule("Never promise a start date without checking the calendar", priority=10)
    agent.add_rule("Quote prices only from the current price sheet", priority=20)
    agent.add_rule("Keep customer messages under five sentences", priority=50)
    for rule in agent.boot():
        print(f"  [{rule.priority:>3}] {rule.rule}")

    heading("2. Memory layer: curated memories")
    agent.remember("Take a 50 percent deposit before ordering materials for any job.",
                   title="Deposit policy (2024 sheet)", kind="procedure", tags=["billing"])
    clock.advance(days=20)
    agent.remember("Take a 30 percent deposit before ordering materials. Jobs under "
                   "500 dollars need no deposit.",
                   title="Deposit policy (current)", kind="procedure", tags=["billing"])
    agent.remember("Exterior painting stops when rain is forecast within four hours. "
                   "Reschedule within two working days.",
                   title="Rain rule for exterior work", kind="procedure", tags=["scheduling"])
    agent.remember("Supplier invoices are filed monthly in the finance folder.",
                   title="Invoice filing", kind="fact", tags=["billing"])
    agent.remember("Labour carries a twelve month warranty. Materials follow the maker's terms.",
                   title="Warranty terms", kind="fact")
    clock.advance(days=1)
    print(f"  stored {len(db.memories)} memories for agent '{agent.agent_id}'")

    query = "what deposit do we take before ordering materials"
    heading(f'3. Recall before any feedback: "{query}"')
    before = agent.recall(query, limit=3)
    show(before)
    print("  Both deposit policies match on words and on vectors. With no feedback the")
    print("  out of date one wins on similarity alone.")

    heading("4. Outcomes layer: the agent reports what actually helped")
    stale = next(h for h in before if "2024" in h.title)
    current = next(h for h in before if "current" in h.title)
    for day in range(3):
        clock.advance(days=1)
        agent.feedback(stale.id, helped=False, query=query,
                       note="customer was quoted the old percentage")
        agent.feedback(current.id, helped=True, query=query)
        print(f"  day {day + 1}: '{stale.title}' not_helped, '{current.title}' helped")
    print(f"  outcome multiplier, stale:   {outcome_multiplier(0, 3):.3f}")
    print(f"  outcome multiplier, current: {outcome_multiplier(3, 0):.3f}")

    heading("5. Recall after feedback: same query, re-ranked")
    after = agent.recall(query, limit=3)
    show(after)
    gap_before = before[0].score - before[1].score
    gap_after = after[0].score - after[1].score
    print(f"  gap between the top two results: {gap_before:.3f} before, {gap_after:.3f} after")

    heading("6. Deposit layer: raw notes wait in an inbox until promoted")
    note = agent.deposit("Customer at the corner unit prefers a text before the crew arrives.",
                         source="call notes")
    noise = agent.deposit("Printer was out of paper again.", source="chat")
    print(f"  inbox holds {len(agent.inbox())} deposits")
    found = any(h.content == note.content for h in agent.recall("text before crew arrives"))
    print(f"  is the raw note recalled before promotion? {'yes' if found else 'no'}")
    agent.promote(note, title="Arrival text preference", kind="preference", tags=["comms"])
    agent.reject(noise)
    print(f"  promoted 1, rejected 1, inbox now holds {len(agent.inbox())}")
    show(agent.recall("text before crew arrives", limit=1))

    heading("7. Isolation: a second owner sees none of it")
    other = MemoryClient(db.as_user("owner-b"), HashEmbedder(), agent_id="office-assistant")
    print(f"  owner-b rules:   {len(other.boot())}")
    print(f"  owner-b recall:  {len(other.recall(query))}")
    print(f"  owner-b inbox:   {len(other.inbox())}")
    print("  On Supabase the same boundary is enforced by the policies in "
          "sql/005_rls_policies.sql.")

    heading("8. Archive: forgetting without deleting")
    agent.forget(stale.id)
    show(agent.recall(query, limit=3))
    print()
    print("done")


if __name__ == "__main__":
    main()
