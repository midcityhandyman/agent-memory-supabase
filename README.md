# agent-memory-supabase

Agents forget between sessions, and plain vector search keeps handing back
the note that looked similar last time even when it led to a wrong answer.
This is a small layered memory on Postgres and Supabase whose recall learns
from recorded outcomes: did this memory actually help.

## What it does

Four layers, each a table with row level security:

| Layer | Table | Purpose |
| --- | --- | --- |
| Rules | `agent_rules` | Standing instructions. Loaded at boot, never ranked. |
| Memories | `memories` | Curated long term memory with full text and pgvector indexes. |
| Deposits | `memory_deposits` | Raw captures in an inbox until promoted or rejected. |
| Outcomes | `memory_outcomes` | Append only feedback: helped, not_helped or neutral. |

Recall is one SQL function, `recall_memories`. It:

1. pulls a candidate pool by full text (`ts_rank_cd` over a weighted
   `tsvector`) and another by pgvector cosine similarity,
2. joins the two pools and tallies the outcomes recorded for each candidate,
3. scores every candidate with `memory_blend_score` and returns the top rows
   together with the components of the score.

```
score = relevance * outcome_multiplier * recency_factor

relevance          = (w_text * text + w_vector * vector) / (w_text + w_vector)
outcome_multiplier = 1 + outcome_weight * (2 * (helped + 2) / (helped + not_helped + 4) - 1)
recency_factor     = (1 - recency_weight) + recency_weight * 0.5 ^ (age_days / half_life_days)
```

The helped rate is smoothed with a 2 and 2 prior, so a memory with no
feedback has a multiplier of exactly 1, one vote moves it by only 0.1, and
the multiplier always stays between 0.5 and 1.5. Feedback can reorder
memories that are about equally relevant. It cannot drag an irrelevant
memory to the top.

The Python package has two backends behind one client:

- `InMemoryBackend`: a pure Python reference of the same recall steps and
  the same arithmetic, so tests and the demo run with no database.
- `SupabaseBackend`: calls the SQL over PostgREST using only the standard
  library.

There are no runtime dependencies.

## Architecture

```mermaid
flowchart TD
    A[Agent] --> C[MemoryClient]
    C --> E{Embedder}
    E -->|no key| H[HashEmbedder, offline]
    E -->|key set| X[HTTP embeddings endpoint]
    C --> B{Backend}
    B -->|tests and demo| M[InMemoryBackend]
    B -->|production| S[SupabaseBackend over PostgREST]
    M --> R[ranking.py reference ranking]
    S --> F[recall_memories SQL function]
    F --> T1[text pool: tsvector and ts_rank_cd]
    F --> T2[vector pool: pgvector cosine]
    F --> T3[outcome tallies]
    T1 --> K[memory_blend_score]
    T2 --> K
    T3 --> K
    R -. "same arithmetic, checked in CI" .- K
    subgraph PG["Postgres with row level security"]
        F
        T1
        T2
        T3
        K
        D[(agent_rules, memories, memory_deposits, memory_outcomes)]
    end
    F --> D
```

## Run it

Needs Python 3.11 or newer. Nothing to install for the demo.

```
python demo/run_demo.py
```

To use it in your own code:

```python
from agent_memory import MemoryClient

memory = MemoryClient.offline()          # or MemoryClient.from_env() for Supabase
memory.add_rule("Quote prices only from the current price sheet", priority=10)
memory.remember("Take a 30 percent deposit before ordering materials.", title="Deposit policy")

rules = memory.boot()
hits = memory.recall("what deposit do we take")
memory.feedback(hits[0].id, helped=True, query="what deposit do we take")
```

### On Supabase

1. Run the files in `sql/` in order (`001` to `005`) in the SQL editor or as
   migrations. Do not run `sql/dev/000_local_shim.sql`: it exists only so
   plain Postgres has `auth.uid()` and the Supabase roles.
2. Copy `.env.example` to `.env`, fill in `SUPABASE_URL`, `SUPABASE_KEY` and
   a signed in user's `SUPABASE_ACCESS_TOKEN`, and load it into the
   environment.
3. Use `MemoryClient.from_env()`. Set `EMBEDDINGS_API_KEY` to use a real
   embedding model (384 dimensions by default), or leave it empty to keep
   the offline hash embedder.

## Tests

```
python -m pip install pytest
python -m pytest
```

That runs the unit tests with no database: the ranking, the text scorer,
the embedders, the in memory backend, the Supabase backend against a fake
transport, static checks that the SQL defaults and constraints agree with
the Python constants, and the demo against its captured output.

The integration tests in `tests/integration` run the real SQL. They are
skipped unless `DATABASE_URL` points at a throwaway Postgres with pgvector:

```
docker run -d --name memory-pg -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16
python -m pip install pytest "psycopg[binary]"
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres python -m pytest tests/integration
```

They check that `memory_blend_score` equals the Python `blend_score` over a
grid of 2160 inputs, that vector recall returns the same order and scores as
the Python reference, that outcomes flip the order of near duplicates, that
awkward query strings never raise, and that the row level security policies
hold (another user sees nothing, cannot vote on or promote rows they do not
own, cannot spoof `owner_id`; outcomes are append only; `anon` gets nothing).
CI runs both suites (`.github/workflows/ci.yml`).

## Demo

`python demo/run_demo.py` walks one agent through all four layers with
invented data, a fixed clock and no network. The captured run is in
`demo/output.txt`. The core of it:

```
3. Recall before any feedback: "what deposit do we take before ordering materials"
  #   score   text    vec  helped  not  title
  1   0.739   0.88   0.67       0    0  Deposit policy (2024 sheet)
  2   0.713   0.88   0.60       0    0  Deposit policy (current)
  3   0.141   0.21   0.09       0    0  Warranty terms

5. Recall after feedback: same query, re-ranked
  #   score   text    vec  helped  not  title
  1   0.863   0.88   0.60       3    0  Deposit policy (current)
  2   0.579   0.88   0.67       0    3  Deposit policy (2024 sheet)
  3   0.140   0.21   0.09       0    0  Warranty terms
```

Two near duplicate notes, one out of date. Similarity alone prefers the
wrong one. After three reports each way the order flips, and the unrelated
note stays where it was.

## Layout

```
sql/                      migrations 001 to 005, plus dev/ shim for plain Postgres
src/agent_memory/
  ranking.py              reference ranking, mirrors memory_blend_score
  textsearch.py           offline stand in for Postgres full text ranking
  embeddings.py           HashEmbedder (offline) and HttpEmbedder
  memory_backend.py       in memory tables with owner scoping
  supabase_backend.py     PostgREST calls
  client.py               MemoryClient
scripts/apply_sql.py      apply the migrations to DATABASE_URL
demo/                     offline demo and its captured output
tests/                    unit tests, tests/integration for live Postgres
```

## Where this pattern runs

This is drawn from systems that run two small service companies day to day,
where assistants handle scheduling, quoting and follow up and need to carry
standing rules and hard won lessons from one session to the next. The code
here is a clean rewrite of the pattern with synthetic data. It contains no
data, prompts or configuration from those systems.

## Limitations

- The offline text scorer is an approximation of Postgres full text search,
  not a port. Stemming and ranking differ, so text scores from the two
  backends are not equal. Only the blend arithmetic and the vector path are
  verified to match.
- `HashEmbedder` only knows shared words. It is for tests and demos. Use a
  real embedding model in production, and if its width is not 384 change
  the `vector(384)` declarations in `sql/`.
- Outcome tallies are counted at query time over the candidate pool. That is
  simple and always current, and fine for thousands of outcomes per memory.
  At larger scale you would keep counters on the memory row.
- All feedback counts the same forever. There is no decay of old outcomes,
  no weighting by who reported them, and no per query context: a memory
  that helps for one kind of question and hurts for another gets one
  blended multiplier.
- Feedback is only as good as whoever records it. Nothing here decides
  whether a memory helped. The agent or a person has to call `feedback`.
- Ownership is per user (`owner_id = auth.uid()`). There are no shared or
  team scoped memories.
- `SupabaseBackend` is tested against a fake transport and the SQL is tested
  against plain Postgres with a shim for `auth.uid()`. Neither has been run
  against a hosted Supabase project as part of this repository's tests.
- Full text search is configured for English.

License: MIT, see the LICENSE file.
