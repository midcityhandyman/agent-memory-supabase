# agent-memory-supabase

AI agents forget between sessions. Plain vector search does not fix that: it
keeps handing back the note that looks most similar, even after that note led
to a wrong answer.

This repo gives an agent a memory in your own Supabase or Postgres database
that gets better with use. The agent says whether a recalled note helped, and
the next recall ranks on that track record as well as on similarity.

It is for teams building agents on Supabase who want memory to live next to
their data, behind row level security, with no extra service to run.

What is in it:

- Five SQL migrations: four tables (standing rules, curated memories, an inbox
  of raw notes, a feedback log), their indexes, and owner only policies.
- One SQL function, `recall_memories`, that blends full text search, pgvector
  similarity, recorded outcomes and age into one explainable score.
- A Python client with no runtime dependencies, plus an in memory backend
  that runs the same ranking with no database.

## What you will see

`python demo/run_demo.py` needs no install, no keys and no network. Two notes
say nearly the same thing and one is out of date. From the real output
(`demo/output.txt`, checked on every CI run):

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

Similarity alone prefers the stale note. After three reports each way the
order flips, and the unrelated note stays where it was. This run is the
offline backend with invented notes: it shows the mechanism, not retrieval
quality on real data.

What has been checked, and how: 95 unit tests with no database, 16 tests of
the real SQL on Postgres 16 with pgvector, and 2 tests that drive the Python
client through a real PostgREST server with signed tokens. CI runs all of
them. Nothing here has been run against a hosted Supabase project; see
[Limitations](#limitations).

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
with the default weights the multiplier stays between 0.5 and 1.5. Feedback
reorders memories that are about equally relevant. The most it can do is a
factor of 3 between a perfect and a terrible track record, and a memory
with zero relevance stays at zero whatever its votes.

The Python package has two backends behind one client:

- `InMemoryBackend`: a pure Python reference of the same recall steps and
  the same arithmetic, so tests and the demo run with no database.
- `SupabaseBackend`: calls the SQL over PostgREST using only the standard
  library.

Text scores differ between the two (the offline text scorer is a stand in
for Postgres full text search). The blend arithmetic and the vector path are
tested to agree.

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
    B -->|Supabase| S[SupabaseBackend over PostgREST]
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

Needs Python 3.11 or newer. The demo runs straight from a clone:

```
python demo/run_demo.py
```

To use the client in your own code, install the package from the clone
(it is not on PyPI):

```
python -m pip install -e .
```

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

These steps are written from the SQL and the PostgREST tests. They have not
been run on a hosted project.

1. Run the files in `sql/` in order (`001` to `005`) in the SQL editor or as
   migrations. They need pgvector 0.8.0 or newer. Do not run
   `sql/dev/000_local_shim.sql`: it exists only so plain Postgres has
   `auth.uid()` and the Supabase roles.
2. Copy `.env.example` to `.env`, fill in `SUPABASE_URL`, `SUPABASE_KEY` (the
   anon key) and a signed in user's `SUPABASE_ACCESS_TOKEN`, and load it into
   the environment. Nothing in this package reads `.env` for you.
3. Use `MemoryClient.from_env()`. Set `EMBEDDINGS_API_KEY` to use a real
   embedding model (384 dimensions by default), or leave it empty to keep
   the offline hash embedder.

## Tests

```
python -m pip install pytest
python -m pytest
```

That runs the 95 unit tests with no database: the ranking, the text scorer,
the embedders, the in memory backend, the Supabase backend against a fake
transport, static checks that the SQL defaults and constraints agree with
the Python constants, and the demo against its captured output.

The 18 integration tests in `tests/integration` are skipped unless they are
pointed at a throwaway Postgres with pgvector, and for the last two a
PostgREST server in front of it (bash syntax; in PowerShell set the three
variables with `$env:NAME = "value"` first):

```
docker network create memory-net
docker run -d --name memory-pg --network memory-net -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16
python -m pip install pytest "psycopg[binary]"
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
python scripts/apply_sql.py --with-shim
docker run -d --name memory-rest --network memory-net -p 3000:3000 -e PGRST_DB_URI=postgres://postgres:postgres@memory-pg:5432/postgres -e PGRST_DB_ANON_ROLE=anon -e PGRST_DB_EXTRA_SEARCH_PATH=public,extensions -e PGRST_JWT_SECRET=local-test-secret-at-least-32-chars-long postgrest/postgrest:v16.6
export POSTGREST_URL=http://localhost:3000 POSTGREST_JWT_SECRET=local-test-secret-at-least-32-chars-long
python -m pytest tests/integration
```

`tests/integration/test_postgres.py` runs the SQL directly. It checks that
`memory_blend_score` equals the Python `blend_score` over a grid of 2160
inputs, that vector recall returns the same order and scores as the Python
reference, that outcomes flip the order of near duplicates, that awkward
query strings never raise, that vector recall still finds your rows when
another owner's rows crowd the shared HNSW index, and that the row level
security policies hold (another user sees nothing, cannot vote on or promote
rows they do not own, cannot spoof `owner_id`; outcomes are append only;
`anon` gets nothing).

`tests/integration/test_postgrest.py` runs a whole session through
`SupabaseBackend` over HTTP: rules, memories, recall, feedback, the inbox,
archive, then a second user and `anon` being refused. The tokens are signed
by the test with the local secret.

CI runs both suites on every push (`.github/workflows/ci.yml`).

## Demo

`python demo/run_demo.py` walks one agent through all four layers with
invented data, a fixed clock and no network: rules at boot, recall before
and after feedback, the inbox, a second owner seeing nothing, and archiving.
The full captured run is in `demo/output.txt`.

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
tests/                    unit tests, tests/integration for live Postgres and PostgREST
```

## Where this pattern runs

The pattern comes from assistants that handle scheduling, quoting and follow
up for two small service companies, and need to carry standing rules and
lessons from one session to the next. The code here is a clean rewrite with
synthetic data. It contains no data, prompts or configuration from those
systems, and no results from them are claimed here.

## Limitations

Not run against live services:

- No hosted Supabase project. The SQL is tested on plain Postgres 16 with
  pgvector 0.8, with `sql/dev/000_local_shim.sql` standing in for
  `auth.uid()` and the `anon`, `authenticated` and `service_role` roles.
  The client is tested through PostgREST v16.6 with tokens the test signs
  itself. Supabase Auth, the API gateway and whatever PostgREST version a
  hosted project runs are not exercised.
- No live embeddings endpoint. `HttpEmbedder` is tested only against a fake
  transport. It sends a `dimensions` field, which not every OpenAI
  compatible server accepts.
- No load. Nothing here measures latency, throughput or recall quality on
  real data, and the only corpus is a handful of invented notes.

Mocked or approximated:

- The offline text scorer is an approximation of Postgres full text search,
  not a port. Stemming and ranking differ, so text scores from the two
  backends are not equal. Only the blend arithmetic and the vector path are
  verified to match.
- `HashEmbedder` only knows shared words. It is for tests and demos. Use a
  real embedding model for real use, and if its width is not 384 change the
  `vector(384)` declarations in `sql/`.

By design, for now:

- With a query embedding, recall returns up to the limit even when nothing
  is closely related. There is no minimum score, so threshold on `score`
  yourself.
- Vector recall on a table shared by many owners depends on pgvector's
  iterative scan. It stops at `hnsw.max_scan_tuples` (20000 by default), so
  a small owner in a very large shared table can still get a short vector
  pool. Partitioning or per tenant indexes would be the next step.
- Outcome tallies are counted at query time over the candidate pool. That is
  simple and always current, and has not been measured at volume. At scale
  you would keep counters on the memory row.
- All feedback counts the same forever. There is no decay of old outcomes,
  no weighting by who reported them, and no per query context: a memory
  that helps for one kind of question and hurts for another gets one
  blended multiplier.
- Feedback is only as good as whoever records it. Nothing here decides
  whether a memory helped. The agent or a person has to call `feedback`.
- Ownership is per user (`owner_id = auth.uid()`). There are no shared or
  team scoped memories.
- The client takes one access token and never refreshes it. Supabase user
  tokens expire, so a long running agent has to build a new client.
- With the service key, row level security is bypassed: recall reads every
  owner's rows for the agent id, and inserts fail because `owner_id` is
  null. The client has no `owner_id` argument.
- Full text search is configured for English.

License: MIT, see the LICENSE file.
