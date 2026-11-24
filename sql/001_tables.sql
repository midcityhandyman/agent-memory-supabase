-- Layered agent memory: tables.
--
-- Four layers, from most to least trusted:
--   agent_rules      standing instructions, always loaded, never ranked
--   memories         curated long term memory, ranked at recall time
--   memory_deposits  raw captures waiting to be promoted or rejected
--   memory_outcomes  feedback: did a recalled memory actually help
--
-- Every row carries owner_id, which row level security keys on
-- (see 005_rls_policies.sql). agent_id is a free text namespace inside one
-- owner, so one account can run several agents with separate memory.

create schema if not exists extensions;
create extension if not exists vector with schema extensions;
-- 003_recall.sql needs pgvector 0.8.0 or newer (hnsw.iterative_scan).

-- Embedding width. 384 matches small sentence embedding models and the
-- offline hash embedder in the Python client. Change it here and in
-- 003_recall.sql and 004_write_functions.sql if your model differs.

create table if not exists public.agent_rules (
  id          uuid primary key default gen_random_uuid(),
  owner_id    uuid not null default auth.uid(),
  agent_id    text not null default 'default',
  rule        text not null check (length(btrim(rule)) > 0),
  priority    integer not null default 100,
  active      boolean not null default true,
  source      text,
  created_at  timestamptz not null default now()
);

create table if not exists public.memories (
  id          uuid primary key default gen_random_uuid(),
  owner_id    uuid not null default auth.uid(),
  agent_id    text not null default 'default',
  kind        text not null default 'fact'
              check (kind in ('fact', 'procedure', 'preference', 'lesson')),
  title       text not null default '',
  content     text not null check (length(btrim(content)) > 0),
  tags        text[] not null default '{}',
  embedding   extensions.vector(384),
  search_tsv  tsvector generated always as (
                setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                setweight(to_tsvector('english', content), 'B')
              ) stored,
  archived_at timestamptz,
  created_at  timestamptz not null default now()
);

create table if not exists public.memory_deposits (
  id                 uuid primary key default gen_random_uuid(),
  owner_id           uuid not null default auth.uid(),
  agent_id           text not null default 'default',
  content            text not null check (length(btrim(content)) > 0),
  source             text,
  status             text not null default 'pending'
                     check (status in ('pending', 'promoted', 'rejected')),
  promoted_memory_id uuid references public.memories (id) on delete set null,
  created_at         timestamptz not null default now(),
  resolved_at        timestamptz
);

create table if not exists public.memory_outcomes (
  id          uuid primary key default gen_random_uuid(),
  owner_id    uuid not null default auth.uid(),
  memory_id   uuid not null references public.memories (id) on delete cascade,
  outcome     text not null check (outcome in ('helped', 'not_helped', 'neutral')),
  query       text,
  note        text,
  created_at  timestamptz not null default now()
);
