-- Indexes for the recall path and for the common list queries.

create index if not exists memories_search_tsv_idx
  on public.memories using gin (search_tsv);

create index if not exists memories_embedding_idx
  on public.memories using hnsw (embedding extensions.vector_cosine_ops);

create index if not exists memories_owner_agent_idx
  on public.memories (owner_id, agent_id)
  where archived_at is null;

create index if not exists memory_outcomes_memory_idx
  on public.memory_outcomes (memory_id, outcome);

create index if not exists memory_deposits_pending_idx
  on public.memory_deposits (owner_id, agent_id, created_at)
  where status = 'pending';

create index if not exists agent_rules_active_idx
  on public.agent_rules (owner_id, agent_id, priority)
  where active;
