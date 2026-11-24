-- Row level security. One rule everywhere: a signed in user sees and
-- changes only rows whose owner_id is their own auth.uid().
--
-- anon gets nothing. service_role bypasses RLS on Supabase, so trusted
-- server code must set owner_id explicitly when it writes.

alter table public.agent_rules     enable row level security;
alter table public.memories        enable row level security;
alter table public.memory_deposits enable row level security;
alter table public.memory_outcomes enable row level security;

revoke all on public.agent_rules, public.memories,
              public.memory_deposits, public.memory_outcomes from anon;
revoke all on public.memory_outcome_stats from anon;

grant usage on schema public, extensions to authenticated, service_role;
grant select, insert, update, delete
  on public.agent_rules, public.memories, public.memory_deposits
  to authenticated, service_role;
-- Outcomes are an append only log for users: no update, no delete.
grant select, insert on public.memory_outcomes to authenticated;
grant select, insert, update, delete on public.memory_outcomes to service_role;
grant select on public.memory_outcome_stats to authenticated, service_role;

-- Functions are executable by PUBLIC by default. Take that away from anon.
revoke execute on function
  public.recall_memories(text, extensions.vector, text, integer, integer,
                         double precision, double precision, double precision,
                         double precision, double precision),
  public.active_rules(text),
  public.promote_deposit(uuid, text, text, text[], extensions.vector),
  public.reject_deposit(uuid),
  public.record_outcome(uuid, text, text, text)
from public, anon;
grant execute on function
  public.recall_memories(text, extensions.vector, text, integer, integer,
                         double precision, double precision, double precision,
                         double precision, double precision),
  public.active_rules(text),
  public.promote_deposit(uuid, text, text, text[], extensions.vector),
  public.reject_deposit(uuid),
  public.record_outcome(uuid, text, text, text)
to authenticated, service_role;

-- agent_rules
drop policy if exists agent_rules_owner_select on public.agent_rules;
create policy agent_rules_owner_select on public.agent_rules
  for select to authenticated using (owner_id = (select auth.uid()));
drop policy if exists agent_rules_owner_insert on public.agent_rules;
create policy agent_rules_owner_insert on public.agent_rules
  for insert to authenticated with check (owner_id = (select auth.uid()));
drop policy if exists agent_rules_owner_update on public.agent_rules;
create policy agent_rules_owner_update on public.agent_rules
  for update to authenticated
  using (owner_id = (select auth.uid()))
  with check (owner_id = (select auth.uid()));
drop policy if exists agent_rules_owner_delete on public.agent_rules;
create policy agent_rules_owner_delete on public.agent_rules
  for delete to authenticated using (owner_id = (select auth.uid()));

-- memories
drop policy if exists memories_owner_select on public.memories;
create policy memories_owner_select on public.memories
  for select to authenticated using (owner_id = (select auth.uid()));
drop policy if exists memories_owner_insert on public.memories;
create policy memories_owner_insert on public.memories
  for insert to authenticated with check (owner_id = (select auth.uid()));
drop policy if exists memories_owner_update on public.memories;
create policy memories_owner_update on public.memories
  for update to authenticated
  using (owner_id = (select auth.uid()))
  with check (owner_id = (select auth.uid()));
drop policy if exists memories_owner_delete on public.memories;
create policy memories_owner_delete on public.memories
  for delete to authenticated using (owner_id = (select auth.uid()));

-- memory_deposits
drop policy if exists memory_deposits_owner_select on public.memory_deposits;
create policy memory_deposits_owner_select on public.memory_deposits
  for select to authenticated using (owner_id = (select auth.uid()));
drop policy if exists memory_deposits_owner_insert on public.memory_deposits;
create policy memory_deposits_owner_insert on public.memory_deposits
  for insert to authenticated with check (owner_id = (select auth.uid()));
drop policy if exists memory_deposits_owner_update on public.memory_deposits;
create policy memory_deposits_owner_update on public.memory_deposits
  for update to authenticated
  using (owner_id = (select auth.uid()))
  with check (owner_id = (select auth.uid()));
drop policy if exists memory_deposits_owner_delete on public.memory_deposits;
create policy memory_deposits_owner_delete on public.memory_deposits
  for delete to authenticated using (owner_id = (select auth.uid()));

-- memory_outcomes: read your own, and only vote on a memory you can see.
-- The subquery runs under the caller's own policies on memories, so a
-- memory owned by someone else is invisible and the insert is refused.
drop policy if exists memory_outcomes_owner_select on public.memory_outcomes;
create policy memory_outcomes_owner_select on public.memory_outcomes
  for select to authenticated using (owner_id = (select auth.uid()));
drop policy if exists memory_outcomes_owner_insert on public.memory_outcomes;
create policy memory_outcomes_owner_insert on public.memory_outcomes
  for insert to authenticated
  with check (
    owner_id = (select auth.uid())
    and exists (select 1 from public.memories m where m.id = memory_id)
  );
