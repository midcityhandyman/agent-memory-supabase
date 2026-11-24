-- Write side helpers. All are security invoker, so row level security
-- decides what the caller may touch.

-- Standing rules for one agent, highest priority (lowest number) first.
create or replace function public.active_rules(p_agent_id text default 'default')
returns setof public.agent_rules
language sql
stable
security invoker
set search_path = public
as $$
  select r.*
  from public.agent_rules r
  where r.active and r.agent_id = p_agent_id
  order by r.priority, r.created_at, r.id
$$;

-- Turn a pending deposit into a curated memory in one transaction.
create or replace function public.promote_deposit(
  p_deposit_id  uuid,
  p_title       text default '',
  p_kind        text default 'fact',
  p_tags        text[] default '{}',
  p_embedding   extensions.vector(384) default null
)
returns uuid
language plpgsql
security invoker
set search_path = public, extensions
as $$
declare
  v_deposit   public.memory_deposits;
  v_memory_id uuid;
begin
  select * into v_deposit
  from public.memory_deposits d
  where d.id = p_deposit_id
  for update;

  if not found then
    raise exception 'deposit % not found', p_deposit_id using errcode = 'P0002';
  end if;
  if v_deposit.status <> 'pending' then
    raise exception 'deposit % is already %', p_deposit_id, v_deposit.status
      using errcode = 'P0001';
  end if;

  insert into public.memories (owner_id, agent_id, kind, title, content, tags, embedding)
  values (v_deposit.owner_id, v_deposit.agent_id, p_kind, coalesce(p_title, ''),
          v_deposit.content, coalesce(p_tags, '{}'), p_embedding)
  returning id into v_memory_id;

  update public.memory_deposits
  set status = 'promoted', promoted_memory_id = v_memory_id, resolved_at = now()
  where id = p_deposit_id;

  return v_memory_id;
end
$$;

create or replace function public.reject_deposit(p_deposit_id uuid)
returns boolean
language sql
security invoker
set search_path = public
as $$
  with changed as (
    update public.memory_deposits
    set status = 'rejected', resolved_at = now()
    where id = p_deposit_id and status = 'pending'
    returning 1
  )
  select exists (select 1 from changed)
$$;

-- Record whether a recalled memory helped. This is what feeds the
-- re-ranking in recall_memories.
create or replace function public.record_outcome(
  p_memory_id  uuid,
  p_outcome    text,
  p_query      text default null,
  p_note       text default null
)
returns uuid
language sql
security invoker
set search_path = public
as $$
  insert into public.memory_outcomes (memory_id, outcome, query, note)
  values (p_memory_id, p_outcome, p_query, p_note)
  returning id
$$;

-- Per memory feedback tallies, handy for dashboards and pruning.
create or replace view public.memory_outcome_stats
with (security_invoker = true) as
select m.id as memory_id,
       m.owner_id,
       m.agent_id,
       count(o.id) filter (where o.outcome = 'helped')     as helped,
       count(o.id) filter (where o.outcome = 'not_helped') as not_helped,
       count(o.id) filter (where o.outcome = 'neutral')    as neutral
from public.memories m
left join public.memory_outcomes o on o.memory_id = m.id
group by m.id, m.owner_id, m.agent_id;
