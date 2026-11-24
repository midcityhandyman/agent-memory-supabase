-- Recall: hybrid retrieval plus outcome aware re-ranking.
--
-- memory_blend_score is pure arithmetic and is mirrored line for line by
-- agent_memory.ranking.blend_score in the Python client. The integration
-- tests compare the two over a grid of inputs.

create or replace function public.memory_blend_score(
  p_text_score        double precision,
  p_vector_score      double precision,
  p_helped            integer,
  p_not_helped        integer,
  p_age_days          double precision,
  p_w_text            double precision default 0.4,
  p_w_vector          double precision default 0.6,
  p_outcome_weight    double precision default 0.5,
  p_prior_helped      double precision default 2.0,
  p_prior_not_helped  double precision default 2.0,
  p_recency_weight    double precision default 0.15,
  p_half_life_days    double precision default 90.0
)
returns double precision
language sql
immutable
as $$
  select coalesce(
    -- 1. relevance: weighted mean of the two similarity signals, each in [0, 1]
    (
      (p_w_text * least(greatest(coalesce(p_text_score, 0), 0), 1)
       + p_w_vector * least(greatest(coalesce(p_vector_score, 0), 0), 1))
      / nullif(p_w_text + p_w_vector, 0)
    )
    -- 2. outcomes: smoothed helped rate mapped to a multiplier around 1.
    --    No feedback gives exactly 1 when the two priors are equal.
    * (
      1 + p_outcome_weight * (
        2 * (
          (greatest(p_helped, 0) + p_prior_helped)
          / (greatest(p_helped, 0) + greatest(p_not_helped, 0)
             + p_prior_helped + p_prior_not_helped)
        ) - 1
      )
    )
    -- 3. recency: a mild exponential decay, never below 1 - recency_weight
    * (
      (1 - p_recency_weight)
      + p_recency_weight * power(0.5::double precision,
                                 greatest(coalesce(p_age_days, 0), 0) / p_half_life_days)
    ),
    0
  )
$$;

create or replace function public.recall_memories(
  p_query            text,
  p_query_embedding  extensions.vector(384) default null,
  p_agent_id         text default 'default',
  p_match_count      integer default 8,
  p_candidate_pool   integer default 50,
  p_w_text           double precision default 0.4,
  p_w_vector         double precision default 0.6,
  p_outcome_weight   double precision default 0.5,
  p_recency_weight   double precision default 0.15,
  p_half_life_days   double precision default 90.0
)
returns table (
  id            uuid,
  kind          text,
  title         text,
  content       text,
  tags          text[],
  created_at    timestamptz,
  text_score    double precision,
  vector_score  double precision,
  helped        integer,
  not_helped    integer,
  score         double precision
)
language sql
stable
security invoker
set search_path = public, extensions
as $$
  with q as (
    -- OR together the stemmed lexemes of the query. websearch_to_tsquery
    -- would AND them, which is too strict for recall over short notes.
    select (
      select string_agg(quote_literal(lexeme), ' | ')
      from unnest(tsvector_to_array(to_tsvector('english', coalesce(p_query, '')))) as lexeme
    )::tsquery as tsq
  ),
  text_hits as (
    select m.id,
           -- flag 32 maps the raw rank r to r / (r + 1), so it lands in [0, 1)
           ts_rank_cd(m.search_tsv, q.tsq, 32)::double precision as text_score
    from public.memories m, q
    where q.tsq is not null
      and m.archived_at is null
      and m.agent_id = p_agent_id
      and m.search_tsv @@ q.tsq
    order by text_score desc
    limit greatest(p_candidate_pool, 1)
  ),
  vector_hits as (
    select m.id,
           greatest(1 - (m.embedding <=> p_query_embedding), 0)::double precision as vector_score
    from public.memories m
    where p_query_embedding is not null
      and m.embedding is not null
      and m.archived_at is null
      and m.agent_id = p_agent_id
    order by m.embedding <=> p_query_embedding
    limit greatest(p_candidate_pool, 1)
  ),
  candidates as (
    select coalesce(t.id, v.id) as id,
           coalesce(t.text_score, 0) as text_score,
           coalesce(v.vector_score, 0) as vector_score
    from text_hits t
    full outer join vector_hits v on v.id = t.id
  ),
  tallies as (
    select o.memory_id,
           (count(*) filter (where o.outcome = 'helped'))::integer as helped,
           (count(*) filter (where o.outcome = 'not_helped'))::integer as not_helped
    from public.memory_outcomes o
    where o.memory_id in (select c.id from candidates c)
    group by o.memory_id
  )
  select m.id, m.kind, m.title, m.content, m.tags, m.created_at,
         c.text_score,
         c.vector_score,
         coalesce(t.helped, 0) as helped,
         coalesce(t.not_helped, 0) as not_helped,
         public.memory_blend_score(
           c.text_score,
           c.vector_score,
           coalesce(t.helped, 0),
           coalesce(t.not_helped, 0),
           extract(epoch from (now() - m.created_at)) / 86400.0,
           p_w_text,
           -- with no query embedding the vector weight drops out entirely
           case when p_query_embedding is null then 0 else p_w_vector end,
           p_outcome_weight,
           2.0,
           2.0,
           p_recency_weight,
           p_half_life_days
         ) as score
  from candidates c
  join public.memories m on m.id = c.id
  left join tallies t on t.memory_id = c.id
  order by score desc, m.created_at desc, m.id
  limit greatest(p_match_count, 0)
$$;
