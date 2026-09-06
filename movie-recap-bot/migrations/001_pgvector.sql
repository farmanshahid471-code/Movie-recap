-- 001_pgvector.sql
-- Supabase / PostgreSQL + pgvector schema for Step D (semantic timestamp mapping).
--
-- Run this ONCE in the Supabase SQL editor (or the pipeline auto-runs it when
-- SUPABASE_DB_URL is configured). It creates:
--   1. the pgvector extension
--   2. jsonb -> vector casts (so the REST API can insert embeddings as arrays)
--   3. transcript_cues      — every dialogue cue of the movie, embedded
--   4. match_cues()         — the RPC the pipeline calls per narration line
--
-- NOTE: embeddings come from all-MiniLM-L6-v2, which produces 384 dimensions.
-- If you switch embedding models, recreate the table with the right vector size.

create extension if not exists vector;

-- PostgREST (the Supabase REST API) sends JSON arrays for vector columns and
-- function arguments. pgvector has no implicit json/jsonb -> vector cast, so
-- inserts/RPC calls would fail with "column is of type vector but expression
-- is of type jsonb". These inout casts fix that; harmless to re-run.
create cast (json as vector)  with inout as implicit;
create cast (jsonb as vector) with inout as implicit;

drop table if exists public.transcript_cues;
create table public.transcript_cues (
    id        bigserial primary key,
    idx       integer     not null,          -- order inside this movie session
    text      text        not null,
    start_ms  integer     not null,
    end_ms    integer     not null,
    session   text        not null default 'default',  -- movie/output name
    embedding vector(384) not null
);

create index on public.transcript_cues
    using hnsw (embedding vector_cosine_ops);

-- Cosine-similarity search used by recap/match.py::SupabaseVectorStore.
create or replace function public.match_cues(
    query_embedding vector(384),
    match_count    int default 3,
    match_threshold float default 0.0,
    session_name   text default 'default'
)
returns table (
    idx        integer,
    text       text,
    start_ms   integer,
    end_ms     integer,
    similarity float
)
language sql stable
as $$
    select
        c.idx,
        c.text,
        c.start_ms,
        c.end_ms,
        1 - (c.embedding <=> query_embedding) as similarity
    from public.transcript_cues c
    where c.session = session_name
      and 1 - (c.embedding <=> query_embedding) >= match_threshold
    order by c.embedding <=> query_embedding
    limit match_count;
$$;

-- Row-level security: the pipeline calls with the service_role key, which
-- bypasses RLS, but keep the policy sane for future anon access.
alter table public.transcript_cues enable row level security;

-- Grants for the service role are automatic; grant anon/authenticated read for
-- debugging through the REST API if ever needed:
-- grant select on public.transcript_cues to anon, authenticated;
-- The RPC is called with the service key; grant execute to the roles you use:
-- grant execute on function public.match_cues(vector(384), int, float, text)
--     to anon, authenticated;

