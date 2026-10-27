-- 1. Enable the pgvector extension (run as superuser/postgres)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create the fragrances table
CREATE TABLE IF NOT EXISTS fragrances (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    brand TEXT NOT NULL,
    gender TEXT,
    rating NUMERIC,
    rating_count INTEGER,
    year INTEGER,
    top_notes TEXT,
    middle_notes TEXT,
    base_notes TEXT,
    main_accords TEXT, -- Comma-separated list of accords (e.g. "citrus, woody, sweet")
    url TEXT,          -- Fragrantica product page URL
    embedding vector(384) -- 384-dimensional vector from all-MiniLM-L6-v2
);

-- 3. Create HNSW Index (Cosine Similarity)
-- Note: If you face memory limitations on database free tiers during index creation,
-- you can safely drop or skip this. At ~72k rows, queries are slower without it.
CREATE INDEX IF NOT EXISTS fragrances_hnsw_idx
ON fragrances USING hnsw (embedding vector_cosine_ops);

-- Also created by scripts/embed_and_upload.py before it reloads the table.
CREATE UNIQUE INDEX IF NOT EXISTS fragrances_name_brand_key ON fragrances (name, brand);

-- 4. LLM usage log (also created automatically at backend startup)
CREATE TABLE IF NOT EXISTS usage_log (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    stage TEXT NOT NULL,              -- 'rewrite' | 'recommend'
    model TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,        -- includes reasoning tokens
    latency_ms INTEGER,
    limit_requests INTEGER,           -- Groq x-ratelimit-* headers at the time of the call
    remaining_requests INTEGER,
    limit_tokens INTEGER,
    remaining_tokens INTEGER
);
CREATE INDEX IF NOT EXISTS usage_log_ts_idx ON usage_log (ts);
