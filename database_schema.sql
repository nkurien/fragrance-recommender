-- 1. Enable the pgvector extension (run as superuser/postgres)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create the fragrances table
CREATE TABLE IF NOT EXISTS fragrances (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    brand TEXT NOT NULL,
    gender TEXT,
    rating NUMERIC,
    top_notes TEXT,
    middle_notes TEXT,
    base_notes TEXT,
    main_accords TEXT, -- Comma-separated list of accords (e.g. "citrus, woody, sweet")
    embedding vector(384) -- 384-dimensional vector from all-MiniLM-L6-v2
);

-- 3. Create HNSW Index (Cosine Similarity)
-- Note: If you face memory limitations on database free tiers during index creation,
-- you can safely drop or skip this. At ~13k rows, queries are extremely fast even without it.
CREATE INDEX IF NOT EXISTS fragrances_hnsw_idx 
ON fragrances USING hnsw (embedding vector_cosine_ops);
