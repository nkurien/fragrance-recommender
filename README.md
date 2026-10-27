# Fragrance Finder

A conversational fragrance recommender. Describe a mood, memory, or scent profile and get back ranked recommendations with match and popularity scores, sourced from a database of ~72k fragrances.

## How it works

Each request runs a two-stage LLM pipeline:

1. **Query rewrite** (gpt-oss-20b) converts conversational input into a structured scent profile and extracts any brand or exclusion intent for SQL filtering.
2. **Vector search** embeds the rewritten profile locally (all-MiniLM-L6-v2) and compares it against pre-computed fragrance embeddings in PostgreSQL + pgvector. Top 30 candidates are re-ranked by a blended score: 50% scent similarity, 35% log-scaled popularity, 15% rating.
3. **Recommendation** (gpt-oss-120b) picks 2-3 best fits from the top 7 candidates and writes a natural language explanation.

Embeddings are generated from notes and accords only, not name or brand, so matches reflect how a fragrance smells rather than name recognition.

## Design decisions

**Why embed from notes/accords only?** Including the fragrance name or brand in the embedding would make "Aventus" match "Aventus" by name similarity rather than scent. The embeddings intentionally know nothing about identity, only smell.

**Why a query rewrite stage?** Raw conversational input maps poorly to the `Notes: X. Accords: Y` format the embedding corpus was built on. A fast 8b model restructures the query into that format before embedding, which significantly improves retrieval quality. It also extracts brand and exclusion filters so "similar to Aventus by Creed" does not restrict results to Creed.

**Why log-scale popularity?** Review counts range from single digits to 100k+. Log-scaling with a 50k anchor compresses that range into a meaningful signal without letting mainstream fragrances dominate every result.

## Stack

| Layer | Technology |
|-------|------------|
| Frontend | React + Vite, deployed on Vercel |
| Backend | FastAPI, deployed on Render |
| Database | PostgreSQL + pgvector (Neon) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local, 384-dim) |
| LLM (rewrite) | Groq -- gpt-oss-20b |
| LLM (recommendation) | Groq -- gpt-oss-120b |

## Architecture

```
User (chat input)
    -> React frontend (Vercel)
    -> FastAPI backend (Render)
        -> gpt-oss-20b: rewrite query to structured scent profile + filters
        -> all-MiniLM-L6-v2: embed rewritten profile (local)
        -> pgvector cosine search -> top 30 candidates (Neon)
        -> re-rank: 50% similarity + 35% log-popularity + 15% rating
        -> gpt-oss-120b: pick best 2-3, write descriptions
    -> Ranked matches with scores and explanation
```

## Running locally

**Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). The frontend proxies `/api` to `localhost:8000` in dev.

## Data

Source: Fragrantica catalog via Kaggle ([ledecanteur/fragrantica-perfumes](https://www.kaggle.com/datasets/ledecanteur/fragrantica-perfumes)), saved as `data/perfumes.csv` (comma-delimited, UTF-8; gitignored). The ingest keeps fragrances with at least 10 votes, at least one accord and one note, giving ~72k rows from ~140k.

Cleaning (`scripts/fragrance_data.py`):
- Notes and accords arrive weighted (`Orange:100|Bergamot:28`). Weights are dropped and every note is kept, including 0-weight ones.
- Gender is mapped from `female`/`male` to `women`/`men` to match the frontend filter.
- Duplicate name+brand rows keep the one with the most votes; invalid years become NULL.

To re-embed and re-upload:
```bash
python scripts/embed_and_upload.py
# stricter vote cutoff (smaller table):
MIN_VOTES=25 python scripts/embed_and_upload.py
```

This truncates and repopulates the `fragrances` table in a single transaction, so a failed insert rolls back and leaves the live data untouched. Embeddings are generated locally and uploaded to Neon in a single batch.
