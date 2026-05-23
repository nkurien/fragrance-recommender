# Fragrance Finder

A conversational fragrance recommender. Describe a mood, memory, or scent profile and get back ranked recommendations with match and popularity scores, sourced from a database of ~24k fragrances.

## How it works

User input is embedded using a local sentence-transformers model and compared against pre-computed fragrance embeddings in a PostgreSQL + pgvector database. The top 30 nearest candidates are re-ranked by a blended score (scent similarity, review count, rating) and the best results are passed to an LLM which writes a natural language explanation.

Embeddings are generated from notes and accords only, so matches reflect how a fragrance smells rather than name recognition.

## Stack

| Layer | Technology |
|-------|------------|
| Frontend | React + Vite, deployed on Vercel |
| Backend | FastAPI, deployed on Render |
| Database | PostgreSQL + pgvector (Neon) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | Groq API (llama-3.1-8b-instant) |

## Architecture

```
User (chat input)
    -> React frontend (Vercel)
    -> FastAPI backend (Render)
        -> Embed query with all-MiniLM-L6-v2
        -> pgvector cosine similarity search (Neon)
        -> Re-rank by blended score (similarity + popularity + rating)
        -> Groq LLM generates recommendation text
    -> Matches with scores and explanation
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

Open [http://localhost:5173](http://localhost:5173).

## Data

Source: Fragrantica dataset (~24k fragrances) via Kaggle.

Each fragrance is embedded from its gender, notes (top, middle, base), and main accords using all-MiniLM-L6-v2 (384 dimensions). Embeddings are generated locally and uploaded to Neon via `scripts/embed_and_upload.py`.
