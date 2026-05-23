# Fragrance Recommender

A chat interface that recommends fragrances based on natural language descriptions. Describe what you're looking for and get back 2-3 recommendations with explanations.

Built for a small group of friends. Runs entirely on free tiers.

## Stack

| Layer | Service |
|-------|---------|
| Frontend | React + Vite, deployed on Vercel |
| Backend | FastAPI, deployed on Render |
| Database | Neon (PostgreSQL + pgvector) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, runs locally) |
| LLM | Groq API (llama-3.1-8b-instant) |

**Note:** Render's free tier spins down after 15 minutes of inactivity. The first request after idle takes 30-50 seconds. This is expected behaviour.

## Architecture

```
User (chat input)
    -> React frontend (Vercel)
    -> FastAPI backend (Render)
        -> pgvector similarity search (Neon)
        -> Groq LLM for recommendation text
    -> Response with matches + explanation
```

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- A [Neon](https://neon.tech) account (free)
- A [Groq](https://console.groq.com) API key (free)

### 1. Database

Run `database_schema.sql` in the Neon SQL editor to create the `fragrances` table and enable the pgvector extension.

### 2. Environment variables

Copy the example file and fill in your values:

```bash
cp backend/.env.example backend/.env
```

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | Neon connection string (from the Neon console, Connection Details) |
| `GROQ_API_KEY` | From [console.groq.com](https://console.groq.com) |

### 3. Upload fragrance data (run once)

Download `fra_cleaned.csv` from Kaggle and place it in `data/`. Then:

```bash
pip install pandas sentence-transformers psycopg2 python-dotenv pgvector
python scripts/embed_and_upload.py
```

This generates embeddings locally and uploads all rows to Neon. Takes a few minutes on CPU.

**Important:** Do not run this on a VPN. Port 5432 outbound is often blocked by VPN deep packet inspection.

### 4. Run the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 5. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). The Vite dev server proxies `/api/` requests to `http://localhost:8000`.

## Deployment

The backend is deployed on Render and the frontend on Vercel, both via GitHub integration. Set `DATABASE_URL` and `GROQ_API_KEY` as environment variables in Render, and `VITE_API_URL` (pointing to the Render service URL) in Vercel.

## Data

Source: [Fragrantica dataset on Kaggle](https://www.kaggle.com) (`fra_cleaned.csv`, ~24k fragrances).

Each fragrance is embedded as: `"{name} by {brand}. Gender: {gender}. Notes: {top}, {middle}, {base}. Accords: {accords}"` using `all-MiniLM-L6-v2` (384-dimensional vectors).
