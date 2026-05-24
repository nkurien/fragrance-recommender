import math
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List
from groq import Groq, RateLimitError
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str
    groq_api_key: str


# Initialize settings (fail fast on missing variables)
settings = Settings()

# Load embedding model once at startup (lightweight, ~22MB)
print("Loading local sentence-transformers/all-MiniLM-L6-v2 embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI(title="Fragrance Recommender API")

# Configure CORS for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For MVP. Adjust to Vercel domain in production.
    allow_credentials=False,  # Wildcard origins do not permit credentials
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Database Connection Pool
# Note: psycopg2 is synchronous. By declaring FastAPI endpoints as standard synchronous functions
# (using `def` instead of `async def`), FastAPI automatically executes them on a background thread pool,
# ensuring the event loop is never blocked.
try:
    db_pool = SimpleConnectionPool(
        1,
        10,
        dsn=settings.database_url,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )
    print("Database connection pool initialized successfully with keepalives.")
except Exception as e:
    print(f"Failed to initialize database pool: {e}")
    db_pool = None

# Initialize Groq Client
groq_client = None
if settings.groq_api_key:
    groq_client = Groq(api_key=settings.groq_api_key)


# Pydantic schemas
class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str


class RecommendRequest(BaseModel):
    description: str
    gender: Optional[str] = None
    history: Optional[List[ChatMessage]] = []


class FragranceMatch(BaseModel):
    name: str
    brand: str
    gender: Optional[str]
    rating: Optional[float]
    rating_count: Optional[int]
    year: Optional[int]
    top_notes: Optional[str]
    middle_notes: Optional[str]
    base_notes: Optional[str]
    main_accords: Optional[str]
    url: Optional[str]
    match_score: Optional[float]
    popularity_score: Optional[float]


class RecommendResponse(BaseModel):
    recommendation: str
    matches: List[FragranceMatch]


POPULARITY_ANCHOR = math.log1p(50_000)


def _build_conditions(
    gender: str | None,
    brand_filter: str | None,
    exclude_name: str | None,
) -> tuple[list, list]:
    """Return (conditions, params) for the pgvector WHERE clause."""
    conditions: list = []
    params: list = []
    if gender:
        conditions.append("LOWER(gender) = LOWER(%s)")
        params.append(gender)
    if brand_filter:
        conditions.append("LOWER(brand) LIKE LOWER(%s)")
        params.append(f"%{brand_filter.replace(' ', '-')}%")
    if exclude_name:
        conditions.append("LOWER(name) NOT LIKE LOWER(%s)")
        params.append(f"%{exclude_name.replace(' ', '-')}%")
    return conditions, params


def _score_and_rank(rows: list) -> list:
    """Re-rank DB rows: 50% similarity + 35% log-popularity + 15% rating."""
    max_log_count = max((math.log1p(r[4] or 0) for r in rows), default=1) or 1
    scored = []
    for row in rows:
        similarity = 1 - row[11]
        popularity = math.log1p(row[4] or 0) / max_log_count
        quality = (float(row[3]) / 5.0) if row[3] is not None else 0
        score = 0.50 * similarity + 0.35 * popularity + 0.15 * quality
        scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def get_query_embedding(query_text: str) -> List[float]:
    try:
        # Encode user query using the local model
        return embedding_model.encode(query_text).tolist()
    except Exception as e:
        print(f"Error generating local embedding: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate query embedding: {str(e)}"
        )


@app.get("/api/health")
def health_check():
    # Verify DB pool is healthy
    db_status = "unconfigured"
    if db_pool:
        try:
            conn = db_pool.getconn()
            cur = conn.cursor()
            cur.execute("SELECT 1;")
            cur.fetchone()
            db_pool.putconn(conn)
            db_status = "healthy"
        except Exception as e:
            db_status = f"error: {str(e)}"

    return {
        "status": "healthy",
        "database": db_status,
        "groq": "configured" if groq_client else "unconfigured",
        "local_embeddings": "loaded",
    }


def rewrite_query(description: str, history: list, gender: str | None = None) -> dict:
    """
    Use a fast LLM to rewrite the user's conversational query into a structured
    scent profile that matches the embedding space, and extract any brand intent.

    Returns a dict with:
      - embedding_text: structured profile for vector search
      - brand_filter:   brand to restrict results to, or null
      - exclude_name:   specific fragrance name to exclude (when user asks for similar to X)
    """
    gender_hint = f" The user has selected a gender filter: {gender}." if gender else ""
    system = (
        f"You are a fragrance expert.{gender_hint} Interpret the user's query and return a JSON object with exactly these fields:\n"
        '- "embedding_text": structured scent description in this format: '
        '"Gender: <men|women|unisex>. Notes: <comma-separated notes>. Accords: <comma-separated accords>". '
        "If the user references a specific fragrance by name, describe its known scent profile in notes/accords. "
        "Use conversation history to resolve follow-ups like 'something newer' or 'by that brand'.\n"
        "- \"brand_filter\": lowercase hyphenated brand slug (e.g. 'creed', 'carolina-herrera') ONLY if the user "
        "explicitly wants results FROM a specific brand (e.g. 'recommend me a Creed', 'show me Dior fragrances'). "
        "IMPORTANT: set to null when the brand only identifies a source fragrance in a 'similar to' query "
        "(e.g. 'something like Aventus by Creed' → brand_filter=null, exclude_name='aventus'). "
        "Also null if the brand appears only in previous assistant responses or if the user wants to avoid a brand.\n"
        "- \"exclude_name\": lowercase hyphenated fragrance name slug (e.g. 'good-girl', 'aventus') if the user "
        "wants something SIMILAR TO a named fragrance, so we exclude that fragrance from results. null otherwise.\n"
        "Return ONLY valid JSON. No explanation, no markdown."
    )

    history_messages = [
        {"role": m["role"], "content": m["content"]} for m in history[-6:]
    ]

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system},
                *history_messages,
                {"role": "user", "content": description},
            ],
            max_tokens=200,
            temperature=0.1,
            response_format={"type": "json_object"},
            timeout=30,
        )
        result = json.loads(response.choices[0].message.content)
        print(f"Query rewrite: {result}")
        return {
            "embedding_text": result.get("embedding_text") or description,
            "brand_filter": result.get("brand_filter"),
            "exclude_name": result.get("exclude_name"),
        }
    except Exception as e:
        print(f"Query rewrite failed ({e}), falling back to raw query.")
        return {
            "embedding_text": description,
            "brand_filter": None,
            "exclude_name": None,
        }


@app.post("/api/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    if not db_pool:
        raise HTTPException(
            status_code=500, detail="Database connection pool is uninitialized."
        )
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API client is unconfigured.")

    conn = None
    matches = []
    try:
        conn = db_pool.getconn()

        # 1. Rewrite the conversational query into a structured scent profile
        history_dicts = [
            {"role": m.role, "content": m.content} for m in (request.history or [])
        ]
        rewrite = rewrite_query(
            request.description, history_dicts, gender=request.gender
        )

        embedding_query = rewrite.get("embedding_text") or request.description
        exclude_name = rewrite.get("exclude_name")
        # If the user wants something *similar to* a named fragrance, the brand in their
        # message identifies the source — it is not a request to restrict results to that brand.
        brand_filter = rewrite.get("brand_filter") if not exclude_name else None
        brand_hint = rewrite.get("brand_filter")  # passed to LLM prompt for context

        print(f"Generating embedding for: '{embedding_query[:120]}...'")
        query_vector = get_query_embedding(embedding_query)

        # 2. Query Postgres pgvector
        cur = conn.cursor()

        conditions, filter_params = _build_conditions(
            request.gender, brand_filter, exclude_name
        )
        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        query_sql = f"""
            SELECT name, brand, gender, rating, rating_count, year, top_notes, middle_notes, base_notes, main_accords, url,
                   embedding <=> %s::vector AS distance
            FROM fragrances
            {where_clause}
            ORDER BY distance LIMIT 30;
        """
        # embedding vector must be first param for the <=> operator
        cur.execute(query_sql, [query_vector] + filter_params)
        rows = cur.fetchall()

        scored = _score_and_rank(rows)

        for blended, row in scored[:30]:
            match_pct = round((1 - row[11]) * 100)
            popularity_pct = min(
                round(math.log1p(row[4] or 0) / POPULARITY_ANCHOR * 100), 100
            )
            matches.append(
                FragranceMatch(
                    name=row[0],
                    brand=row[1],
                    gender=row[2],
                    rating=float(row[3]) if row[3] is not None else None,
                    rating_count=row[4],
                    year=row[5],
                    top_notes=row[6],
                    middle_notes=row[7],
                    base_notes=row[8],
                    main_accords=row[9],
                    url=row[10],
                    match_score=match_pct,
                    popularity_score=popularity_pct,
                )
            )

    except Exception as e:
        print(f"Database query error: {e}")
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
    finally:
        if conn:
            db_pool.putconn(conn)

    if not matches:
        return RecommendResponse(
            recommendation="I couldn't find any fragrances matching your criteria. Try describing your preference in different terms!",
            matches=[],
        )

    # 3. Format candidates for LLM prompt
    candidates_text = ""
    for idx, match in enumerate(matches):
        notes_desc = []
        if match.top_notes:
            notes_desc.append(f"Top: {match.top_notes}")
        if match.middle_notes:
            notes_desc.append(f"Middle: {match.middle_notes}")
        if match.base_notes:
            notes_desc.append(f"Base: {match.base_notes}")
        notes_str = "; ".join(notes_desc) if notes_desc else "No notes listed"

        candidates_text += f"{idx + 1}. **{match.name}** by {match.brand}\n"
        candidates_text += f"   - Profile: {match.gender or 'Unisex'}, Rating: {match.rating or 'N/A'}/5\n"
        candidates_text += f"   - Notes: {notes_str}\n"
        candidates_text += f"   - Accords: {match.main_accords or 'None'}\n\n"

    # 4. Invoke LLM reasoning on Groq
    system_prompt = (
        "You are an elegant, highly knowledgeable, and poetic fragrance sommelier.\n"
        "Your task is to recommend the best fragrances from the provided candidate list.\n\n"
        "RULES:\n"
        "1. Recommend 2-3 fragrances. If the candidate list does not contain enough good matches, "
        "recommend fewer — never pad with poor fits.\n"
        "2. CRITICAL: You may ONLY recommend fragrances that appear verbatim in the candidate list. "
        "Do not invent, recall from memory, or suggest any fragrance not in the list.\n"
        "3. For each recommendation, describe why it fits using poetic, sensory language referencing "
        "specific notes or accords. Keep it to 2-3 sentences per fragrance.\n"
        "4. Be warm and welcoming. Do NOT open with salutations like 'Dear friend' or 'My dear'.\n"
        "5. Use conversation history to understand follow-up requests in context.\n"
        "6. Format your response using markdown: bold (**name**) each fragrance name, "
        "and use a blank line between each recommendation."
    )

    brand_context = (
        (
            f"\nNote: the user's query references the brand '{brand_hint}'. "
            "Use your judgement — prefer candidates from that brand if the user wants them; "
            "handle 'not X' or 'similar to X' accordingly."
        )
        if brand_hint
        else ""
    )

    user_prompt = (
        f'The user wants: "{request.description}"{brand_context}\n\n'
        f"Candidate Fragrances (you may ONLY recommend from this list):\n{candidates_text}\n"
        f"Recommend the best matches and explain why."
    )

    history_messages = [
        {"role": msg.role, "content": msg.content}
        for msg in (request.history or [])[-6:]
    ]

    recommendation_text = None

    try:
        chat_completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                *history_messages,
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1000,
            temperature=0.4,
            timeout=30,
        )
        recommendation_text = chat_completion.choices[0].message.content
    except RateLimitError as e:
        print(f"Groq rate limit hit: {e}")
        import re

        retry_match = re.search(r"try again in (\d+m[\d.]+s|\d+[\d.]*s)", str(e))
        retry_hint = (
            f" Please try again in {retry_match.group(1)}."
            if retry_match
            else " Please try again in a few minutes."
        )
        recommendation_text = (
            f"I've hit my usage limit for the moment and can't write descriptions right now.{retry_hint} In the meantime, here are the top matches I found: "
            + ", ".join(f"**{m.name}** by {m.brand}" for m in matches[:3])
            + "."
        )
    except Exception as e:
        print(f"Groq API error: {e}")
        recommendation_text = (
            "My sommelier reasoning is unavailable right now. Here are the top matches I found: "
            + ", ".join(f"**{m.name}** by {m.brand}" for m in matches[:3])
            + "."
        )

    return RecommendResponse(recommendation=recommendation_text, matches=matches)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
