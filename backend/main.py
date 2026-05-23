import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Optional, List
from groq import Groq
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    database_url: str = Field(..., env="DATABASE_URL")
    groq_api_key: str = Field(..., env="GROQ_API_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Initialize settings (fail fast on missing variables)
settings = Settings()

# Load embedding model once at startup (lightweight, ~22MB)
print("Loading local sentence-transformers/all-MiniLM-L6-v2 embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

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
    db_pool = SimpleConnectionPool(1, 10, dsn=settings.database_url)
    print("Database connection pool initialized successfully.")
except Exception as e:
    print(f"Failed to initialize database pool: {e}")
    db_pool = None

# Initialize Groq Client
groq_client = None
if settings.groq_api_key:
    groq_client = Groq(api_key=settings.groq_api_key)

# Pydantic schemas
class RecommendRequest(BaseModel):
    description: str
    gender: Optional[str] = None # Optional: 'male', 'female', 'unisex'

class FragranceMatch(BaseModel):
    name: str
    brand: str
    gender: Optional[str]
    rating: Optional[float]
    top_notes: Optional[str]
    middle_notes: Optional[str]
    base_notes: Optional[str]
    main_accords: Optional[str]

class RecommendResponse(BaseModel):
    recommendation: str
    matches: List[FragranceMatch]

def get_query_embedding(query_text: str) -> List[float]:
    try:
        # Encode user query using the local model
        return embedding_model.encode(query_text).tolist()
    except Exception as e:
        print(f"Error generating local embedding: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate query embedding: {str(e)}"
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
        "local_embeddings": "loaded"
    }

@app.post("/api/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database connection pool is uninitialized.")
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API client is unconfigured.")

    # 1. Vectorize user query
    print(f"Generating embedding for query: '{request.description}'")
    query_vector = get_query_embedding(request.description)
    
    # 2. Query Postgres pgvector
    conn = None
    matches = []
    try:
        conn = db_pool.getconn()
        cur = conn.cursor()
        
        # Build vector search query
        # Using cosine distance operator <=>
        query_sql = """
            SELECT name, brand, gender, rating, top_notes, middle_notes, base_notes, main_accords,
                   embedding <=> %s::vector AS distance
            FROM fragrances
        """
        
        # Optional: Add hard gender filtering if requested
        params = [query_vector]
        if request.gender:
            query_sql += " WHERE LOWER(gender) = LOWER(%s)"
            params.append(request.gender)
            
        query_sql += " ORDER BY distance LIMIT 5;"
        
        cur.execute(query_sql, params)
        rows = cur.fetchall()
        
        for row in rows:
            matches.append(FragranceMatch(
                name=row[0],
                brand=row[1],
                gender=row[2],
                rating=float(row[3]) if row[3] is not None else None,
                top_notes=row[4],
                middle_notes=row[5],
                base_notes=row[6],
                main_accords=row[7]
            ))
            
    except Exception as e:
        print(f"Database query error: {e}")
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
    finally:
        if conn:
            db_pool.putconn(conn)

    if not matches:
        return RecommendResponse(
            recommendation="I couldn't find any fragrances matching your criteria. Try describing your preference in different terms!",
            matches=[]
        )

    # 3. Format candidates for LLM prompt
    candidates_text = ""
    for idx, match in enumerate(matches):
        notes_desc = []
        if match.top_notes: notes_desc.append(f"Top: {match.top_notes}")
        if match.middle_notes: notes_desc.append(f"Middle: {match.middle_notes}")
        if match.base_notes: notes_desc.append(f"Base: {match.base_notes}")
        notes_str = "; ".join(notes_desc) if notes_desc else "No notes listed"
        
        candidates_text += f"{idx + 1}. **{match.name}** by {match.brand}\n"
        candidates_text += f"   - Profile: {match.gender or 'Unisex'}, Rating: {match.rating or 'N/A'}/5\n"
        candidates_text += f"   - Notes: {notes_str}\n"
        candidates_text += f"   - Accords: {match.main_accords or 'None'}\n\n"

    # 4. Invoke LLM reasoning on Groq
    system_prompt = (
        "You are an elegant, highly knowledgeable, and poetic fragrance sommelier.\n"
        "Your task is to recommend the best fragrances from the provided list of candidates "
        "that match what the user is looking for.\n\n"
        "RULES:\n"
        "1. Recommend exactly 2-3 fragrances from the list below.\n"
        "2. Do NOT invent, hallucinate, or suggest any fragrance that is not in the candidate list.\n"
        "3. For each recommendation, describe why it fits using poetic, sensory, and engaging language "
        "referencing specific notes (top, middle, base) or main accords. Keep it to 2-3 sentences per fragrance.\n"
        "4. Be warm and welcoming."
    )
    
    user_prompt = (
        f"The user wants: \"{request.description}\"\n\n"
        f"Candidate Fragrances:\n{candidates_text}\n"
        f"Recommend the best matches and explain why."
    )
    
    try:
        chat_completion = groq_client.chat.completions.create(
            # llama-3.1-8b-instant is standard, low-latency, and active on Groq
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        recommendation_text = chat_completion.choices[0].message.content
        
    except Exception as e:
        print(f"Groq API Error: {e}")
        # Soft fallback explanation if LLM fails, listing candidates directly
        fallback_matches = ", ".join([f"{m.name} by {m.brand}" for m in matches[:3]])
        recommendation_text = (
            f"Here are top matches that fit your profile: {fallback_matches}. "
            "(Apologies, my sommelier reasoning module is currently resting, but these candidates "
            "closely match your description based on database records!)"
        )

    return RecommendResponse(
        recommendation=recommendation_text,
        matches=matches
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
