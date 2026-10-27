import os
import sys
import time
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

from fragrance_data import MIN_VOTES, load_fragrances

# Load environment variables from backend/.env if available
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../backend/.env"))

DATABASE_URL = os.environ.get("DATABASE_URL")
CSV_PATH = os.environ.get(
    "CSV_PATH", os.path.join(os.path.dirname(__file__), "../data/perfumes.csv")
)
MIN_VOTES_ENV = int(os.environ.get("MIN_VOTES", MIN_VOTES))

if not DATABASE_URL:
    print("Error: DATABASE_URL environment variable is not set.")
    print("Please set it in your environment or add it to backend/.env")
    sys.exit(1)

if not os.path.exists(CSV_PATH):
    print(f"Error: CSV file not found at '{CSV_PATH}'.")
    print(
        "Please download 'perfumes.csv' (Fragrantica Perfumes, ledecanteur) from Kaggle and place it in the 'data/' directory."
    )
    sys.exit(1)

# 1. Initialize SentenceTransformer
print("Loading sentence-transformers/all-MiniLM-L6-v2 model...")
# downloads once, then caches locally
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Load and clean the dataset
print(f"Reading dataset from {CSV_PATH} (min {MIN_VOTES_ENV} votes)...")
df = load_fragrances(CSV_PATH, min_votes=MIN_VOTES_ENV)

# 3. Generate Embeddings
# Embedding text uses only the olfactory profile — intentionally excludes name/brand
# so vector similarity reflects scent similarity, not name similarity.
print(
    f"Generating embeddings for {len(df)} fragrances (this may take a few minutes on CPU)..."
)
embeddings = model.encode(df["embedding_text"].tolist(), show_progress_bar=True)
print("Embedding generation completed successfully.")

# 4. Connect to Supabase/Neon and Upload
print("Connecting to PostgreSQL database...")
conn = None
for attempt in range(5):
    try:
        # Use connect_timeout to fail fast and retry if gateway cold start drops connection
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=30)
        break
    except Exception as e:
        print(f"Connection attempt {attempt + 1} failed: {e}")
        if attempt < 4:
            print("Retrying in 5 seconds...")
            time.sleep(5)
else:
    print("Could not connect after 5 attempts.")
    sys.exit(1)

register_vector(conn)
cur = conn.cursor()

# Optional check for pgvector extension
cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
if not cur.fetchone():
    print("pgvector extension not enabled in DB. Attempting to enable it...")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()

print("Uploading data in batches...")


def none_if_empty(value):
    return value if value else None


insert_data = [
    (
        row.name,
        row.brand,
        row.gender,
        row.rating,
        row.rating_count,
        row.year,
        none_if_empty(row.top_notes),
        none_if_empty(row.middle_notes),
        none_if_empty(row.base_notes),
        row.main_accords,
        none_if_empty(row.url),
        embeddings[idx].tolist(),
    )
    for idx, row in enumerate(df.itertuples(index=False))
]

insert_query = """
    INSERT INTO fragrances (name, brand, gender, rating, rating_count, year, top_notes, middle_notes, base_notes, main_accords, url, embedding)
    VALUES %s
"""

# TRUNCATE is transactional in Postgres: nothing is committed until the final
# conn.commit(), so a failed insert rolls back and the live table stays intact.
# (A table-swap would briefly double storage, which the Neon free tier can't spare.)
cur.execute(
    "CREATE UNIQUE INDEX IF NOT EXISTS fragrances_name_brand_key ON fragrances (name, brand);"
)
print("Clearing existing fragrances table...")
cur.execute("TRUNCATE TABLE fragrances;")

execute_values(cur, insert_query, insert_data)
conn.commit()
print(f"Successfully uploaded {len(insert_data)} fragrances to the database!")

cur.close()
conn.close()
print("Database connection closed.")
