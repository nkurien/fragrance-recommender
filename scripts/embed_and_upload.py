import os
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

# Load environment variables from backend/.env if available
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../backend/.env'))

DATABASE_URL = os.environ.get("DATABASE_URL")
CSV_PATH = os.environ.get("CSV_PATH", os.path.join(os.path.dirname(__file__), "../data/fra_cleaned.csv"))

if not DATABASE_URL:
    print("Error: DATABASE_URL environment variable is not set.")
    print("Please set it in your environment or add it to backend/.env")
    sys.exit(1)

if not os.path.exists(CSV_PATH):
    print(f"Error: CSV file not found at '{CSV_PATH}'.")
    print("Please download 'fra_cleaned.csv' from Kaggle and place it in the 'data/' directory.")
    sys.exit(1)

# 1. Initialize SentenceTransformer
print("Loading sentence-transformers/all-MiniLM-L6-v2 model...")
# downloads once, then caches locally
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Load and Preprocess CSV
print(f"Reading dataset from {CSV_PATH}...")
# Delimiter is semicolons in raw data, encoding is latin1 for European characters
df = pd.read_csv(CSV_PATH, sep=';', encoding='latin1')

# Clean nulls to avoid NaN string addition
cols_to_fill = [
    'url', 'Perfume', 'Brand', 'Gender', 'Rating Value', 'Rating Count',
    'Top', 'Middle', 'Base',
    'mainaccord1', 'mainaccord2', 'mainaccord3', 'mainaccord4', 'mainaccord5'
]
for col in cols_to_fill:
    if col in df.columns:
        df[col] = df[col].fillna('').astype(str).str.strip()
    else:
        df[col] = ''

# Helper to aggregate accords
def clean_accords(row):
    accords = [row['mainaccord1'], row['mainaccord2'], row['mainaccord3'], row['mainaccord4'], row['mainaccord5']]
    return ", ".join([a for a in accords if a])

df['accords_list'] = df.apply(clean_accords, axis=1)

# Create embedding text using only olfactory profile — intentionally exclude name/brand
# so vector similarity reflects scent similarity, not name similarity.
print("Preparing text for embedding generation...")
df['embedding_text'] = (
    "Gender: " + df['Gender'] + ". " +
    "Notes: " + df['Top'] + ", " + df['Middle'] + ", " + df['Base'] + ". " +
    "Accords: " + df['accords_list']
)

# 3. Generate Embeddings
print(f"Generating embeddings for {len(df)} fragrances (this may take a few minutes on CPU)...")
embeddings = model.encode(df['embedding_text'].tolist(), show_progress_bar=True)
print("Embedding generation completed successfully.")

# 4. Connect to Supabase/Neon and Upload
import time

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

# Prepare batch insert tuples
insert_data = []
for idx, row in df.iterrows():
    # Clean rating to numeric/float (replace commas with dots)
    rating_val = None
    rating_str = row['Rating Value']
    if rating_str and rating_str != 'nan' and rating_str != '':
        try:
            rating_val = float(rating_str.replace(',', '.'))
        except ValueError:
            pass

    rating_count_val = None
    rating_count_str = row['Rating Count']
    if rating_count_str and rating_count_str != 'nan' and rating_count_str != '':
        try:
            rating_count_val = int(float(rating_count_str.replace(',', '.')))
        except ValueError:
            pass

    insert_data.append((
        row['Perfume'],
        row['Brand'],
        row['Gender'] if row['Gender'] else None,
        rating_val,
        rating_count_val,
        row['Top'] if row['Top'] else None,
        row['Middle'] if row['Middle'] else None,
        row['Base'] if row['Base'] else None,
        row['accords_list'] if row['accords_list'] else None,
        row['url'] if row['url'] else None,
        embeddings[idx].tolist()
    ))
    
# Execute batch insert using execute_values (extremely fast)
insert_query = """
    INSERT INTO fragrances (name, brand, gender, rating, rating_count, top_notes, middle_notes, base_notes, main_accords, url, embedding)
    VALUES %s
"""

# Empty table first to avoid duplicate MVP uploads
print("Clearing existing fragrances table...")
cur.execute("TRUNCATE TABLE fragrances;")

execute_values(cur, insert_query, insert_data)
conn.commit()
print(f"Successfully uploaded {len(insert_data)} fragrances to the database!")

cur.close()
conn.close()
print("Database connection closed.")
