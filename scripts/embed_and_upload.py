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
df = pd.read_csv(CSV_PATH)

# Clean nulls to avoid NaN string addition
cols_to_fill = [
    'Perfume', 'Brand', 'Gender', 'Rating', 
    'Top Notes', 'Middle Notes', 'Base Notes', 
    'Main Accord 1', 'Main Accord 2', 'Main Accord 3', 'Main Accord 4', 'Main Accord 5'
]
for col in cols_to_fill:
    if col in df.columns:
        df[col] = df[col].fillna('').astype(str).str.strip()
    else:
        df[col] = ''

# Helper to aggregate accords
def clean_accords(row):
    accords = [row['Main Accord 1'], row['Main Accord 2'], row['Main Accord 3'], row['Main Accord 4'], row['Main Accord 5']]
    return ", ".join([a for a in accords if a])

df['accords_list'] = df.apply(clean_accords, axis=1)

# Create embedding text
print("Preparing text for embedding generation...")
df['embedding_text'] = (
    df['Perfume'] + " by " + df['Brand'] + ". " +
    "Gender: " + df['Gender'] + ". " +
    "Notes: " + df['Top Notes'] + ", " + df['Middle Notes'] + ", " + df['Base Notes'] + ". " +
    "Accords: " + df['accords_list']
)

# 3. Generate Embeddings
print(f"Generating embeddings for {len(df)} fragrances (this may take a few minutes on CPU)...")
embeddings = model.encode(df['embedding_text'].tolist(), show_progress_bar=True)
print("Embedding generation completed successfully.")

# 4. Connect to Supabase/Neon and Upload
print("Connecting to PostgreSQL database...")
try:
    conn = psycopg2.connect(DATABASE_URL)
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
        # Clean rating to numeric/float or None if empty
        rating_val = None
        if row['Rating'] and row['Rating'] != 'nan':
            try:
                rating_val = float(row['Rating'])
            except ValueError:
                pass
                
        insert_data.append((
            row['Perfume'],
            row['Brand'],
            row['Gender'] if row['Gender'] else None,
            rating_val,
            row['Top Notes'] if row['Top Notes'] else None,
            row['Middle Notes'] if row['Middle Notes'] else None,
            row['Base Notes'] if row['Base Notes'] else None,
            row['accords_list'] if row['accords_list'] else None,
            embeddings[idx].tolist()
        ))
        
    # Execute batch insert using execute_values (extremely fast)
    insert_query = """
        INSERT INTO fragrances (name, brand, gender, rating, top_notes, middle_notes, base_notes, main_accords, embedding)
        VALUES %s
    """
    
    # Empty table first to avoid duplicate MVP uploads
    print("Clearing existing fragrances table...")
    cur.execute("TRUNCATE TABLE fragrances;")
    
    execute_values(cur, insert_query, insert_data)
    conn.commit()
    print(f"Successfully uploaded {len(insert_data)} fragrances to the database!")

except Exception as e:
    print(f"An error occurred during database upload: {e}")
    sys.exit(1)
finally:
    if 'conn' in locals() and conn:
        cur.close()
        conn.close()
        print("Database connection closed.")
