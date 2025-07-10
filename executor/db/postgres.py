import psycopg2
import os
from dotenv import load_dotenv
from psycopg2 import Binary

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB = os.getenv("POSTGRES_DB", "engine_face")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "Rahmanda07")

def get_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )

def save_embedding_to_postgres(uuid: str, embedding: bytes):
    conn = get_connection()
    cur = conn.cursor()
    query = """
        INSERT INTO face_embedding (uuid, embedding)
        VALUES (%s, %s)
        ON CONFLICT (uuid) DO UPDATE SET embedding = EXCLUDED.embedding;
    """
    cur.execute(query, (uuid, Binary(embedding)))
    conn.commit()
    cur.close()
    conn.close()

def load_all_embeddings_from_postgres():
    conn = get_connection()
    cur = conn.cursor()
    query = "SELECT uuid, embedding FROM face_embedding;"
    cur.execute(query)
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results  # list of tuples (uuid, embedding)