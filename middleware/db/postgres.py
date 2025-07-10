import psycopg2
from psycopg2.extras import RealDictCursor
from middleware.config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

def get_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )

def save_metadata_to_postgres(uuid: str, name: str, origin: str):
    conn = get_connection()
    cur = conn.cursor()
    query = """
        INSERT INTO face_metadata (uuid, name, asal)
        VALUES (%s, %s, %s)
        ON CONFLICT (uuid) DO NOTHING;
    """
    cur.execute(query, (uuid, name, origin))
    conn.commit()
    cur.close()
    conn.close()

def get_metadata_from_postgres(uuid: str):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    query = "SELECT * FROM face_metadata WHERE uuid = %s;"
    cur.execute(query, (uuid,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result if result else {}

def check_duplicate_name_origin(name: str, origin: str):
    conn = get_connection()
    cur = conn.cursor()
    query = "SELECT uuid FROM face_metadata WHERE name = %s AND asal = %s;"
    cur.execute(query, (name, origin))
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result[0] if result else None

def get_metadata_by_name_origin(name: str, origin: str):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    query = "SELECT * FROM face_metadata WHERE name = %s AND asal = %s;"
    cur.execute(query, (name, origin))
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result