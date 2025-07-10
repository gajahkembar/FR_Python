# src/db.py
import psycopg2
from datetime import datetime
import numpy as np

def save_metadata_to_postgres(uuid, name, asal):
    try:
        conn = psycopg2.connect(
            dbname="engine_face",
            user="postgres",
            password="Rahmanda07",
            host="localhost",
            port=5432
        )
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO face_metadata (uuid, name, asal, timestamp)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (uuid) DO NOTHING;
        """, (uuid, name, asal, datetime.now()))
        conn.commit()
        cur.close()
        conn.close()
        print(f"[DB] Metadata saved: {uuid} - {name} ({asal})")
    except Exception as e:
        print(f"[DB] Failed to save metadata: {e}")