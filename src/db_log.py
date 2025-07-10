# src/db_log.py

import psycopg2
from datetime import datetime
import numpy as np

def insert_log(trx_id, operation, user_id, similarity, result):
    print(f"[DEBUG] Logging to DB: {trx_id}, {operation}, {user_id}, {similarity}, {result}")
    try:
        with psycopg2.connect(
            dbname="engine_face",
            user="postgres",
            password="Rahmanda07",
            host="localhost",
            port=5432
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO activity_logs (trx_id, operation, user_id, similarity, result)
                    VALUES (%s, %s, %s, %s, %s)
                """, (trx_id, operation, user_id, similarity, result))
    except Exception as e:
        print(f"[DB_LOG] Error logging activity: {e}")