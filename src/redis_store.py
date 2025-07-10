# src/redis_store.py

import redis
import json
import numpy as np

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

def save_metadata_to_redis(uuid, name, asal):
    key = f"face:{uuid}"
    value = {
        "name": name,
        "asal": asal
    }
    try:
        r.set(key, json.dumps(value))
        print(f"[REDIS] Saved metadata for {uuid}")
    except Exception as e:
        print(f"[REDIS] Error saving metadata: {e}")

def get_metadata_from_redis(uuid):
    key = f"face:{uuid}"
    try:
        data = r.get(key)
        return json.loads(data) if data else None
    except Exception as e:
        print(f"[REDIS] Error getting metadata: {e}")
        return None