import redis
import json
import os
from middleware.config import REDIS_HOST, REDIS_PORT

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True
)

def save_metadata_to_redis(user_id: str, name: str, origin: str):
    key = f"user:{user_id}"
    value = json.dumps({"name": name, "origin": origin})
    redis_client.set(key, value)

def get_metadata_from_redis(user_id: str):
    key = f"user:{user_id}"
    value = redis_client.get(key)
    if value:
        return json.loads(value)
    return {}

def check_duplicate_in_redis(name: str, origin: str):
    key = f"user_by_name_origin:{name.lower()}_{origin.lower()}"
    return redis_client.get(key)  # return UUID if exists

def save_name_origin_to_redis(name: str, origin: str, uuid: str):
    key = f"user_by_name_origin:{name.lower()}_{origin.lower()}"
    redis_client.set(key, uuid)

def delete_metadata_from_redis(user_id: str):
    redis_client.delete(f"user:{user_id}")

def delete_name_origin_mapping(user_id: str):
    pattern = "user_by_name_origin:*"
    for key in redis_client.scan_iter(match=pattern):
        if redis_client.get(key) == user_id:
            redis_client.delete(key)
            break