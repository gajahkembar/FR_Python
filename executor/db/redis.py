import redis
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=False  # untuk menyimpan bytes
)

def save_embedding_to_redis(uuid: str, embedding: bytes):
    key = f"face:{uuid}"
    redis_client.set(key, embedding)

def load_all_embeddings_from_redis():
    keys = redis_client.keys("face:*")
    embeddings = {}
    for key in keys:
        uuid = key.decode().split(":")[1] if isinstance(key, bytes) else key.split(":")[1]
        embeddings[uuid] = redis_client.get(key)
    return embeddings

def delete_embedding_from_redis(user_id):
    key = f"face:{user_id}"
    redis_client.delete(key)