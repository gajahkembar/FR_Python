# import base64

# with open("/mnt/c/Users/rahma/Pictures/Aaron_Eckhart_0001.jpg", "rb") as f:
#     b64_bytes = base64.b64encode(f.read()).decode()

# print(b64_bytes)  # masukkan ini ke Postman

# from src.db import save_metadata_to_postgres
# import uuid

# uid = str(uuid.uuid4())
# save_metadata_to_postgres(uid, "Rahma", "Yogyakarta")

from src.redis_store import save_metadata_to_redis, get_metadata_from_redis
import uuid

uid = str(uuid.uuid4())
save_metadata_to_redis(uid, "Rahma", "Jogja")

data = get_metadata_from_redis(uid)
print("Fetched:", data)