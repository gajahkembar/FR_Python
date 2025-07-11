import redis
import numpy as np

r = redis.Redis(host='localhost', port=6379, db=0)
all_keys = r.keys('face_embedding:*')
print(all_keys)
