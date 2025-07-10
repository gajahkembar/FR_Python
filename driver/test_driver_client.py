import grpc
from proto import driver_pb2, driver_pb2_grpc, common_pb2

# Sambungkan ke Driver
channel = grpc.insecure_channel('localhost:1968')
stub = driver_pb2_grpc.DriverServiceStub(channel)

# Dummy face embedding (misal hasil ArcFace 512 dimensi)
dummy_embedding = common_pb2.FaceEmbedding(
    user_id="query_test",
    embedding=[0.01 * i for i in range(512)]
)

# Kirim ke Driver
response = stub.RouteIdentify(dummy_embedding)

print("Hasil dari Driver:")
print(f"User ID: {response.user_id}")
print(f"Similarity: {response.similarity}")
print(f"Message: {response.message}")