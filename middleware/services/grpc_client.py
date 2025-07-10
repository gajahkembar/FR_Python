import uuid
import grpc
from typing import List
from proto import controller_pb2, controller_pb2_grpc
from middleware.db.postgres import (
    save_metadata_to_postgres,
    get_metadata_from_postgres,
    check_duplicate_name_origin,
    get_metadata_by_name_origin
)
from middleware.db.redis import (
    save_metadata_to_redis,
    get_metadata_from_redis,
    check_duplicate_in_redis,
    save_name_origin_to_redis
)

# gRPC stub helper
def get_controller_stub():
    channel = grpc.insecure_channel("localhost:1967")
    return controller_pb2_grpc.ControllerServiceStub(channel)

# Register wajah
def register_face(image_bytes: bytes, name: str, origin: str):
    stub = get_controller_stub()

    # 🔍 Cek duplikasi dari Redis dulu
    existing_uuid = check_duplicate_in_redis(name, origin)
    if existing_uuid:
        return {
            "user_id": existing_uuid,
            "status": "duplicate_name_origin"
        }

    # 🔍 Lanjut cek PostgreSQL jika Redis kosong
    existing_uuid = check_duplicate_name_origin(name, origin)
    if existing_uuid:
        save_name_origin_to_redis(name, origin, existing_uuid)  # cache ke Redis
        return {
            "user_id": existing_uuid,
            "status": "duplicate_name_origin"
        }

    # UUID baru
    user_id = str(uuid.uuid4())

    # ✅ Simpan metadata (Redis + PostgreSQL)
    save_metadata_to_postgres(user_id, name, origin)
    save_metadata_to_redis(user_id, name, origin)
    save_name_origin_to_redis(name, origin, user_id)

    # ✅ Kirim ke controller untuk simpan embedding ke backend
    request = controller_pb2.RegisterRequest(
        image_data=image_bytes,
        name=name,
        origin=origin,
        user_id=user_id
    )
    stub.RegisterFace(request)

    print(f"✅ Registered new user {name} from {origin} with UUID {user_id}")
    return {
        "user_id": user_id,
        "status": "success"
    }

# Identifikasi wajah
def identify_face(image_bytes: bytes):
    stub = get_controller_stub()
    request = controller_pb2.IdentifyRequest(image_data=image_bytes)
    response = stub.Identify(request)

    matches = []
    for match in response.top_matches:
        metadata = get_metadata_from_redis(match.user_id)
        if not metadata:
            metadata = get_metadata_from_postgres(match.user_id)
            # Optional: cache kembali ke Redis jika berhasil
            if metadata:
                save_metadata_to_redis(
                    match.user_id,
                    metadata.get("name", ""),
                    metadata.get("asal", "")
                )
        matches.append({
            "user_id": match.user_id,
            "similarity": match.similarity,
            "name": metadata.get("name", ""),
            "origin": metadata.get("origin", metadata.get("asal", ""))  # handle asal/nama
        })

    return {
        "user_id": response.user_id,
        "similarity": response.similarity,
        "result": response.result,
        "top_matches": matches
    }

def verify_face(image1_bytes: bytes, image2_bytes: bytes):
    stub = get_controller_stub()
    request = controller_pb2.VerifyRequest(
        image1=image1_bytes,
        image2=image2_bytes
    )
    response = stub.VerifyFaces(request)
    
    return {
        "similarity": response.similarity,
        "result": response.result
    }