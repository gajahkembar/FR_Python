import uuid
import grpc
import base64
import os
import shutil
from typing import List
from proto import controller_pb2, controller_pb2_grpc
from middleware.db.postgres import (
    save_metadata_to_postgres,
    get_metadata_from_postgres,
    check_duplicate_name_origin,
    get_metadata_by_name_origin,
    delete_metadata_from_postgres
)
from middleware.db.redis import (
    save_metadata_to_redis,
    get_metadata_from_redis,
    check_duplicate_in_redis,
    save_name_origin_to_redis,
    delete_metadata_from_redis, 
    delete_name_origin_mapping
)

# gRPC stub helper
def get_controller_stub():
    channel = grpc.insecure_channel("localhost:1967")
    return controller_pb2_grpc.ControllerServiceStub(channel)

def encode_image_to_base64(path: str) -> str:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

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
def identify_face(image_bytes: bytes, similarity_threshold: float = 0.4):
    stub = get_controller_stub()
    request = controller_pb2.IdentifyRequest(image_data=image_bytes)
    response = stub.Identify(request)

    face_results = []
    for face_result in response.results:
        matches = []
        for match in face_result.top_matches:
            if match.similarity < similarity_threshold:
                continue

            metadata = get_metadata_from_redis(match.user_id)
            if not metadata:
                metadata = get_metadata_from_postgres(match.user_id)
                if metadata:
                    save_metadata_to_redis(
                        match.user_id,
                        metadata.get("name", ""),
                        metadata.get("asal", "")
                    )

            name = metadata.get("name", "")
            origin = metadata.get("origin", metadata.get("asal", ""))
            safe_name = name.strip()
            folder = f"{safe_name}_{origin}_{match.user_id}"
            gallery_filename = f"{folder}.jpg"
            gallery_path = os.path.join("data", folder, gallery_filename)
            gallery_image = encode_image_to_base64(gallery_path)

            matches.append({
                "user_id": match.user_id,
                "similarity": match.similarity,
                "name": name,
                "origin": origin,
                "gallery_image": gallery_image
            })

        crop_b64 = base64.b64encode(face_result.crop_image).decode("utf-8")

        face_results.append({
            "face_index": face_result.face_index,
            "top_matches": matches,
            "crop_image": crop_b64
        })

    return {
        "message": "OK",
        "faces": face_results
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

def delete_face(name: str, origin: str):
    name = name.strip().lower()
    origin = origin.strip().lower()
    
    stub = get_controller_stub()

    user_id = check_duplicate_in_redis(name, origin)
    if not user_id:
        user_id = check_duplicate_name_origin(name, origin)
    if not user_id:
        return {"status": "not_found"}

    delete_metadata_from_postgres(user_id)
    delete_metadata_from_redis(user_id)
    delete_name_origin_mapping(user_id)

    folder = f"{name}_{origin}_{user_id}"
    folder_path = os.path.join("data", folder)

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    else:
        print(f"📁 Folder '{folder_path}' tidak ditemukan, mungkin sudah terhapus sebelumnya.")

    request = controller_pb2.DeleteRequest(user_id=user_id)
    stub.DeleteFace(request)

    return {
        "status": "deleted",
        "user_id": user_id
    }