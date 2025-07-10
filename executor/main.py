import grpc
from concurrent import futures
from datetime import datetime
import logging
import sys
import os
import base64
import numpy as np
from multiprocessing import Process, set_start_method
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from proto import executor_pb2, executor_pb2_grpc, common_pb2
from src.embedder import ArcFaceEmbedder 
from executor.db.postgres import save_embedding_to_postgres
from executor.db.redis import save_embedding_to_redis, load_all_embeddings_from_redis

def setup_logger(port):
    log_file = "executor/executor.log"
    logger = logging.getLogger(f"Executor-{port}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter(f"%(asctime)s - [{port}] - %(levelname)s - %(message)s")
        handler = logging.FileHandler(log_file, mode='a')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

class ExecutorServicer(executor_pb2_grpc.ExecutorServiceServicer):
    def __init__(self, logger):
        self.logger = logger
        self.embedder = ArcFaceEmbedder("models/w600k_r50.onnx")
        self.logger.info("Executor initialized and ready.")

    def load_dummy_gallery(self):
        gallery = []
        for i in range(3):
            dummy_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            emb = self.embedder.get_embedding(dummy_img)
            user_id = f"user_{i+1}"
            gallery.append(common_pb2.FaceEmbedding(user_id=user_id, embedding=emb.tolist()))
        return gallery

    def add_base64_padding(self, b64_string: str) -> str:
        return b64_string + "=" * (-len(b64_string) % 4)

    def ComputeSimilarity(self, request, context):
        self.logger.info("Received ComputeSimilarity request")

        try:
            raw_bytes = request.image_data  # Sudah berupa bytes
            self.logger.info(f"Image data received: type={type(raw_bytes)}, size={len(raw_bytes)} bytes")

            img_array = np.frombuffer(raw_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("cv2.imdecode returned None")

            query_emb = self.embedder.get_embedding(img)

            # Hitung similarity ke semua
            embeddings = load_all_embeddings_from_redis()
            scored_matches = []

            for user_id, emb_bytes in embeddings.items():
                gallery_emb = np.frombuffer(emb_bytes, dtype=np.float32)
                sim = self.cosine_similarity(query_emb, gallery_emb)
                self.logger.info(f"Compared with {user_id}, sim={sim:.4f}")
                scored_matches.append((user_id, sim))

            # Urutkan dari tertinggi ke terendah
            scored_matches.sort(key=lambda x: x[1], reverse=True)

            # Ambil top-3 (atau sesuai kebutuhan)
            top_k = 3
            top_matches = [
                executor_pb2.FaceMatch(user_id=user_id, similarity=sim)
                for user_id, sim in scored_matches[:top_k]
            ]

            # Log hasil terbaik
            if top_matches:
                self.logger.info(f"Top match: {top_matches[0].user_id} (sim={top_matches[0].similarity:.4f})")
            else:
                self.logger.info("No match found")

            return executor_pb2.ComputeResponse(top_matches=top_matches)

        except Exception as e:
            self.logger.warning(f"Failed to decode image from image_data: {e}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Invalid image data")
            return executor_pb2.ComputeResponse()

    def cosine_similarity(self, a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def VerifyFaces(self, request, context):
        self.logger.info("Received VerifyFaces request")
        try:
            img1 = self.decode_image(request.image1)
            img2 = self.decode_image(request.image2)

            emb1 = self.embedder.get_embedding(img1)
            emb2 = self.embedder.get_embedding(img2)

            sim = self.cosine_similarity(emb1, emb2)
            self.logger.info(f"Similarity: {sim:.4f}")

            if sim >= 0.6:
                result = "MATCH"
            elif sim < 0.4:
                result = "NOT_MATCH"
            else:
                result = "ADJUDICATION_NEEDED"

            return executor_pb2.VerifyResponse(similarity=sim, result=result)
        except Exception as e:
            self.logger.error(f"Verification error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return executor_pb2.VerifyResponse()

    def decode_image(self, raw_bytes):
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode failed")
        return img

    def RegisterFace(self, request, context):
        self.logger.info(f"Received RegisterFace: user_id={request.user_id}, name={request.name}, origin={request.origin}")

        try:
            # Decode image
            img = self.decode_image(request.image_data)
            self.logger.info("✅ Image successfully decoded")

            # Generate embedding
            emb = self.embedder.get_embedding(img)
            self.logger.info("✅ Embedding successfully generated")

            # Simpan ke Redis dan PostgreSQL
            emb_bytes = emb.astype(np.float32).tobytes()

            try:
                save_embedding_to_redis(request.user_id, emb_bytes)
                self.logger.info(f"✅ Embedding saved to Redis for {request.user_id}")
            except Exception as e:
                self.logger.error(f"❌ Failed to save embedding to Redis: {e}")

            try:
                save_embedding_to_postgres(request.user_id, emb_bytes)
                self.logger.info(f"✅ Embedding saved to PostgreSQL for {request.user_id}")
            except Exception as e:
                self.logger.error(f"❌ Failed to save embedding to PostgreSQL: {e}")

            return executor_pb2.RegisterResponse(message="OK")

        except Exception as e:
            self.logger.error(f"❌ Failed to register face: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return executor_pb2.RegisterResponse(message="Failed")

def run_executor(port):
    logger = setup_logger(port)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    executor_pb2_grpc.add_ExecutorServiceServicer_to_server(ExecutorServicer(logger), server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info(f"Executor running on port {port}")
    server.start()
    server.wait_for_termination()

def run_all():
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    ports = [6001, 6002, 6003]
    processes = []
    for port in ports:
        p = Process(target=run_executor, args=(port,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--multi":
        run_all()
    else:
        port = int(sys.argv[1]) if len(sys.argv) > 1 else 6001
        run_executor(port)