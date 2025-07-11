import grpc
import logging
import sys
import os
import time
import numpy as np
import cv2
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Process, set_start_method

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from proto import executor_pb2, executor_pb2_grpc
from src.embedder import ArcFaceEmbedder
from src.aligner import get_aligned_face
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


def compute_similarity_for_uid(query_emb_bytes, uid, emb_bytes):
    query_emb = np.frombuffer(query_emb_bytes, dtype=np.float32)
    emb = np.frombuffer(emb_bytes, dtype=np.float32)
    norm_a, norm_b = np.linalg.norm(query_emb), np.linalg.norm(emb)
    sim = float(np.dot(query_emb, emb) / (norm_a * norm_b)) if norm_a and norm_b else 0.0
    return uid, sim


class ExecutorServicer(executor_pb2_grpc.ExecutorServiceServicer):
    def __init__(self, logger):
        self.logger = logger
        self.embedder = ArcFaceEmbedder("models/w600k_r50.onnx")
        self.logger.info("🚀 Executor ready.")

    def decode_image(self, raw_bytes):
        img = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return img

    def cosine_similarity(self, a, b):
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (norm_a * norm_b)) if norm_a and norm_b else 0.0

    def ComputeSimilarity(self, request, context):
        trx_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
        self.logger.info(f"[{trx_id}] ▶️ ComputeSimilarity request received")
        try:
            # Decode & align image
            img = self.decode_image(request.image_data)
            aligned = get_aligned_face(img)
            if aligned is None or aligned.size == 0:
                raise ValueError("Alignment failed")

            # Get query embedding
            query_emb = self.embedder.get_embedding(aligned)
            query_emb_norm = query_emb / np.linalg.norm(query_emb)

            # Load gallery embeddings
            embeddings = load_all_embeddings_from_redis()
            if not embeddings:
                self.logger.warning(f"[{trx_id}] ⚠️ Redis gallery empty")
                return executor_pb2.ComputeResponse()

            uid_list = list(embeddings.keys())

            # Convert Redis binary -> numpy array
            try:
                emb_array = np.stack([
                    np.frombuffer(v, np.float32) for v in embeddings.values()
                ])
            except Exception as e:
                self.logger.error(f"[{trx_id}] ❌ Failed to stack embeddings: {e}")
                context.set_code(grpc.StatusCode.INTERNAL)
                return executor_pb2.ComputeResponse()

            # Normalize embeddings for cosine similarity
            emb_norm = emb_array / np.linalg.norm(emb_array, axis=1, keepdims=True)

            # Vectorized cosine similarity
            start = time.time()
            sims = emb_norm @ query_emb_norm
            top_indices = sims.argsort()[::-1][:3]
            top_k = [(uid_list[i], float(sims[i])) for i in top_indices]
            end = time.time()

            self.logger.info(f"[{trx_id}] ⚡ Vector similarity done in {end - start:.4f}s (n={len(uid_list)})")

            if top_k:
                self.logger.info(f"[{trx_id}] ✅ Top match: {top_k[0][0]} (sim={top_k[0][1]:.4f})")
            else:
                self.logger.info(f"[{trx_id}] ⚠️ No match found")

            return executor_pb2.ComputeResponse(top_matches=[
                executor_pb2.FaceMatch(user_id=uid, similarity=sim) for uid, sim in top_k
            ])
        except Exception as e:
            self.logger.warning(f"[{trx_id}] ❌ ComputeSimilarity error: {e}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return executor_pb2.ComputeResponse()

    def VerifyFaces(self, request, context):
        trx_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
        self.logger.info(f"[{trx_id}] ▶️ VerifyFaces request received")
        try:
            img1 = self.decode_image(request.image1)
            img2 = self.decode_image(request.image2)
            aligned1 = get_aligned_face(img1)
            aligned2 = get_aligned_face(img2)
            emb1 = self.embedder.get_embedding(aligned1)
            emb2 = self.embedder.get_embedding(aligned2)

            sim = self.cosine_similarity(emb1, emb2)
            result = "MATCH" if sim >= 0.6 else "NOT_MATCH" if sim < 0.4 else "ADJUDICATION_NEEDED"
            self.logger.info(f"[{trx_id}] ✅ VerifyFaces result: {result} (similarity={sim:.4f})")

            return executor_pb2.VerifyResponse(similarity=sim, result=result)
        except Exception as e:
            self.logger.error(f"[{trx_id}] ❌ Verify error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return executor_pb2.VerifyResponse()

    def RegisterFace(self, request, context):
        trx_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
        self.logger.info(f"[{trx_id}] ▶️ RegisterFace user_id={request.user_id}")
        try:
            img = self.decode_image(request.image_data)
            aligned = get_aligned_face(img)
            if aligned is None or aligned.size == 0:
                raise ValueError("Face alignment failed")

            emb = self.embedder.get_embedding(aligned)
            emb_bytes = emb.astype(np.float32).tobytes()

            try:
                save_embedding_to_redis(request.user_id, emb_bytes)
                self.logger.info(f"[{trx_id}] ✅ Saved to Redis")
            except Exception as e:
                self.logger.warning(f"[{trx_id}] ⚠️ Redis save failed: {e}")

            try:
                save_embedding_to_postgres(request.user_id, emb_bytes)
                self.logger.info(f"[{trx_id}] ✅ Saved to PostgreSQL")
            except Exception as e:
                self.logger.warning(f"[{trx_id}] ⚠️ PostgreSQL save failed: {e}")

            self.logger.info(f"[{trx_id}] ✅ RegisterFace success")
            return executor_pb2.RegisterResponse(message="OK")
        except Exception as e:
            self.logger.error(f"[{trx_id}] ❌ Register failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return executor_pb2.RegisterResponse(message="Failed")


def run_executor(port):
    logger = setup_logger(port)
    server = grpc.server(ThreadPoolExecutor(max_workers=os.cpu_count()))
    executor_pb2_grpc.add_ExecutorServiceServicer_to_server(ExecutorServicer(logger), server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info(f"🟢 Executor running on port {port}")
    server.start()
    server.wait_for_termination()


def run_all():
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    ports = list(range(6001, 6005))
    procs = [Process(target=run_executor, args=(p,)) for p in ports]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--multi":
        run_all()
    else:
        port = int(sys.argv[1]) if len(sys.argv) > 1 else 6001
        run_executor(port)