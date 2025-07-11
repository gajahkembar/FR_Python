# face_engine/driver/main.py

import grpc
import logging
from datetime import datetime
from concurrent import futures
import itertools
import threading
import sys
import os
import numpy as np
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from proto import driver_pb2, driver_pb2_grpc, common_pb2
from proto import executor_pb2, executor_pb2_grpc
from src.embedder import ArcFaceEmbedder
from src.matcher import cosine_similarity
from src.aligner import get_aligned_face

# Logging
log_file = "driver/driver.log"
logging.basicConfig(
    filename=log_file,
    filemode='a',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Driver")

# Daftar executor yang akan diakses round-robin
EXECUTOR_ADDRESSES = [
    "localhost:6001",
    "localhost:6002",
    "localhost:6003"
]
executor_cycle = itertools.cycle(EXECUTOR_ADDRESSES)
cycle_lock = threading.Lock()

class DriverServicer(driver_pb2_grpc.DriverServiceServicer):
    def __init__(self):
        self.executor_cycle = executor_cycle
        self.embedder = ArcFaceEmbedder("models/w600k_r50.onnx")

    def RouteIdentify(self, request, context):
        with cycle_lock:
            executor_addr = next(self.executor_cycle)

        logger.info(f"Routing to Executor at {executor_addr}")

        try:
            with grpc.insecure_channel(executor_addr) as channel:
                stub = executor_pb2_grpc.ExecutorServiceStub(channel)
                compute_request = executor_pb2.ComputeRequest(
                    image_data=request.image_data
                )
                compute_response = stub.ComputeSimilarity(compute_request)

            if not compute_response.top_matches:
                logger.warning("No matches returned from Executor")
                return driver_pb2.IdentifyResult(
                    user_id="",
                    similarity=0.0,
                    message="NO MATCHES",
                    top_matches=[],
                )

            best_match = compute_response.top_matches[0]
            logger.info(f"Top-1 from Executor: {best_match.user_id} with sim={best_match.similarity:.4f}")

            return driver_pb2.IdentifyResult(
                user_id=best_match.user_id,
                similarity=best_match.similarity,
                message="OK",
                top_matches=compute_response.top_matches
            )

        except Exception as e:
            logger.error(f"Executor call failed: {e}")
            return driver_pb2.IdentifyResult(
                user_id="",
                similarity=0.0,
                message="FAILED",
                top_matches=[],
            )
    
    def RouteVerify(self, request, context):
        trx_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
        try:
            logger.info(f"[{trx_id}] Performing local verification in Driver")

            # Decode image1
            img1_array = np.frombuffer(request.image1, dtype=np.uint8)
            img1 = cv2.imdecode(img1_array, cv2.IMREAD_COLOR)
            if img1 is None:
                raise ValueError("Image1 decoding failed")

            # Decode image2
            img2_array = np.frombuffer(request.image2, dtype=np.uint8)
            img2 = cv2.imdecode(img2_array, cv2.IMREAD_COLOR)
            if img2 is None:
                raise ValueError("Image2 decoding failed")

            # Face alignment
            try:
                aligned1 = get_aligned_face(img1)
                aligned2 = get_aligned_face(img2)
            except Exception as e:
                raise ValueError(f"Face alignment failed: {e}")

            # Dapatkan embedding
            emb1 = self.embedder.get_embedding(aligned1)
            emb2 = self.embedder.get_embedding(aligned2)

            # Cosine similarity
            sim = cosine_similarity(emb1, emb2)
            logger.info(f"[{trx_id}] Cosine similarity: {sim:.4f}")

            # Thresholds
            TAR_TH = 0.5
            FAR_TH = 0.3
            if sim >= TAR_TH:
                result = "MATCH"
            elif sim < FAR_TH:
                result = "NOT_MATCH"
            else:
                result = "REVIEW"

            return driver_pb2.VerifyResult(similarity=sim, result=result)

        except Exception as e:
            logger.error(f"[{trx_id}] Local verification failed: {e}")
            return driver_pb2.VerifyResult(similarity=0.0, result="FAILED")

    def RegisterFace(self, request, context):
        trx_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
        try:
            logger.info(f"[{trx_id}] RegisterFace received: user_id={request.user_id}")

            # Decode image
            img_array = np.frombuffer(request.image_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Image decoding failed")

            # Dapatkan embedding
            embedding = self.embedder.get_embedding(img)

            # Kirim ke executor untuk disimpan
            with cycle_lock:
                executor_addr = next(self.executor_cycle)
            logger.info(f"[{trx_id}] Routing RegisterFace to Executor at {executor_addr}")

            with grpc.insecure_channel(executor_addr) as channel:
                stub = executor_pb2_grpc.ExecutorServiceStub(channel)
                executor_req = executor_pb2.RegisterRequest(
                    user_id=request.user_id,
                    image_data=request.image_data,
                    name="",
                    origin=""
                )
                executor_res = stub.RegisterFace(executor_req)

            logger.info(f"[{trx_id}] RegisterFace success: {executor_res.message}")
            return driver_pb2.RegisterResponse(
                user_id=request.user_id,
                message=executor_res.message
            )

        except Exception as e:
            logger.error(f"[{trx_id}] RegisterFace failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return driver_pb2.RegisterResponse(
                user_id=request.user_id,
                message="Failed to register"
            )

def serve(port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    driver_pb2_grpc.add_DriverServiceServicer_to_server(DriverServicer(), server)
    server.add_insecure_port(f'[::]:{port}')
    logger.info(f"Driver running on port {port}")
    server.start()
    server.wait_for_termination()

def run_on_port(port: int):
    log_file = f"driver/driver-{port}.log"
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    serve(port)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    ports = [1968]
    if len(sys.argv) > 1:
        ports = list(map(int, sys.argv[1:]))

    processes = []
    for port in ports:
        proc = multiprocessing.Process(target=run_on_port, args=(port,))
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()