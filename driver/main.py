import grpc
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import itertools
import threading
import sys
import os
import numpy as np
import cv2
import time
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from proto import driver_pb2, driver_pb2_grpc, executor_pb2, executor_pb2_grpc
from src.embedder import ArcFaceEmbedder
from src.matcher import cosine_similarity
from src.aligner import get_aligned_faces

# Logging global (semua driver share satu file)
log_file = "driver/driver.log"
logging.basicConfig(
    filename=log_file,
    filemode='a',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Driver")

# Executor Configuration
EXECUTOR_ADDRESSES = [f"localhost:{port}" for port in range(6001, 6006)]
executor_cycle = itertools.cycle(EXECUTOR_ADDRESSES)
cycle_lock = threading.Lock()

class DriverServicer(driver_pb2_grpc.DriverServiceServicer):
    def __init__(self):
        self.executor_cycle = executor_cycle
        self.embedder = ArcFaceEmbedder("models/w600k_r50.onnx")
        self.stubs = {
            addr: executor_pb2_grpc.ExecutorServiceStub(grpc.insecure_channel(addr))
            for addr in EXECUTOR_ADDRESSES
        }

    def get_executor_stub(self):
        with cycle_lock:
            addr = next(self.executor_cycle)
        logger.info(f"Routing to Executor at {addr}")
        return self.stubs[addr]

    def RouteIdentify(self, request, context):
        trx_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
        try:
            stub = self.get_executor_stub()
            compute_req = executor_pb2.ComputeRequest(image_data=request.image_data)
            response = stub.ComputeSimilarity(compute_req)

            if not response.results:
                return driver_pb2.IdentifyResult(message="NO FACE DETECTED", results=[])

            results = []

            for face in response.results:
                best = face.top_matches[0]
                logger.info(f"[{trx_id}] Face-{face.face_index}: best match {best.user_id} (sim={best.similarity:.4f})")

                top_matches = [
                    executor_pb2.FaceMatch(user_id=match.user_id, similarity=match.similarity)
                    for match in face.top_matches
                ]
                result = executor_pb2.FaceResult(
                    face_index=face.face_index,
                    top_matches=top_matches,
                    crop_image=face.crop_image
                )
                results.append(result)

            return driver_pb2.IdentifyResult(message="OK", results=results)

        except Exception as e:
            logger.error(f"[{trx_id}] ❌ RouteIdentify failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return driver_pb2.IdentifyResult(message="FAILED", results=[])

    def RouteVerify(self, request, context):
        trx_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
        start_time = time.time()
        logger.info(f"[{trx_id}] ▶️ RouteVerify request received")

        try:
            # Decode kedua gambar
            img1 = cv2.imdecode(np.frombuffer(request.image1, dtype=np.uint8), cv2.IMREAD_COLOR)
            img2 = cv2.imdecode(np.frombuffer(request.image2, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img1 is None or img2 is None:
                raise ValueError("Image decoding failed")

            # Proses alignment secara paralel
            with ThreadPoolExecutor(max_workers=2) as executor:
                future1 = executor.submit(get_aligned_faces, img1)
                future2 = executor.submit(get_aligned_faces, img2)
                aligned1, aligned2 = future1.result(), future2.result()

            if not aligned1 or not aligned2:
                raise ValueError("Face alignment failed")

            face1 = aligned1[0]
            face2 = aligned2[0]

            # Proses embedding secara paralel
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Benar (mengirim satu wajah hasil crop)
                emb1_future = executor.submit(self.embedder.get_embedding, face1)
                emb2_future = executor.submit(self.embedder.get_embedding, face2)
                emb1, emb2 = emb1_future.result(), emb2_future.result()

            # Hitung cosine similarity
            sim = cosine_similarity(emb1, emb2)
            result = "MATCH" if sim >= 0.5 else "NOT_MATCH" if sim < 0.3 else "REVIEW"

            logger.info(f"[{trx_id}] ✅ RouteVerify result: {result} (similarity={sim:.4f})")

            total_time = time.time() - start_time
            logger.info(f"[{trx_id}] ⏱️ Total duration: {total_time:.4f}s")

            return driver_pb2.VerifyResult(similarity=sim, result=result)

        except Exception as e:
            logger.error(f"[{trx_id}] ❌ RouteVerify failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return driver_pb2.VerifyResult(similarity=0.0, result="FAILED")

    def RegisterFace(self, request, context):
        trx_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
        try:
            logger.info(f"[{trx_id}] RegisterFace user_id={request.user_id}")
            img = cv2.imdecode(np.frombuffer(request.image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Image decode failed")

            stub = self.get_executor_stub()
            executor_req = executor_pb2.RegisterRequest(
                user_id=request.user_id,
                image_data=request.image_data,
                name=request.name,
                origin=request.origin
            )
            res = stub.RegisterFace(executor_req)

            return driver_pb2.RegisterResponse(user_id=request.user_id, message=res.message)
        except Exception as e:
            logger.error(f"[{trx_id}] RegisterFace failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return driver_pb2.RegisterResponse(user_id=request.user_id, message="FAILED")

def serve(port):
    server = grpc.server(ThreadPoolExecutor(max_workers=os.cpu_count()))
    driver_pb2_grpc.add_DriverServiceServicer_to_server(DriverServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info(f"Driver running on port {port}")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Jalankan semua port default jika tidak diberi argumen
    ports = list(range(1968, 1968 + 3))  # 3 driver
    if len(sys.argv) > 1:
        ports = list(map(int, sys.argv[1:]))

    processes = []
    for port in ports:
        p = multiprocessing.Process(target=serve, args=(port,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()