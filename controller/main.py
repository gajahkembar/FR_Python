# face_engine/controller/main.py

import grpc
import logging
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures as futures
from datetime import datetime
from dotenv import load_dotenv
import sys
import os
import base64
import uuid
import numpy as np
from collections import deque

import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from proto import controller_pb2, controller_pb2_grpc, driver_pb2_grpc, driver_pb2
from src.db import save_metadata_to_postgres
from src.redis_store import save_metadata_to_redis

# Load environment
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Logging
log_file = "controller/controller.log"
logging.basicConfig(
    filename=log_file,
    filemode='a',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Controller")


class ControllerServicer(controller_pb2_grpc.ControllerServiceServicer):
    def __init__(self):
        driver_addrs = os.getenv("DRIVER_PORTS", "127.0.0.1:1968").split(',')
        self.driver_queue = deque(driver_addrs)
        self.driver_channels = {addr: grpc.insecure_channel(addr) for addr in driver_addrs}
        self.driver_stubs = {addr: driver_pb2_grpc.DriverServiceStub(self.driver_channels[addr]) for addr in driver_addrs}
        logger.info(f"Controller initialized with driver pool: {driver_addrs}")

    def get_next_driver(self):
        self.driver_queue.rotate(-1)
        return self.driver_queue[0]

    def decode_image_bytes(self, image_data):
        return base64.b64decode(image_data) if isinstance(image_data, str) else image_data

    def Identify(self, request, context):
        trx_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
        logger.info(f"[{trx_id}] Received Identify request")
        try:
            raw_image_bytes = self.decode_image_bytes(request.image_data)

            driver_addr = self.get_next_driver()
            stub = self.driver_stubs[driver_addr]

            start_time = datetime.now()
            response = stub.RouteIdentify(driver_pb2.ImageQuery(image_data=raw_image_bytes))
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"[{trx_id}] gRPC Identify duration: {duration:.4f}s")

            # Logging semua hasil wajah
            for face in response.results:
                best = face.top_matches[0]
                logger.info(f"[{trx_id}] 🧠 Face-{face.face_index}: match {best.user_id} (sim={best.similarity:.4f})")
                logger.info(f"[{trx_id}] Face-{face.face_index} crop size: {len(face.crop_image)} bytes")

            return controller_pb2.IdentifyResponse(
                message="OK",
                results=response.results
            )

        except Exception as e:
            logger.exception(f"[{trx_id}] Identify failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return controller_pb2.IdentifyResponse(message="FAILED", results=[])

    def VerifyFaces(self, request, context):
        trx_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
        try:
            driver_addr = self.get_next_driver()
            stub = self.driver_stubs[driver_addr]

            start_time = datetime.now()
            res = stub.RouteVerify(driver_pb2.VerifyImagePair(
                image1=request.image1,
                image2=request.image2
            ))
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"[{trx_id}] gRPC Verify duration: {duration:.4f}s")

            return controller_pb2.VerifyResponse(
                similarity=res.similarity,
                result=res.result
            )
        except Exception as e:
            logger.exception(f"[{trx_id}] Verify failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return controller_pb2.VerifyResponse()

    def RegisterFace(self, request, context):
        trx_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
        logger.info(f"[{trx_id}] Received RegisterFace request")
        try:
            user_id = request.user_id or str(uuid.uuid4())

            # Parallel save metadata
            with ThreadPoolExecutor(max_workers=2) as executor:
                executor.submit(save_metadata_to_postgres, user_id, request.name, request.origin)
                executor.submit(save_metadata_to_redis, user_id, request.name, request.origin)

            driver_addr = self.get_next_driver()
            stub = self.driver_stubs[driver_addr]

            res = stub.RegisterFace(driver_pb2.RegisterRequest(
                user_id=user_id,
                image_data=request.image_data,
                name=request.name,
                origin=request.origin
            ))

            logger.info(f"[{trx_id}] Register success: {res.message}")
            return controller_pb2.RegisterResponse(
                user_id=user_id,
                message=res.message
            )
        except Exception as e:
            logger.exception(f"[{trx_id}] RegisterFace failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return controller_pb2.RegisterResponse(user_id="", message="Internal error")

    def DeleteFace(self, request, context):
        trx_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
        logger.info(f"[{trx_id}] Received DeleteFace request")

        try:
            user_id = request.user_id
            if not user_id:
                logger.warning(f"[{trx_id}] Empty user_id in request")
                return controller_pb2.DeleteResponse(message="user_id_missing")

            # Kirim permintaan hapus ke semua driver
            for addr in self.driver_stubs:
                try:
                    self.driver_stubs[addr].DeleteFace(driver_pb2.DeleteRequest(user_id=user_id))
                    logger.info(f"[{trx_id}] Sent delete to driver {addr}")
                except Exception as e:
                    logger.warning(f"[{trx_id}] Driver {addr} failed to delete: {e}")

            return controller_pb2.DeleteResponse(message="success")

        except Exception as e:
            logger.exception(f"[{trx_id}] DeleteFace failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return controller_pb2.DeleteResponse(message="error")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    controller_pb2_grpc.add_ControllerServiceServicer_to_server(ControllerServicer(), server)

    PORT = int(os.getenv("PORT", "1967"))
    server.add_insecure_port(f'[::]:{PORT}')
    logger.info(f"Controller running on port {PORT}")
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()