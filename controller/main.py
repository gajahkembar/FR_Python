# face_engine/controller/main.py

import grpc
from concurrent import futures
from datetime import datetime
import logging
import sys
import os
import numpy as np
import itertools
import base64
from dotenv import load_dotenv
import uuid
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from proto import controller_pb2, controller_pb2_grpc, driver_pb2_grpc, driver_pb2, executor_pb2, executor_pb2_grpc, common_pb2
from src.db_log import insert_log
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

# Dummy embedder
def dummy_generate_embedding(image_data: bytes):
    np.random.seed(len(image_data))  # Seed based on input length
    return list(np.random.rand(512).astype(np.float32))

class ControllerServicer(controller_pb2_grpc.ControllerServiceServicer):
    def __init__(self):
        driver_addrs = os.getenv("DRIVER_PORTS", "127.0.0.1:1968").split(',')
        executor_addrs = os.getenv("EXECUTOR_PORTS", "127.0.0.1:6001,127.0.0.1:6002,127.0.0.1:6003").split(',')
        
        self.driver_cycle = itertools.cycle(driver_addrs)
        self.executor_cycle = itertools.cycle(executor_addrs)

        logger.info(f"Controller initialized with driver pool: {driver_addrs}")
        logger.info(f"Controller initialized with executor pool: {executor_addrs}")

    def Identify(self, request, context):
        logger.info("Received Identify request")
        trx_id = datetime.now().strftime('%Y%m%d%H%M%S%f')

        try:
            # Decode base64 to raw bytes
            try:
                logger.info(f"[{trx_id}] image_data type before decode: {type(request.image_data)}")

                if isinstance(request.image_data, str):
                    raw_image_bytes = base64.b64decode(request.image_data)
                else:
                    raw_image_bytes = request.image_data

                logger.info(f"[{trx_id}] raw_image_bytes type after decode: {type(raw_image_bytes)}, size={len(raw_image_bytes)} bytes")
            except Exception as decode_err:
                logger.error(f"[{trx_id}] Failed to decode base64 image: {decode_err}")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Invalid base64 image")
                return controller_pb2.IdentifyResponse()

            # Round-robin pick driver
            driver_address = next(self.driver_cycle)
            logger.info(f"[{trx_id}] Routing to Driver at {driver_address}")

            # Kirim raw image ke driver
            with grpc.insecure_channel(driver_address) as channel:
                stub = driver_pb2_grpc.DriverServiceStub(channel)
                image_query = driver_pb2.ImageQuery(image_data=raw_image_bytes)
                response = stub.RouteIdentify(image_query)

            logger.info(f"[{trx_id}] Result: user={response.user_id}, similarity={response.similarity:.4f}, msg={response.message}")

            # Konfigurasi threshold
            MATCH_THRESHOLD = 0.5
            REVIEW_THRESHOLD = 0.3

            if response.similarity >= MATCH_THRESHOLD:
                result = "MATCH"
            elif response.similarity >= REVIEW_THRESHOLD:
                result = "REVIEW"
            else:
                result = "NOT_MATCH"
            
            return controller_pb2.IdentifyResponse(
                user_id=response.user_id,
                similarity=response.similarity,
                result=result,
                top_matches=response.top_matches
            )

        except grpc.RpcError as e:
            logger.error(f"[{trx_id}] gRPC error: {e.code()} {e.details()}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return controller_pb2.IdentifyResponse()
        except Exception as e:
            logger.exception(f"[{trx_id}] Unexpected error in Identify")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return controller_pb2.IdentifyResponse()

    def VerifyFaces(self, request, context):
        trx_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
        try:
            # Pilih driver secara round-robin
            driver_address = next(self.driver_cycle)
            logger.info(f"[{trx_id}] Routing to Driver (Verify) at {driver_address}")

            with grpc.insecure_channel(driver_address) as channel:
                stub = driver_pb2_grpc.DriverServiceStub(channel)
                verify_req = driver_pb2.VerifyImagePair(
                    image1=request.image1,
                    image2=request.image2
                )
                res = stub.RouteVerify(verify_req)

            logger.info(f"[{trx_id}] Verify result: sim={res.similarity:.4f}, result={res.result}")            
            return controller_pb2.VerifyResponse(
                similarity=res.similarity,
                result=res.result
            )

        except grpc.RpcError as e:
            logger.error(f"[{trx_id}] gRPC error in Verify: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return controller_pb2.VerifyResponse()
        except Exception as e:
            logger.exception(f"[{trx_id}] Unexpected error in Verify")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return controller_pb2.VerifyResponse()

    def RegisterFace(self, request, context):
        print(f"✅ masuk ke Controller.RegisterFace: {request.user_id}")
        trx_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
        try:
            logger.info(f"[{trx_id}] Received RegisterFace request")

            # Buat UUID untuk user baru
            user_id = request.user_id
            logger.info(f"[{trx_id}] Generated UUID: {user_id}")

            # Simpan metadata ke PostgreSQL dan Redis
            save_metadata_to_postgres(user_id, request.name, request.origin)
            save_metadata_to_redis(user_id, request.name, request.origin)

            # Ambil driver secara round-robin
            driver_address = next(self.driver_cycle)
            logger.info(f"[{trx_id}] Routing RegisterFace to Driver at {driver_address}")

            with grpc.insecure_channel(driver_address) as channel:
                stub = driver_pb2_grpc.DriverServiceStub(channel)
                register_req = driver_pb2.RegisterRequest(
                    image_data=request.image_data,
                    user_id=user_id
                )
                res = stub.RegisterFace(register_req)

            logger.info(f"[{trx_id}] RegisterFace success: user_id={res.user_id}, message={res.message}")
            return controller_pb2.RegisterResponse(
                user_id=user_id,
                message=res.message
            )

        except grpc.RpcError as e:
            logger.error(f"[{trx_id}] gRPC error in RegisterFace: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return controller_pb2.RegisterResponse(user_id="", message="gRPC error")
        except Exception as e:
            logger.exception(f"[{trx_id}] Unexpected error in RegisterFace")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return controller_pb2.RegisterResponse(user_id="", message="Internal error")

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