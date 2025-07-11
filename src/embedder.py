# src/embedder.py
import onnxruntime as ort
import numpy as np
import cv2
import time
import logging

logger = logging.getLogger("Embedder")

class ArcFaceEmbedder:
    def __init__(self, model_path="models/w600k_r50.onnx"):
        logger.info(f"📦 Loading ONNX model from: {model_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        start_load = time.time()
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        end_load = time.time()

        logger.info(f"✅ Model loaded in {end_load - start_load:.4f}s")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        img = cv2.resize(image, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 127.5 - 1.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        img = np.expand_dims(img, axis=0)   # NCHW
        return img

    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        input_tensor = self.preprocess(image)

        start_infer = time.time()
        embedding = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        end_infer = time.time()

        logger.debug(f"🧠 ONNX inference time: {end_infer - start_infer:.4f}s")

        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        return (embedding / norm)[0]