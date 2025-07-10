# src/embedder.py
import onnxruntime as ort
import numpy as np
import cv2

class ArcFaceEmbedder:
    def __init__(self, model_path="models/w600k_r50.onnx"):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        img = cv2.resize(image, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 127.5 - 1.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        img = np.expand_dims(img, axis=0)   # NCHW
        return img

    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        input_tensor = self.preprocess(image)
        embedding = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        return (embedding / norm)[0]