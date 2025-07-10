# utils/face_align_utils.py (opsional)
import mediapipe as mp
import cv2
import numpy as np

def get_landmarks_mediapipe(img):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        face = results.multi_face_landmarks[0]
        landmarks = np.array([
            [face.landmark[33].x * img.shape[1], face.landmark[33].y * img.shape[0]],  # mata kiri
            [face.landmark[263].x * img.shape[1], face.landmark[263].y * img.shape[0]],  # mata kanan
            [face.landmark[1].x * img.shape[1], face.landmark[1].y * img.shape[0]],    # hidung
            [face.landmark[61].x * img.shape[1], face.landmark[61].y * img.shape[0]],  # mulut kiri
            [face.landmark[291].x * img.shape[1], face.landmark[291].y * img.shape[0]] # mulut kanan
        ], dtype=np.float32)
        return landmarks