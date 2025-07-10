# src/aligner.py
import cv2
import numpy as np

# 5-point landmark standard (mata kiri, mata kanan, hidung, mulut kiri, mulut kanan)
SRC_POINTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def warp_and_crop_face(img: np.ndarray, landmarks: np.ndarray, image_size=(112, 112)):
    dst = SRC_POINTS
    M, _ = cv2.estimateAffinePartial2D(landmarks, dst)
    aligned = cv2.warpAffine(img, M, image_size, borderValue=0.0)
    return aligned