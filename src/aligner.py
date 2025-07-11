import time
from insightface.app import FaceAnalysis
from src.alignment_utils import warp_and_crop_face

app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

def get_aligned_face(img):
    total_start = time.time()

    detect_start = time.time()
    faces = app.get(img)
    detect_end = time.time()

    if not faces:
        raise ValueError("No face detected")

    face = faces[0]
    if face.kps is None:
        raise ValueError("No landmarks found for alignment")

    align_start = time.time()
    aligned = warp_and_crop_face(img, face.kps, image_size=(112, 112))
    align_end = time.time()

    if aligned is None or aligned.size == 0:
        raise ValueError("Face detected but warp_and_crop failed")

    total_end = time.time()

    # print(f"[⏱️] Face Detection Time: {detect_end - detect_start:.4f}s")
    # print(f"[⏱️] Face Alignment Time: {align_end - align_start:.4f}s")
    # print(f"[⏱️] Total Alignment Function Time: {total_end - total_start:.4f}s")

    return aligned