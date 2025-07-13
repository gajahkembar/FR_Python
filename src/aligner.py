import time
from insightface.app import FaceAnalysis
from src.alignment_utils import warp_and_crop_face

app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

def get_aligned_faces(img):
    total_start = time.time()

    detect_start = time.time()
    faces = app.get(img)
    detect_end = time.time()

    if not faces:
        raise ValueError("No face detected")

    aligned_faces = []

    align_start = time.time()
    for face in faces:
        if face.kps is not None:
            aligned = warp_and_crop_face(img, face.kps, image_size=(112, 112))
            if aligned is not None and aligned.size != 0:
                aligned_faces.append(aligned)
    align_end = time.time()

    total_end = time.time()
    # print(f"[⏱️] Total faces detected: {len(faces)}, aligned: {len(aligned_faces)}")
    # print(f"[⏱️] Detection Time: {detect_end - detect_start:.4f}s | Alignment Time: {align_end - align_start:.4f}s | Total: {total_end - total_start:.4f}s")

    return aligned_faces