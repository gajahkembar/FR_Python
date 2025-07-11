from insightface.app import FaceAnalysis
from src.alignment_utils import warp_and_crop_face

app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

def get_aligned_face(img):
    faces = app.get(img)
    if not faces:
        raise ValueError("No face detected")

    face = faces[0]
    if face.kps is None:
        raise ValueError("No landmarks found for alignment")

    aligned = warp_and_crop_face(img, face.kps, image_size=(112, 112))
    if aligned is None or aligned.size == 0:
        raise ValueError("Face detected but warp_and_crop failed")
    return aligned