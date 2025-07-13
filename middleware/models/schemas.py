from pydantic import BaseModel
from typing import List, Optional

class FaceMatch(BaseModel):
    user_id: str
    similarity: float
    name: str
    origin: str
    gallery_image: Optional[str] = None

class FaceResult(BaseModel):
    face_index: int
    top_matches: List[FaceMatch]
    crop_image: str

class IdentifyResponse(BaseModel):
    message: str
    faces: List[FaceResult]