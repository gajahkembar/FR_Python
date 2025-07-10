from pydantic import BaseModel
from typing import List

class FaceMatch(BaseModel):
    user_id: str
    similarity: float
    name: str
    origin: str

class IdentifyResponse(BaseModel):
    user_id: str
    similarity: float
    result: str
    top_matches: List[FaceMatch]