from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from middleware.services.grpc_client import register_face, identify_face, verify_face
from middleware.models.schemas import IdentifyResponse

router = APIRouter()

@router.post("/register")
async def register_endpoint(
    file: UploadFile = File(...),
    name: str = Form(...),
    origin: str = Form(...)
):
    try:
        image_bytes = await file.read()
        result = register_face(image_bytes, name, origin)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/identify", response_model=IdentifyResponse)
async def identify_endpoint(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = identify_face(image_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/verify")
async def verify_endpoint(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        img1 = await file1.read()
        img2 = await file2.read()
        result = verify_face(img1, img2)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))