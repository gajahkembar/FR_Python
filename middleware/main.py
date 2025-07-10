from fastapi import FastAPI
from middleware.api.routes import router as api_router

app = FastAPI(
    title="FR Middleware",
    version="1.0.0"
)

app.include_router(api_router, prefix="/api")