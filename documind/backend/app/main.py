from fastapi import FastAPI
import logging
from app.core.config import get_settings
from app.api import auth, documents, query

settings = get_settings()
logging.basicConfig(level=logging.INFO)
app = FastAPI(title=settings.app_name)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(query.router, prefix="/query", tags=["query"])


@app.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name}
