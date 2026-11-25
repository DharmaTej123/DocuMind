from fastapi import FastAPI
from app.core.config import get_settings
from app.core.logging_config import setup_logging

settings = get_settings()
setup_logging(settings.log_level)  # must run before the imports below (they build the graph)

from app.api import auth, documents, query  # noqa: E402

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
