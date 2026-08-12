"""
One-command local setup: creates the pgvector extension (if missing) and
all application tables. Idempotent — safe to run multiple times.

Usage:
    python -m app.db.init_db

For Docker Compose users: not needed, docker-compose.yml's db service can
have this wired in as an entrypoint step, or just run it once after
`docker compose up` via:
    docker compose exec backend python -m app.db.init_db
"""
from sqlalchemy import text

from app.core.config import get_settings
from app.db.models import Base
from app.db.session import engine


def init_db():
    settings = get_settings()
    print(f"Connecting to: {settings.database_url.split('@')[-1]}")

    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    print("pgvector extension ready.")

    Base.metadata.create_all(engine)
    print("Tables created:", ", ".join(Base.metadata.tables.keys()))


if __name__ == "__main__":
    init_db()
