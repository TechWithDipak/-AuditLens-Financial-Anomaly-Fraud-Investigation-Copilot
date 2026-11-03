"""Database engine, session management, and table lifecycle."""

from collections.abc import Generator
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.app.core.config import settings


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy ORM models."""


# SQLite-specific connect args (check_same_thread=False for async FastAPI threadpools)
connect_args: dict[str, Any] = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=settings.DEBUG,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def init_db(target_engine=None) -> None:
    """Create all relational database tables."""
    eng = target_engine or engine
    Base.metadata.create_all(bind=eng)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a thread-local database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
