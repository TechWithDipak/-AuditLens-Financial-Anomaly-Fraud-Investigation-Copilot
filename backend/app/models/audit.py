"""SQLAlchemy Model for Immutable Analyst Action Audit Logging."""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, Text

from app.db.base import Base


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    entity_type = Column(String(32), index=True, nullable=False)  # "case", "model", "dataset"
    entity_id = Column(String(64), index=True, nullable=False)
    action = Column(
        String(64), index=True, nullable=False
    )  # "created", "status_changed", "note_added"
    actor = Column(String(128), default="system", nullable=False)
    details = Column(Text, nullable=True)  # JSON or text description of changes
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
