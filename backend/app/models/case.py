"""SQLAlchemy Models for Case Management and Investigation Notes."""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.db.base import Base


class Case(Base):
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    case_number = Column(String(64), unique=True, index=True, nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(
        String(32), default="NEW", index=True, nullable=False
    )  # NEW, UNDER_REVIEW, ESCALATED, CLOSED_BENIGN, CLOSED_SUSPICIOUS
    priority = Column(
        String(16), default="MEDIUM", index=True, nullable=False
    )  # LOW, MEDIUM, HIGH, CRITICAL
    assigned_analyst = Column(String(128), default="analyst_default", nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    notes = relationship(
        "CaseNote",
        back_populates="case",
        cascade="all, delete-orphan",
        order_by="desc(CaseNote.created_at)",
    )
    transactions = relationship(
        "CaseTransaction", back_populates="case", cascade="all, delete-orphan"
    )


class CaseNote(Base):
    __tablename__ = "case_notes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    case_id = Column(
        Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    author = Column(String(128), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    case = relationship("Case", back_populates="notes")


class CaseTransaction(Base):
    __tablename__ = "case_transactions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    case_id = Column(
        Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    transaction_index = Column(Integer, nullable=False)
    amount = Column(Float, nullable=False)
    fraud_probability = Column(Float, nullable=True)
    anomaly_score = Column(Float, nullable=True)
    added_reason = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    case = relationship("Case", back_populates="transactions")
