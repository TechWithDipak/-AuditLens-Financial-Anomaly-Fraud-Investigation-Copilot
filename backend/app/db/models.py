"""Relational ORM database models for transactions, cases, notes, and audit events."""

from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.session import Base


class TransactionModel(Base):
    """Persisted transaction record with scores and feature payload."""

    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    time: Mapped[float] = mapped_column(Float, nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    features: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # JSON-serialized features (V1-V28)
    anomaly_score: Mapped[float | None] = mapped_column(
        Float, nullable=True, index=True
    )
    fraud_probability: Mapped[float | None] = mapped_column(
        Float, nullable=True, index=True
    )
    is_flagged: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, index=True
    )
    score_type: Mapped[str] = mapped_column(
        String(32), default="unscored", nullable=False
    )
    label: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )  # Known ground truth (0 or 1), null if unlabeled
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )


class CaseModel(Base):
    """Investigation case container grouping flagged transactions for human adjudication."""

    __tablename__ = "cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(
        String(32), unique=True, index=True, nullable=False
    )
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(
        String(32), default="NEW", nullable=False, index=True
    )  # NEW, IN_REVIEW, ESCALATED, CLOSED_FALSE_POSITIVE, CLOSED_CONFIRMED_FRAUD
    priority: Mapped[str] = mapped_column(
        String(16), default="MEDIUM", nullable=False, index=True
    )  # LOW, MEDIUM, HIGH, CRITICAL
    assigned_to: Mapped[str | None] = mapped_column(
        String(128), nullable=True, default=None
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    transactions: Mapped[list["CaseTransactionModel"]] = relationship(
        "CaseTransactionModel",
        back_populates="case",
        cascade="all, delete-orphan",
    )
    notes: Mapped[list["CaseNoteModel"]] = relationship(
        "CaseNoteModel",
        back_populates="case",
        cascade="all, delete-orphan",
        order_by="CaseNoteModel.created_at.asc()",
    )


class CaseTransactionModel(Base):
    """Association linking an investigation case to one or more transaction IDs."""

    __tablename__ = "case_transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    transaction_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    case: Mapped["CaseModel"] = relationship("CaseModel", back_populates="transactions")


class CaseNoteModel(Base):
    """Auditable analyst note or timestamped comment attached to an investigation case."""

    __tablename__ = "case_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    author: Mapped[str] = mapped_column(String(128), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    case: Mapped["CaseModel"] = relationship("CaseModel", back_populates="notes")


class AuditEventModel(Base):
    """Immutable audit log tracking all investigator actions, state transitions, and system events."""

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True
    )  # e.g. CASE_CREATED, CASE_STATUS_UPDATED, NOTE_ADDED, TRANSACTION_SCORED
    entity_type: Mapped[str] = mapped_column(
        String(32), nullable=False, index=True
    )  # e.g. case, transaction, dataset, model
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    actor: Mapped[str] = mapped_column(
        String(128), nullable=False, default="system", index=True
    )
    details: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON payload
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
