"""Service for transaction ingestion, filtering, scoring, and explainability."""

import json
import logging

import pandas as pd
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from backend.app.db.models import TransactionModel
from backend.app.schemas.transaction import (
    FeatureContributionSchema,
    TransactionExplanationResponse,
    TransactionListResponse,
    TransactionResponse,
    TransactionScoreResponse,
)
from backend.app.services.audit_service import record_audit_event
from backend.app.services.model_service import get_explainer, get_inference_engine
from ml.data.schema import REQUIRED_FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


def ingest_transactions_dataframe(
    db: Session,
    df: pd.DataFrame,
    actor: str = "analyst_system",
    auto_score: bool = True,
) -> int:
    """Ingest a DataFrame of validated transactions into SQLite with optional scoring.

    Args:
        db: Active database session.
        df: Validated pandas DataFrame with Time, Amount, V1-V28.
        actor: Identity of actor performing upload.
        auto_score: If True, automatically runs ML inference to score rows.

    Returns:
        int: Number of transactions newly ingested.
    """
    if df.empty:
        return 0

    engine = get_inference_engine() if auto_score else None
    predictions_list = None

    if engine is not None:
        try:
            predictions_list = engine.score_batch(df)
        except (ValueError, RuntimeError, KeyError, TypeError, OSError) as exc:
            logger.warning(
                f"Auto-scoring during ingestion failed: {exc}. Ingesting without scores."
            )

    ingested_count = 0
    tx_models: list[TransactionModel] = []

    for idx, (_, row) in enumerate(df.iterrows()):
        tx_id = f"TX_{int(row.get('Time', idx)):08d}_{idx}"

        # Check if already exists
        existing = db.scalar(
            select(TransactionModel.id).where(TransactionModel.transaction_id == tx_id)
        )
        if existing:
            continue

        # Extract PCA features into clean JSON
        features = {
            col: float(row[col]) for col in REQUIRED_FEATURE_COLUMNS if col in row
        }
        features_json = json.dumps(features)

        anomaly_score = None
        fraud_prob = None
        is_flagged = False
        score_type = "unscored"

        if predictions_list is not None and idx < len(predictions_list):
            pred_rec = predictions_list[idx]
            anomaly_score = float(pred_rec.anomaly_score)
            fraud_prob = (
                float(pred_rec.fraud_probability)
                if pred_rec.fraud_probability is not None
                else 0.0
            )
            is_flagged = bool(pred_rec.predicted_class == 1 or fraud_prob >= 0.5)
            score_type = "dual_scoring"

        label = None
        if TARGET_COLUMN in row and pd.notna(row[TARGET_COLUMN]):
            try:
                label = int(row[TARGET_COLUMN])
            except (ValueError, TypeError):
                label = None

        tx = TransactionModel(
            transaction_id=tx_id,
            time=float(row.get("Time", 0.0)),
            amount=float(row.get("Amount", 0.0)),
            features=features_json,
            anomaly_score=anomaly_score,
            fraud_probability=fraud_prob,
            is_flagged=is_flagged,
            score_type=score_type,
            label=label,
        )
        tx_models.append(tx)
        ingested_count += 1

    if tx_models:
        db.add_all(tx_models)
        db.commit()

        record_audit_event(
            db=db,
            event_type="DATASET_INGESTED",
            entity_type="dataset",
            entity_id=f"batch_{ingested_count}",
            actor=actor,
            details={
                "ingested_count": ingested_count,
                "auto_scored": engine is not None,
            },
        )

    return ingested_count


def list_transactions(
    db: Session,
    page: int = 1,
    page_size: int = 50,
    min_amount: float | None = None,
    max_amount: float | None = None,
    is_flagged: bool | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    score_type: str | None = None,
) -> TransactionListResponse:
    """Retrieve filtered and paginated transactions."""
    page = max(1, page)
    page_size = min(max(1, page_size), 100)

    query = select(TransactionModel)

    if min_amount is not None:
        query = query.where(TransactionModel.amount >= min_amount)
    if max_amount is not None:
        query = query.where(TransactionModel.amount <= max_amount)
    if is_flagged is not None:
        query = query.where(TransactionModel.is_flagged == is_flagged)
    if min_score is not None:
        query = query.where(TransactionModel.fraud_probability >= min_score)
    if max_score is not None:
        query = query.where(TransactionModel.fraud_probability <= max_score)
    if score_type:
        query = query.where(TransactionModel.score_type == score_type)

    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0
    total_pages = max(1, (total + page_size - 1) // page_size)

    query = (
        query.order_by(desc(TransactionModel.id))
        .offset((page - 1) * page_size)
        .limit(page_size)
    )

    records = db.scalars(query).all()

    items = []
    for r in records:
        try:
            feat_dict = json.loads(r.features)
        except (json.JSONDecodeError, TypeError):
            feat_dict = {}

        items.append(
            TransactionResponse(
                id=r.id,
                transaction_id=r.transaction_id,
                time=r.time,
                amount=r.amount,
                features=feat_dict,
                anomaly_score=r.anomaly_score,
                fraud_probability=r.fraud_probability,
                is_flagged=r.is_flagged,
                score_type=r.score_type,
                label=r.label,
                created_at=r.created_at,
            )
        )

    return TransactionListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


def get_transaction_by_id(
    db: Session, tx_identifier: str | int
) -> TransactionModel | None:
    """Fetch transaction by either internal DB integer ID or public string transaction_id."""
    if isinstance(tx_identifier, int) or (
        isinstance(tx_identifier, str) and tx_identifier.isdigit()
    ):
        tx = db.scalar(
            select(TransactionModel).where(TransactionModel.id == int(tx_identifier))
        )
        if tx:
            return tx

    return db.scalar(
        select(TransactionModel).where(
            TransactionModel.transaction_id == str(tx_identifier)
        )
    )


def score_transaction_features(
    time: float,
    amount: float,
    features: dict[str, float],
    transaction_id: str = "ADHOC_001",
) -> TransactionScoreResponse:
    """Run real-time inference on arbitrary transaction inputs."""
    engine = get_inference_engine()
    if engine is None:
        # Graceful fallback heuristic when models not yet loaded
        return TransactionScoreResponse(
            transaction_id=transaction_id,
            anomaly_score=0.10,
            fraud_probability=0.05,
            is_flagged=False,
            decision_threshold=0.5,
            score_type="fallback_unloaded",
            model_version_anomaly="unloaded",
            model_version_classifier="unloaded",
        )

    row_data = {"Time": time, "Amount": amount}
    row_data.update(features)
    row_series = pd.Series(row_data)

    pred = engine.score_single(row_series, threshold=0.5)
    fraud_prob = (
        float(pred.fraud_probability) if pred.fraud_probability is not None else 0.0
    )
    is_flagged = bool(pred.predicted_class == 1 or fraud_prob >= 0.5)

    return TransactionScoreResponse(
        transaction_id=str(transaction_id),
        anomaly_score=float(pred.anomaly_score),
        fraud_probability=fraud_prob,
        is_flagged=is_flagged,
        decision_threshold=0.5,
        score_type="dual_scoring",
        model_version_anomaly=pred.anomaly_model_version or "iso_forest_v0.1.0",
        model_version_classifier=pred.classifier_model_version or "logreg_v0.1.0",
    )


def explain_transaction_by_id(
    db: Session,
    tx_identifier: str | int,
) -> TransactionExplanationResponse:
    """Generate SHAP or fallback feature contribution explanation for a stored transaction."""
    tx = get_transaction_by_id(db, tx_identifier)
    if not tx:
        raise ValueError(f"Transaction not found: {tx_identifier}")

    try:
        features_dict = json.loads(tx.features)
    except (json.JSONDecodeError, TypeError):
        features_dict = {}

    row_data = {"Time": tx.time, "Amount": tx.amount}
    row_data.update(features_dict)
    row_series = pd.Series(row_data)

    explainer = get_explainer()
    if explainer is None:
        raise RuntimeError("Explainability service unavailable: Models not loaded.")

    explanation = explainer.explain_transaction(
        row_series,
        transaction_id=tx.transaction_id,
        threshold=0.5,
    )

    top_contrib_schemas = [
        FeatureContributionSchema(
            feature_name=c.feature_name,
            transformed_value=c.transformed_value,
            contribution=c.contribution,
            direction=c.direction,
            raw_value=c.raw_value,
            relative_magnitude=c.relative_magnitude,
            plain_text_description=c.plain_text_description,
        )
        for c in explanation.top_contributions
    ]

    return TransactionExplanationResponse(
        transaction_id=str(explanation.transaction_id),
        model_name=explanation.model_name,
        model_version=explanation.model_version,
        explanation_method=explanation.explanation_method,
        score_type=explanation.score_type,
        prediction_score=explanation.prediction_score,
        decision_threshold=explanation.decision_threshold,
        is_flagged=explanation.is_flagged,
        base_value=explanation.base_value,
        top_contributions=top_contrib_schemas,
        all_contributions=explanation.all_contributions,
        summary_text=explanation.summary_text,
        warnings=explanation.warnings,
        limitations=explanation.limitations,
        timestamp=explanation.timestamp,
    )
