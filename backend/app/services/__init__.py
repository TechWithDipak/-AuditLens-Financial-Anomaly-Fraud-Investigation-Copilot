"""Backend domain services for AuditLens."""

from backend.app.services.audit_service import list_audit_events, record_audit_event
from backend.app.services.case_service import (
    add_case_note,
    create_case,
    get_case,
    list_cases,
    update_case,
)
from backend.app.services.model_service import (
    get_evaluation_report,
    get_explainer,
    get_inference_engine,
    get_models_metadata,
    reset_model_caches,
)
from backend.app.services.transaction_service import (
    explain_transaction_by_id,
    get_transaction_by_id,
    ingest_transactions_dataframe,
    list_transactions,
    score_transaction_features,
)

__all__ = [
    "add_case_note",
    "create_case",
    "explain_transaction_by_id",
    "get_case",
    "get_evaluation_report",
    "get_explainer",
    "get_inference_engine",
    "get_models_metadata",
    "get_transaction_by_id",
    "ingest_transactions_dataframe",
    "list_audit_events",
    "list_cases",
    "list_transactions",
    "record_audit_event",
    "reset_model_caches",
    "score_transaction_features",
    "update_case",
]

