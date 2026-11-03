"""Service managing machine learning models, inference engines, and evaluation metadata."""

import json
import logging
from pathlib import Path

from backend.app.core.config import settings
from backend.app.schemas.model import (
    ModelEvaluationResponse,
    ModelInfoSchema,
    ModelMetadataResponse,
)
from ml.explainability.explainer import AuditLensExplainer
from ml.models.inference import AuditLensInferenceEngine

logger = logging.getLogger(__name__)

_cached_engine: AuditLensInferenceEngine | None = None
_cached_explainer: AuditLensExplainer | None = None


def get_inference_engine(
    artifacts_dir: str | Path | None = None,
) -> AuditLensInferenceEngine | None:
    """Retrieve or initialize the cached inference engine singleton."""
    global _cached_engine
    if _cached_engine is not None:
        return _cached_engine

    target_dir = Path(artifacts_dir or settings.MODEL_ARTIFACTS_DIR)
    try:
        if (target_dir / "preprocessor.joblib").exists():
            _cached_engine = AuditLensInferenceEngine.load_from_directory(target_dir)
            logger.info(f"Loaded AuditLens inference engine from {target_dir}")
            return _cached_engine
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        logger.warning(
            f"Could not load inference engine from {target_dir}: {exc}. Model scoring will use fallback heuristics."
        )

    return None


def get_explainer() -> AuditLensExplainer | None:
    """Retrieve or initialize the cached transaction explainer."""
    global _cached_explainer
    if _cached_explainer is not None:
        return _cached_explainer

    engine = get_inference_engine()
    if engine is not None and engine.preprocessor is not None:
        # Use classifier if available, otherwise anomaly model
        active_model = engine.classifier_model or engine.anomaly_model
        if active_model is not None:
            try:
                _cached_explainer = AuditLensExplainer(
                    preprocessor=engine.preprocessor,
                    model=active_model,
                    use_shap=True,
                )
                logger.info(
                    f"Initialized AuditLens explainer for model {active_model.model_name}"
                )
                return _cached_explainer
            except (ValueError, RuntimeError, OSError) as exc:
                logger.warning(f"Could not initialize SHAP explainer: {exc}")

    return None


def reset_model_caches() -> None:
    """Reset cached singleton engines (useful for tests or post-retraining)."""
    global _cached_engine, _cached_explainer
    _cached_engine = None
    _cached_explainer = None


def get_models_metadata() -> ModelMetadataResponse:
    """Return runtime metadata and active model configurations."""
    engine = get_inference_engine()

    models_info: dict[str, ModelInfoSchema] = {}
    ready = engine is not None
    threshold = 0.5
    feature_count = (
        len(engine.preprocessor.feature_names_out_)
        if (
            engine
            and engine.preprocessor
            and hasattr(engine.preprocessor, "feature_names_out_")
            and engine.preprocessor.feature_names_out_ is not None
        )
        else 30
    )

    if engine and engine.anomaly_model and hasattr(engine.anomaly_model, "metadata"):
        meta = engine.anomaly_model.metadata
        models_info["anomaly_detector"] = ModelInfoSchema(
            model_name=meta.model_name,
            model_version=meta.model_version,
            model_type="unsupervised_anomaly",
            description=f"{meta.algorithm} ({meta.model_type})",
            features=meta.feature_names,
            hyperparameters=meta.hyperparameters,
        )

    if (
        engine
        and engine.classifier_model
        and hasattr(engine.classifier_model, "metadata")
    ):
        meta = engine.classifier_model.metadata
        models_info["fraud_classifier"] = ModelInfoSchema(
            model_name=meta.model_name,
            model_version=meta.model_version,
            model_type="supervised_classifier",
            description=f"{meta.algorithm} ({meta.model_type})",
            features=meta.feature_names,
            hyperparameters=meta.hyperparameters,
        )

    return ModelMetadataResponse(
        inference_engine_ready=ready,
        operational_threshold=threshold,
        feature_count=feature_count,
        models=models_info,
    )


def get_evaluation_report() -> ModelEvaluationResponse:
    """Read the latest Phase 5 model evaluation report JSON."""
    models_dir = Path(settings.MODEL_ARTIFACTS_DIR)
    eval_json = models_dir / "evaluation_comparison.json"
    baseline_json = models_dir / "baseline_evaluation.json"

    if eval_json.exists():
        try:
            data = json.loads(eval_json.read_text(encoding="utf-8"))
            return ModelEvaluationResponse(
                generated_at=data.get("generated_at"),
                best_model_name=data.get("best_model_name"),
                operational_threshold=data.get("operational_threshold"),
                models_evaluated=data.get("models_evaluated", []),
                dataset_summary=data.get("dataset_summary", {}),
                limitations=data.get("limitations", []),
            )
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning(f"Error parsing evaluation report {eval_json}: {exc}")

    if baseline_json.exists():
        try:
            data = json.loads(baseline_json.read_text(encoding="utf-8"))
            return ModelEvaluationResponse(
                generated_at="baseline_evaluation",
                best_model_name="Logistic Regression Baseline",
                operational_threshold=0.5,
                models_evaluated=[
                    {
                        "model_name": "Logistic Regression Baseline",
                        "metrics": data.get("baseline_metrics", {}),
                    }
                ],
                dataset_summary=data.get("split_summary", {}),
                limitations=[
                    "Baseline evaluation only. Run Phase 5 evaluation comparison for full benchmarking."
                ],
            )
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning(f"Error parsing baseline report: {exc}")

    return ModelEvaluationResponse(
        generated_at=None,
        best_model_name="Uninitialized",
        operational_threshold=0.5,
        models_evaluated=[],
        dataset_summary={},
        limitations=["No evaluation reports found in models directory."],
    )
