"""Explainability service generating SHAP and fallback feature attributions."""

import logging
from typing import Any

import numpy as np
import pandas as pd

from ml.data.schema import REQUIRED_FEATURE_COLUMNS
from ml.explainability.schemas import (
    FeatureContribution,
    GlobalFeatureImportance,
    TransactionExplanation,
)
from ml.models.anomaly import IsolationForestAnomalyDetector
from ml.models.classifier import LogisticRegressionFraudClassifier
from ml.models.random_forest import RandomForestFraudClassifier
from ml.preprocessing.pipeline import AuditLensPreprocessor

logger = logging.getLogger(__name__)


STANDARD_CAVEATS = [
    "Feature contributions reflect statistical model weighting, NOT legal proof of intent or causality.",
    "PCA variables (V1–V28) are anonymized orthogonal dimensions; physical merchant/user semantics are masked.",
    "Advisory only: High scores prioritize analyst review; an analyst must independently verify facts.",
]


class AuditLensExplainer:
    """Unified model explainability engine with native SHAP and robust fallback attribution."""

    def __init__(
        self,
        preprocessor: AuditLensPreprocessor,
        model: LogisticRegressionFraudClassifier
        | RandomForestFraudClassifier
        | IsolationForestAnomalyDetector,
        background_data: np.ndarray | None = None,
        use_shap: bool = True,
    ):
        self.preprocessor = preprocessor
        self.model = model
        self.background_data = background_data
        self.use_shap = use_shap

        self.shap_explainer: Any = None
        self.method: str = "fallback_attribution"
        self._base_value: float = 0.0

        if not getattr(self.model, "is_fitted", False):
            raise ValueError(
                f"Model {self.model.model_name} must be fitted before creating explainer."
            )

        # Attempt to initialize SHAP
        if self.use_shap:
            self._init_shap()

    def _init_shap(self) -> None:
        """Initialize SHAP explainer based on model architecture with graceful fallback."""
        try:
            import shap

            if isinstance(self.model, LogisticRegressionFraudClassifier):
                if self.background_data is not None:
                    # Use a sample of background data (max 100 rows for efficiency)
                    bg_sample = self.background_data[:100]
                    self.shap_explainer = shap.LinearExplainer(
                        self.model.model,
                        bg_sample,
                        feature_perturbation="interventional",
                    )
                else:
                    self.shap_explainer = shap.LinearExplainer(
                        self.model.model,
                        masker=shap.maskers.Independent(
                            np.zeros((1, len(self.preprocessor.output_feature_names_)))
                        ),
                    )
                self.method = "shap_linear"
                self._base_value = float(self.shap_explainer.expected_value)

            elif isinstance(self.model, RandomForestFraudClassifier):
                self.shap_explainer = shap.TreeExplainer(self.model.model)
                self.method = "shap_tree"
                # TreeExplainer expected value for binary classification
                ev = self.shap_explainer.expected_value
                self._base_value = float(
                    ev[1] if isinstance(ev, (list, np.ndarray)) and len(ev) > 1 else ev
                )

            elif isinstance(self.model, IsolationForestAnomalyDetector):
                # Isolation Forest uses fallback distance attribution
                self.method = "isolation_forest_distance_attribution"
                self._base_value = 0.5

        except Exception as exc:
            logger.warning(
                f"SHAP initialization failed for {self.model.model_name}: {exc}. "
                "Falling back to native mathematical feature attribution.",
                exc_info=exc,
            )
            self.shap_explainer = None
            self.method = "fallback_linear_attribution"

    def _compute_fallback_contributions(self, X_trans: np.ndarray) -> np.ndarray:
        """Compute exact mathematical feature contributions when SHAP is unavailable."""
        n_samples, n_features = X_trans.shape

        if isinstance(self.model, LogisticRegressionFraudClassifier):
            # For linear models: contribution_i = coef_i * (x_i - mean_i)
            coefs = self.model.model.coef_[0]
            # Contribution vector
            return X_trans * coefs

        elif isinstance(self.model, RandomForestFraudClassifier):
            # Tree Gini importance weighted deviation from median
            importances = self.model.model.feature_importances_
            medians = np.zeros(n_features)
            return (X_trans - medians) * importances

        elif isinstance(self.model, IsolationForestAnomalyDetector):
            # Statistical deviation relative to training median
            return np.abs(X_trans)

        return np.zeros((n_samples, n_features))

    def explain_transaction(
        self,
        transaction: dict[str, Any] | pd.Series | pd.DataFrame,
        transaction_id: str | int = 0,
        threshold: float = 0.5,
        top_k: int = 5,
    ) -> TransactionExplanation:
        """Generate a complete, technically grounded explanation for an individual transaction.

        Args:
            transaction: Transaction record containing raw feature inputs.
            transaction_id: Identifier for case tracking.
            threshold: Decision threshold for flag status.
            top_k: Number of top influential features to report.

        Returns:
            TransactionExplanation: Complete explanation contract.
        """
        # Convert to DataFrame
        if isinstance(transaction, dict):
            df = pd.DataFrame([transaction])
        elif isinstance(transaction, pd.Series):
            df = pd.DataFrame([transaction.to_dict()])
        else:
            df = transaction.copy()

        # Validate required columns
        missing = [col for col in REQUIRED_FEATURE_COLUMNS if col not in df.columns]
        warnings_list: list[str] = []
        if missing:
            raise ValueError(f"Transaction missing required feature columns: {missing}")

        raw_row = df[REQUIRED_FEATURE_COLUMNS].iloc[0]

        # Preprocess features
        X_trans = self.preprocessor.transform(df[REQUIRED_FEATURE_COLUMNS])
        feature_names = self.preprocessor.output_feature_names_

        # 1. Obtain Prediction Score
        if isinstance(
            self.model, (LogisticRegressionFraudClassifier, RandomForestFraudClassifier)
        ):
            score = float(self.model.predict_fraud_probabilities(X_trans)[0])
            score_type = "fraud_probability"
            is_flagged = score >= threshold
        else:
            scores, _ = self.model.score_transactions(X_trans)
            score = float(scores[0])
            score_type = "anomaly_score"
            is_flagged = score >= threshold

        # 2. Compute Attributions (SHAP or Fallback)
        attributions: np.ndarray | None = None
        current_method = self.method

        if self.shap_explainer is not None:
            try:
                shap_vals = self.shap_explainer.shap_values(X_trans)
                if isinstance(shap_vals, list) and len(shap_vals) > 1:
                    # Binary classification: index 1 represents positive fraud class
                    attributions = np.asarray(shap_vals[1])[0]
                elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                    attributions = shap_vals[0, :, 1]
                else:
                    attributions = np.asarray(shap_vals)[0]
            except Exception as exc:
                logger.warning(
                    f"SHAP explanation failed on instance: {exc}. Using fallback.",
                    exc_info=exc,
                )
                warnings_list.append(
                    f"SHAP attribution failed: {exc!s}. Applied fallback attribution."
                )
                current_method = "fallback_attribution"

        if attributions is None:
            attributions = self._compute_fallback_contributions(X_trans)[0]

        attributions = np.asarray(attributions, dtype=float)

        # 3. Compile Feature Contributions
        total_abs = float(np.sum(np.abs(attributions))) + 1e-9
        contributions: list[FeatureContribution] = []
        all_contribs_dict: dict[str, float] = {}

        for i, feat_name in enumerate(feature_names):
            val_trans = float(X_trans[0, i])
            contrib = float(attributions[i])
            all_contribs_dict[feat_name] = round(contrib, 4)

            # Map raw value if feature exists in raw columns
            raw_val = float(raw_row[feat_name]) if feat_name in raw_row else None

            # Direction
            if contrib > 0.001:
                direction = "increases_risk"
                desc = (
                    f"Feature '{feat_name}' (value: {val_trans:.2f}) increased the fraud score by "
                    f"+{contrib:.3f} relative to typical baseline patterns."
                )
            elif contrib < -0.001:
                direction = "decreases_risk"
                desc = (
                    f"Feature '{feat_name}' (value: {val_trans:.2f}) reduced the fraud score by "
                    f"{contrib:.3f} (consistent with typical legitimate activity)."
                )
            else:
                direction = "neutral"
                desc = f"Feature '{feat_name}' had negligible impact on this transaction's score."

            contributions.append(
                FeatureContribution(
                    feature_name=feat_name,
                    transformed_value=round(val_trans, 4),
                    contribution=round(contrib, 4),
                    direction=direction,
                    raw_value=round(raw_val, 2) if raw_val is not None else None,
                    relative_magnitude=round(abs(contrib) / total_abs, 4),
                    plain_text_description=desc,
                )
            )

        # 4. Rank by absolute contribution
        contributions_sorted = sorted(
            contributions, key=lambda c: abs(c.contribution), reverse=True
        )
        top_k_contributions = contributions_sorted[:top_k]

        # 5. Build Analyst Summary Text
        top_elevating = [
            c.feature_name
            for c in top_k_contributions
            if c.direction == "increases_risk"
        ]
        top_reducing = [
            c.feature_name
            for c in top_k_contributions
            if c.direction == "decreases_risk"
        ]

        summary_parts = [
            (
                f"Model '{self.model.model_name}' (v{self.model.model_version}) assigned a {score_type.replace('_', ' ')} "
                f"of {score:.3f} (operational review threshold: {threshold:.2f})."
            )
        ]
        if is_flagged:
            summary_parts.append(
                f"FLAGGED FOR REVIEW: Score exceeds operational threshold. Key risk-elevating signals: {', '.join(top_elevating) if top_elevating else 'General atypicality'}."
            )
        else:
            summary_parts.append(
                f"WITHIN NORMAL BOUNDS: Score is below operational threshold. Stabilizing signals: {', '.join(top_reducing) if top_reducing else 'Conforms to baseline'}."
            )

        summary_text = " ".join(summary_parts)

        return TransactionExplanation(
            transaction_id=transaction_id,
            model_name=self.model.model_name,
            model_version=self.model.model_version,
            explanation_method=current_method,
            score_type=score_type,
            prediction_score=score,
            decision_threshold=threshold,
            is_flagged=is_flagged,
            base_value=round(self._base_value, 4),
            top_contributions=top_k_contributions,
            all_contributions=all_contribs_dict,
            summary_text=summary_text,
            warnings=warnings_list,
            limitations=STANDARD_CAVEATS,
        )

    def explain_batch(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5,
        top_k: int = 5,
    ) -> list[TransactionExplanation]:
        """Generate explanations for a batch of transactions."""
        explanations: list[TransactionExplanation] = []
        for idx in range(len(df)):
            tx_id = df.index[idx]
            row_df = df.iloc[[idx]]
            exp = self.explain_transaction(
                transaction=row_df,
                transaction_id=tx_id,
                threshold=threshold,
                top_k=top_k,
            )
            explanations.append(exp)
        return explanations

    def get_global_feature_importance(self, top_n: int = 10) -> GlobalFeatureImportance:
        """Compute global feature importance ranking across the model."""
        feature_names = self.preprocessor.output_feature_names_

        if isinstance(self.model, LogisticRegressionFraudClassifier):
            # Mean absolute coefficient weight
            coefs = np.abs(self.model.model.coef_[0])
            sorted_idx = np.argsort(coefs)[::-1]
            method = "absolute_logistic_regression_coefficients"
            rankings = [
                {"feature": feature_names[i], "importance": round(float(coefs[i]), 4)}
                for i in sorted_idx[:top_n]
            ]

        elif isinstance(self.model, RandomForestFraudClassifier):
            # Gini importance
            importances = self.model.model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            method = "mean_impurity_decrease_gini"
            rankings = [
                {
                    "feature": feature_names[i],
                    "importance": round(float(importances[i]), 4),
                }
                for i in sorted_idx[:top_n]
            ]

        else:
            method = "anomaly_distance_heuristic"
            rankings = [
                {"feature": f, "importance": 1.0} for f in feature_names[:top_n]
            ]

        return GlobalFeatureImportance(
            model_name=self.model.model_name,
            model_version=self.model.model_version,
            method=method,
            feature_rankings=rankings,
        )
