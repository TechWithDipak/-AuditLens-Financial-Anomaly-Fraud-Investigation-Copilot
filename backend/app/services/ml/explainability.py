"""Model Explainability via SHAP (Shapley Additive exPlanations)."""

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import shap

from app.core.config import settings
from app.schemas.evaluation import FeatureAttribution, SHAPExplanationResponse


class ExplainabilityService:
    @staticmethod
    def explain_transaction(
        model_name: str,
        transaction_features: Dict[str, float],
        decision_threshold: float = 0.5,
    ) -> SHAPExplanationResponse:
        """Compute local SHAP attribution values for a single transaction instance."""
        artifacts_dir = Path(settings.ARTIFACTS_DIR)

        # Resolve model artifact
        if "xgb" in model_name.lower():
            artifact_file = artifacts_dir / "xgboost_model.joblib"
            display_name = "XGBoost_Classifier"
        elif "cart" in model_name.lower() or "tree" in model_name.lower():
            artifact_file = artifacts_dir / "cart_model.joblib"
            display_name = "CART_DecisionTree"
        else:
            # Default to XGBoost if available, else CART
            artifact_file = artifacts_dir / "xgboost_model.joblib"
            display_name = "XGBoost_Classifier"
            if not artifact_file.exists():
                artifact_file = artifacts_dir / "cart_model.joblib"
                display_name = "CART_DecisionTree"

        if not artifact_file.exists():
            raise FileNotFoundError(
                f"Trained model artifact '{artifact_file.name}' not found. Please train the model first."
            )

        saved = joblib.load(artifact_file)
        model = saved["model"]
        preprocessor = saved["preprocessor"]
        feature_names = preprocessor.feature_columns

        # Build feature DataFrame aligned with training columns
        input_data = {}
        for col in feature_names:
            input_data[col] = [float(transaction_features.get(col, 0.0))]
        df_single = pd.DataFrame(input_data)

        # Transform using training preprocessor
        X_scaled = preprocessor.transform(df_single)

        # Predicted probability
        prob = float(model.predict_proba(X_scaled)[0, 1])
        pred_label = 1 if prob >= decision_threshold else 0

        # Compute SHAP values
        try:
            explainer = shap.TreeExplainer(model)
            shap_raw = explainer.shap_values(X_scaled)

            # Extract base value
            exp_val = explainer.expected_value
            if isinstance(exp_val, (list, np.ndarray)):
                base_val = float(exp_val[1]) if len(exp_val) > 1 else float(exp_val[0])
            else:
                base_val = float(exp_val)

            # Handle shape variants: (1, n_features, 2) vs (1, n_features)
            if isinstance(shap_raw, list):
                shap_vector = np.asarray(shap_raw[1][0], dtype=float)
            elif isinstance(shap_raw, np.ndarray):
                if shap_raw.ndim == 3 and shap_raw.shape[2] == 2:
                    shap_vector = np.asarray(shap_raw[0, :, 1], dtype=float)
                elif shap_raw.ndim == 2:
                    shap_vector = np.asarray(shap_raw[0, :], dtype=float)
                else:
                    shap_vector = np.asarray(shap_raw.flatten(), dtype=float)
            else:
                shap_vector = np.zeros(len(feature_names), dtype=float)

        except Exception:
            # Safe fallback if TreeExplainer fails
            base_val = 0.5
            shap_vector = np.zeros(len(feature_names), dtype=float)

        # Build attribution list
        attributions: List[FeatureAttribution] = []
        raw_dict: Dict[str, float] = {}

        for feat_name, feat_val, sv in zip(
            feature_names, df_single.iloc[0], shap_vector, strict=False
        ):
            val_float = round(float(sv), 4)
            raw_dict[feat_name] = val_float
            attributions.append(
                FeatureAttribution(
                    feature_name=feat_name,
                    feature_value=round(float(feat_val), 4),
                    shap_value=val_float,
                    contribution="increases_risk" if val_float > 0 else "decreases_risk",
                )
            )

        # Sort by absolute impact
        attributions.sort(key=lambda a: abs(a.shap_value), reverse=True)

        return SHAPExplanationResponse(
            model_name=display_name,
            base_value=round(base_val, 4),
            predicted_probability=round(prob, 4),
            decision_threshold=decision_threshold,
            prediction_label=pred_label,
            top_attributions=attributions[:10],
            raw_shap_values=raw_dict,
        )
