"""Unit 5H: Boosting — XGBoost Fraud Classifier."""

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from app.core.config import settings
from app.schemas.supervised import SupervisedTrainResponse
from app.services.ml.metrics import calculate_evaluation_metrics
from app.services.ml.preprocessing import LeakageSafePreprocessor


class XGBoostService:
    @staticmethod
    def train_and_evaluate(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> SupervisedTrainResponse:
        """Fit XGBoost gradient boosted trees with class imbalance weighting and evaluate on test split."""
        preprocessor = LeakageSafePreprocessor()
        X_train = preprocessor.fit_transform(train_df)
        y_train = train_df["Class"].to_numpy(dtype=int)

        X_test = preprocessor.transform(test_df)
        y_test = test_df["Class"].to_numpy(dtype=int)

        feature_names = preprocessor.feature_columns

        # Calculate scale_pos_weight for class imbalance
        neg_count = int(np.sum(y_train == 0))
        pos_count = int(np.sum(y_train == 1))
        scale_pos_weight = float(neg_count / max(pos_count, 1))

        model = XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=random_state,
            n_jobs=1,  # Single thread for deterministic reproducibility across platforms
        )
        model.fit(X_train, y_train)

        # Predictions & posterior probabilities
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Standardized metrics
        metrics = calculate_evaluation_metrics(y_test, y_pred, y_prob)

        # Feature importances
        raw_importances = model.feature_importances_
        importances: Dict[str, float] = {}
        for feat, imp in zip(feature_names, raw_importances, strict=False):
            if imp > 0.001:
                importances[feat] = round(float(imp), 4)
        sorted_importances = dict(
            sorted(importances.items(), key=lambda item: item[1], reverse=True)
        )

        # Save model artifact
        artifacts_dir = Path(settings.ARTIFACTS_DIR)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = str(artifacts_dir / "xgboost_model.joblib")
        joblib.dump({"model": model, "preprocessor": preprocessor}, artifact_path)

        return SupervisedTrainResponse(
            model_name="XGBoost_Classifier",
            algorithm="Gradient Tree Boosting (XGBoost)",
            hyperparameters={
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "n_estimators": n_estimators,
                "scale_pos_weight": round(scale_pos_weight, 2),
                "eval_metric": "aucpr",
                "random_state": random_state,
            },
            train_samples=len(train_df),
            test_samples=len(test_df),
            fraud_count_test=int(np.sum(y_test)),
            metrics=metrics,
            feature_importances=sorted_importances,
            rules_text=None,
            artifact_path=artifact_path,
        )
