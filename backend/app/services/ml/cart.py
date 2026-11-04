"""Unit 5G: Classification and Regression Trees (CART) Baseline Classifier."""

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from app.core.config import settings
from app.schemas.supervised import SupervisedTrainResponse
from app.services.ml.metrics import calculate_evaluation_metrics
from app.services.ml.preprocessing import LeakageSafePreprocessor


class CARTService:
    @staticmethod
    def train_and_evaluate(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        max_depth: int = 5,
        min_samples_leaf: int = 5,
        class_weight: str = "balanced",
        random_state: int = 42,
    ) -> SupervisedTrainResponse:
        """Fit interpretable CART Decision Tree on train split and evaluate on held-out test split."""
        preprocessor = LeakageSafePreprocessor()
        X_train = preprocessor.fit_transform(train_df)
        y_train = train_df["Class"].to_numpy(dtype=int)

        X_test = preprocessor.transform(test_df)
        y_test = test_df["Class"].to_numpy(dtype=int)

        feature_names = preprocessor.feature_columns

        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
        )
        model.fit(X_train, y_train)

        # Predictions & posterior probabilities
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate standardized metrics
        metrics = calculate_evaluation_metrics(y_test, y_pred, y_prob)

        # Feature importances
        importances: Dict[str, float] = {}
        for feat, imp in zip(feature_names, model.feature_importances_, strict=False):
            if imp > 0.001:
                importances[feat] = round(float(imp), 4)
        sorted_importances = dict(
            sorted(importances.items(), key=lambda item: item[1], reverse=True)
        )

        # Human-readable decision rules
        rules = export_text(model, feature_names=feature_names, max_depth=max_depth)

        # Save model artifact
        artifacts_dir = Path(settings.ARTIFACTS_DIR)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = str(artifacts_dir / "cart_model.joblib")
        joblib.dump({"model": model, "preprocessor": preprocessor}, artifact_path)

        return SupervisedTrainResponse(
            model_name="CART_DecisionTree",
            algorithm="Classification and Regression Trees (CART)",
            hyperparameters={
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "class_weight": class_weight,
                "random_state": random_state,
            },
            train_samples=len(train_df),
            test_samples=len(test_df),
            fraud_count_test=int(np.sum(y_test)),
            metrics=metrics,
            feature_importances=sorted_importances,
            rules_text=rules,
            artifact_path=artifact_path,
        )
