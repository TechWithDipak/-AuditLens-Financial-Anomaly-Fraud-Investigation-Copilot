"""Unit 5I: Ensemble Learning (Soft Voting and Stacking)."""

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from app.core.config import settings
from app.schemas.supervised import (
    ModelComparisonItem,
    ModelComparisonResponse,
    SupervisedTrainResponse,
)
from app.services.ml.cart import CARTService
from app.services.ml.metrics import calculate_evaluation_metrics
from app.services.ml.preprocessing import LeakageSafePreprocessor
from app.services.ml.xgboost_model import XGBoostService


class EnsembleService:
    @staticmethod
    def train_and_evaluate(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        ensemble_type: str = "voting",  # "voting" or "stacking"
        random_state: int = 42,
    ) -> SupervisedTrainResponse:
        """Fit ensemble combining CART and XGBoost and evaluate on held-out test split."""
        preprocessor = LeakageSafePreprocessor()
        X_train = preprocessor.fit_transform(train_df)
        y_train = train_df["Class"].to_numpy(dtype=int)

        X_test = preprocessor.transform(test_df)
        y_test = test_df["Class"].to_numpy(dtype=int)

        feature_names = preprocessor.feature_columns

        # Base estimators
        cart = DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=random_state
        )
        neg_count = int(np.sum(y_train == 0))
        pos_count = int(np.sum(y_train == 1))
        scale_pos = float(neg_count / max(pos_count, 1))
        xgb = XGBClassifier(
            max_depth=4,
            learning_rate=0.1,
            n_estimators=100,
            scale_pos_weight=scale_pos,
            eval_metric="aucpr",
            random_state=random_state,
            n_jobs=1,
        )

        if ensemble_type == "stacking":
            meta_learner = LogisticRegression(class_weight="balanced", random_state=random_state)
            model = StackingClassifier(
                estimators=[("cart", cart), ("xgb", xgb)],
                final_estimator=meta_learner,
                cv=3,  # Out-of-fold CV to prevent leakage
                n_jobs=1,
            )
            algo_name = "Stacking Classifier (CART + XGBoost -> Logistic Meta-Learner)"
        else:
            model = VotingClassifier(
                estimators=[("cart", cart), ("xgb", xgb)],
                voting="soft",
                n_jobs=1,
            )
            algo_name = "Soft Voting Ensemble (CART + XGBoost)"

        model.fit(X_train, y_train)

        # Predictions & posterior probabilities
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Standardized metrics
        metrics = calculate_evaluation_metrics(y_test, y_pred, y_prob)

        # Feature importances (averaged from base models if voting, or meta weights if stacking)
        importances: Dict[str, float] = {}
        if hasattr(cart, "feature_importances_") and hasattr(xgb, "feature_importances_"):
            cart_imp = model.named_estimators_["cart"].feature_importances_
            xgb_imp = model.named_estimators_["xgb"].feature_importances_
            avg_imp = (cart_imp + xgb_imp) / 2.0
            for feat, imp in zip(feature_names, avg_imp, strict=False):
                if imp > 0.001:
                    importances[feat] = round(float(imp), 4)
        sorted_importances = dict(
            sorted(importances.items(), key=lambda item: item[1], reverse=True)
        )

        # Save artifact
        artifacts_dir = Path(settings.ARTIFACTS_DIR)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = str(artifacts_dir / f"ensemble_{ensemble_type}_model.joblib")
        joblib.dump({"model": model, "preprocessor": preprocessor}, artifact_path)

        return SupervisedTrainResponse(
            model_name=f"Ensemble_{ensemble_type.capitalize()}",
            algorithm=algo_name,
            hyperparameters={
                "ensemble_type": ensemble_type,
                "base_estimators": ["DecisionTree", "XGBoost"],
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

    @classmethod
    def compare_all_models(
        cls,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        random_state: int = 42,
    ) -> ModelComparisonResponse:
        """Run CART, XGBoost, and Ensemble on the identical held-out test split for fair comparison."""
        cart_res = CARTService.train_and_evaluate(train_df, test_df, random_state=random_state)
        xgb_res = XGBoostService.train_and_evaluate(train_df, test_df, random_state=random_state)
        ens_res = cls.train_and_evaluate(
            train_df, test_df, ensemble_type="voting", random_state=random_state
        )

        items = [
            ModelComparisonItem(
                model_name=cart_res.model_name,
                algorithm_type="Interpretable Decision Tree",
                pr_auc=cart_res.metrics.pr_auc,
                f1_score=cart_res.metrics.f1_score,
                precision=cart_res.metrics.precision,
                recall=cart_res.metrics.recall,
                roc_auc=cart_res.metrics.roc_auc,
            ),
            ModelComparisonItem(
                model_name=xgb_res.model_name,
                algorithm_type="Gradient Boosted Trees",
                pr_auc=xgb_res.metrics.pr_auc,
                f1_score=xgb_res.metrics.f1_score,
                precision=xgb_res.metrics.precision,
                recall=xgb_res.metrics.recall,
                roc_auc=xgb_res.metrics.roc_auc,
            ),
            ModelComparisonItem(
                model_name=ens_res.model_name,
                algorithm_type="Soft-Voting Ensemble",
                pr_auc=ens_res.metrics.pr_auc,
                f1_score=ens_res.metrics.f1_score,
                precision=ens_res.metrics.precision,
                recall=ens_res.metrics.recall,
                roc_auc=ens_res.metrics.roc_auc,
            ),
        ]

        # Determine best model by PR-AUC (primary metric for imbalanced fraud)
        best_model = max(items, key=lambda x: x.pr_auc).model_name

        return ModelComparisonResponse(
            test_sample_count=len(test_df),
            test_fraud_count=int(test_df["Class"].sum()),
            comparison=items,
            best_model_by_prauc=best_model,
        )
