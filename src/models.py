"""Machine learning models: PCA, Customer K-Means, CART, XGBoost, and Soft Voting Ensemble."""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    precision_recall_curve,
    auc,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

from src.config import ARTIFACTS_DIR, RANDOM_SEED


class CustomerSegmentation:
    """Customer-level K-Means segmentation on behavioral aggregate profiles."""

    def __init__(self, n_clusters: int = 4, random_state: int = RANDOM_SEED):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.feature_cols: List[str] = []
        self.customer_profiles_: Dict[int, Dict[str, Any]] = {}
        self.silhouette_: float = 0.0
        self.cluster_summaries_: List[Dict[str, Any]] = []

    def _build_customer_vectors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate one row per cardholder."""
        # Temporal span per customer in days
        min_date = df.groupby("cc_num")["trans_date_trans_time"].min()
        max_date = df.groupby("cc_num")["trans_date_trans_time"].max()
        span_days = ((max_date - min_date).dt.total_seconds() / 86400.0).clip(lower=1.0)

        txn_counts = df.groupby("cc_num")["amt"].count()
        txns_per_day = txn_counts / span_days

        avg_amt = df.groupby("cc_num")["amt"].mean()
        amt_std = df.groupby("cc_num")["amt"].std().fillna(0.0)
        avg_dist = df.groupby("cc_num")["distance_km"].mean()

        # Category shares for top categories
        top_cats = ["grocery_pos", "gas_transport", "shopping_net", "misc_net", "entertainment"]
        cat_piv = pd.crosstab(df["cc_num"], df["category"], normalize="index")
        for cat in top_cats:
            if cat not in cat_piv.columns:
                cat_piv[cat] = 0.0
        cat_shares = cat_piv[top_cats]

        cust_df = pd.DataFrame(
            {
                "avg_amount": avg_amt,
                "amount_std": amt_std,
                "txns_per_day": txns_per_day,
                "avg_distance": avg_dist,
            }
        ).join(cat_shares)

        return cust_df.fillna(0.0)

    def fit(self, df: pd.DataFrame) -> "CustomerSegmentation":
        """Fit K-Means clustering on customer behavioral profiles."""
        cust_df = self._build_customer_vectors(df)
        self.feature_cols = list(cust_df.columns)

        X_scaled = self.scaler.fit_transform(cust_df)
        self.kmeans.fit(X_scaled)

        # Silhouette score
        if len(cust_df) > self.n_clusters:
            # Subsample if large for fast calculation
            sample_size = min(2000, len(cust_df))
            self.silhouette_ = float(
                silhouette_score(
                    X_scaled,
                    self.kmeans.labels_,
                    sample_size=sample_size,
                    random_state=self.random_state,
                )
            )
        else:
            self.silhouette_ = 0.0

        # Assign clusters to customers
        cust_df["cluster"] = self.kmeans.labels_
        # Distance to centroid
        distances = np.linalg.norm(X_scaled - self.kmeans.cluster_centers_[self.kmeans.labels_], axis=1)
        cust_df["distance_to_center"] = distances

        # Store profile lookup dict
        for cc, row in cust_df.iterrows():
            self.customer_profiles_[int(cc)] = {
                "cluster": int(row["cluster"]),
                "distance_to_center": float(row["distance_to_center"]),
                "avg_amount": float(row["avg_amount"]),
                "amount_std": float(row["amount_std"]),
                "txns_per_day": float(row["txns_per_day"]),
                "avg_distance": float(row["avg_distance"]),
            }

        # Compute cluster summary descriptions
        self.cluster_summaries_ = []
        for c in range(self.n_clusters):
            c_mask = self.kmeans.labels_ == c
            c_count = int(np.sum(c_mask))
            c_data = cust_df[cust_df["cluster"] == c]
            self.cluster_summaries_.append(
                {
                    "cluster_id": c,
                    "customer_count": c_count,
                    "avg_amount": float(c_data["avg_amount"].mean()),
                    "txns_per_day": float(c_data["txns_per_day"].mean()),
                    "avg_distance": float(c_data["avg_distance"].mean()),
                }
            )

        return self

    def get_customer_features(self, cc_nums: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (cluster_ids, distances_to_center) for a series of cc_nums."""
        clusters = np.empty(len(cc_nums), dtype=int)
        dists = np.empty(len(cc_nums), dtype=float)

        default_cluster = 0
        default_dist = float(np.mean([p["distance_to_center"] for p in self.customer_profiles_.values()])) if self.customer_profiles_ else 1.0

        for i, cc in enumerate(cc_nums):
            cc_int = int(cc)
            if cc_int in self.customer_profiles_:
                clusters[i] = self.customer_profiles_[cc_int]["cluster"]
                dists[i] = self.customer_profiles_[cc_int]["distance_to_center"]
            else:
                clusters[i] = default_cluster
                dists[i] = default_dist

        return clusters, dists

    def save(self, file_path: Optional[Path] = None) -> Path:
        if file_path is None:
            file_path = ARTIFACTS_DIR / "kmeans_model.joblib"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)
        return file_path

    @classmethod
    def load(cls, file_path: Optional[Path] = None) -> "CustomerSegmentation":
        if file_path is None:
            file_path = ARTIFACTS_DIR / "kmeans_model.joblib"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing KMeans artifact: {file_path}")
        return joblib.load(file_path)


class PCAModel:
    """PCA for dimensionality reduction and 2D visualization."""

    def __init__(self, n_components: int = 5, random_state: int = RANDOM_SEED):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.explained_variance_ratio_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray) -> "PCAModel":
        self.pca.fit(X)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(X)

    def save(self, file_path: Optional[Path] = None) -> Path:
        if file_path is None:
            file_path = ARTIFACTS_DIR / "pca_model.joblib"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)
        return file_path

    @classmethod
    def load(cls, file_path: Optional[Path] = None) -> "PCAModel":
        if file_path is None:
            file_path = ARTIFACTS_DIR / "pca_model.joblib"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing PCA artifact: {file_path}")
        return joblib.load(file_path)


class SoftVotingEnsemble:
    """Soft voting ensemble combining XGBoost and CART probabilities."""

    def __init__(self, xgb_model: XGBClassifier, cart_model: DecisionTreeClassifier):
        self.xgb_model = xgb_model
        self.cart_model = cart_model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_xgb = self.xgb_model.predict_proba(X)[:, 1]
        p_cart = self.cart_model.predict_proba(X)[:, 1]
        p_ens = 0.5 * (p_xgb + p_cart)
        # Return 2D array (p0, p1)
        return np.vstack([1.0 - p_ens, p_ens]).T


def tune_decision_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Tune the decision threshold on validation set to maximize F1-score on the PR curve.
    Returns: (best_threshold, best_f1, best_precision, best_recall)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Avoid division by zero
    f1_scores = np.where(
        (precisions + recalls) > 0,
        2.0 * (precisions * recalls) / (precisions + recalls),
        0.0,
    )

    best_idx = int(np.argmax(f1_scores[:-1])) if len(thresholds) > 0 else 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.50
    best_f1 = float(f1_scores[best_idx]) if len(f1_scores) > 0 else 0.0
    best_prec = float(precisions[best_idx]) if len(precisions) > 0 else 0.0
    best_rec = float(recalls[best_idx]) if len(recalls) > 0 else 0.0

    # Ensure reasonable bounds
    best_threshold = float(np.clip(best_threshold, 0.05, 0.95))
    return best_threshold, best_f1, best_prec, best_rec


def evaluate_model_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Compute comprehensive test evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    # PR-AUC
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recalls, precisions))

    # ROC-AUC
    roc_auc = float(roc_auc_score(y_true, y_prob))

    # Precision, Recall, F1
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }
