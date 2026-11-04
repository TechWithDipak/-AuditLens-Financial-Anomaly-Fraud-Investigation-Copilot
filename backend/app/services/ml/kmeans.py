"""Unit 3A: K-Means Clustering Implementation."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from app.schemas.algorithms import ClusterSummary, KMeansRunResponse


class KMeansService:
    @staticmethod
    def run_kmeans(
        X: np.ndarray,
        df_original: pd.DataFrame,
        k: int = 3,
        compute_elbow: bool = True,
        random_state: int = 42,
    ) -> KMeansRunResponse:
        """Fit K-Means, compute cluster summaries, and evaluate inertia/silhouette."""
        n_samples, n_features = X.shape
        if n_samples < k:
            raise ValueError(
                f"Sample count ({n_samples}) must be greater than or equal to k ({k})."
            )
        if not np.isfinite(X).all():
            raise ValueError("Input array contains NaN or infinite values.")

        model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = model.fit_predict(X)

        # Inertia and Silhouette score
        inertia = float(model.inertia_)
        sil_score: Optional[float] = None
        if n_samples >= 2 and len(np.unique(labels)) > 1:
            # Subsample for silhouette if dataset is large to maintain rapid responsiveness
            sil_sample_size = min(1000, n_samples)
            if sil_sample_size == n_samples:
                sil_score = round(float(silhouette_score(X, labels)), 4)
            else:
                sil_score = round(
                    float(
                        silhouette_score(
                            X, labels, sample_size=sil_sample_size, random_state=random_state
                        )
                    ),
                    4,
                )

        # Elbow curve analysis (k=2 to min(8, n_samples-1))
        elbow_curve: Optional[List[Dict[str, float]]] = None
        if compute_elbow:
            elbow_curve = []
            max_elbow_k = min(8, n_samples - 1)
            for test_k in range(2, max_elbow_k + 1):
                km_test = KMeans(n_clusters=test_k, random_state=random_state, n_init="auto")
                km_test.fit(X)
                elbow_curve.append(
                    {"k": float(test_k), "inertia": round(float(km_test.inertia_), 2)}
                )

        # Summaries in original feature units
        feature_cols = [
            c
            for c in df_original.columns
            if c != "Class" and pd.api.types.is_numeric_dtype(df_original[c])
        ]
        summaries: List[ClusterSummary] = []

        for cid in range(k):
            mask = labels == cid
            cluster_count = int(np.sum(mask))
            prop_pct = round((cluster_count / n_samples) * 100.0, 2)

            means: Dict[str, float] = {}
            if cluster_count > 0:
                cluster_df = df_original.loc[mask, feature_cols]
                for col in feature_cols[:6]:  # Include prominent features (Time, Amount, V1-V4)
                    means[col] = round(float(cluster_df[col].mean()), 4)

            summaries.append(
                ClusterSummary(
                    cluster_id=cid,
                    sample_count=cluster_count,
                    proportion_pct=prop_pct,
                    feature_means=means,
                )
            )

        return KMeansRunResponse(
            k=k,
            inertia=round(inertia, 2),
            silhouette_score=sil_score,
            elbow_curve=elbow_curve,
            cluster_summaries=summaries,
            cluster_assignments_preview=[int(x) for x in labels[:20]],
        )
