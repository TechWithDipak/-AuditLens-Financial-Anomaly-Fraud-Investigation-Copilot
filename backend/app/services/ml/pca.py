"""Unit 3C: Principal Component Analysis (PCA) Implementation."""

from typing import List

import numpy as np
from sklearn.decomposition import PCA

from app.schemas.algorithms import PCARunResponse


class PCAService:
    @staticmethod
    def run_pca(
        X: np.ndarray,
        feature_names: List[str],
        n_components: int = 2,
        random_state: int = 42,
    ) -> PCARunResponse:
        """Fit PCA on feature matrix, calculate explained variance and 2D projection."""
        n_samples, n_features = X.shape
        max_possible = min(n_samples, n_features)

        if n_components < 1 or n_components > max_possible:
            raise ValueError(
                f"n_components ({n_components}) must be between 1 and min(samples={n_samples}, features={n_features})."
            )
        if not np.isfinite(X).all():
            raise ValueError("Input array contains NaN or non-finite values.")

        pca = PCA(n_components=n_components, random_state=random_state)
        X_projected = pca.fit_transform(X)

        exp_var = [round(float(v), 4) for v in pca.explained_variance_ratio_]
        cum_var = [round(float(v), 4) for v in np.cumsum(pca.explained_variance_ratio_)]
        total_var = round(float(np.sum(pca.explained_variance_ratio_)), 4)

        proj_preview: List[List[float]] = [
            [round(float(coord), 4) for coord in row] for row in X_projected[:20]
        ]

        return PCARunResponse(
            n_components=n_components,
            explained_variance_ratio=exp_var,
            cumulative_variance_ratio=cum_var,
            total_explained_variance=total_var,
            projection_preview=proj_preview,
            feature_names=feature_names,
        )
