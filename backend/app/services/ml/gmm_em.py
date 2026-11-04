"""Unit 3B: Gaussian Mixture Model (GMM) fitted via Expectation-Maximization (EM)."""

from typing import List

import numpy as np
from sklearn.mixture import GaussianMixture

from app.schemas.algorithms import GMMRunResponse


class GMMService:
    @staticmethod
    def run_gmm(
        X: np.ndarray,
        n_components: int = 3,
        covariance_type: str = "full",
        unusual_percentile: float = 1.0,
        random_state: int = 42,
    ) -> GMMRunResponse:
        """Fit Gaussian Mixture Model via EM, compute responsibilities, log-likelihoods, and unusualness."""
        n_samples, n_features = X.shape
        if n_samples < n_components:
            raise ValueError(
                f"Sample count ({n_samples}) must be greater than component count ({n_components})."
            )
        if not np.isfinite(X).all():
            raise ValueError("Input array contains NaN or non-finite values.")

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
        )
        gmm.fit(X)

        # E-step: compute posterior responsibilities gamma_ik
        responsibilities = gmm.predict_proba(X)

        # Per-sample log-likelihood: log p(x_i)
        log_likelihoods = gmm.score_samples(X)

        # Unusualness ranking: lowest log-likelihoods under fitted distribution
        threshold = float(np.percentile(log_likelihoods, unusual_percentile))
        unusual_mask = log_likelihoods <= threshold
        unusual_indices = np.where(unusual_mask)[0].tolist()

        resp_preview: List[List[float]] = [
            [round(float(p), 4) for p in row] for row in responsibilities[:10]
        ]
        ll_preview: List[float] = [round(float(val), 4) for val in log_likelihoods[:10]]

        return GMMRunResponse(
            n_components=n_components,
            covariance_type=covariance_type,
            converged=bool(gmm.converged_),
            n_iter=int(gmm.n_iter_),
            aic=round(float(gmm.aic(X)), 2),
            bic=round(float(gmm.bic(X)), 2),
            unusual_threshold_log_likelihood=round(threshold, 4),
            unusual_sample_count=len(unusual_indices),
            responsibilities_preview=resp_preview,
            log_likelihoods_preview=ll_preview,
            unusual_indices_preview=unusual_indices[:20],
        )
