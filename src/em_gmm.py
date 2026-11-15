"""Hand-written Expectation-Maximization Gaussian Mixture Model in NumPy."""

from typing import List, Tuple, Optional
import numpy as np
from scipy.special import logsumexp
from scipy.linalg import solve_triangular


class HandWrittenGMM:
    """
    Gaussian Mixture Model fitted with Expectation-Maximization from scratch.
    Implements:
    - E-step: Posterior responsibilities gamma_ik and sample log-likelihoods
    - M-step: Parameter updates for weights, means, and covariance matrices
    - Anomaly scoring: Per-sample log-likelihood p(x)
    - Convergence tracking: Log-likelihood history per iteration
    """

    def __init__(
        self,
        n_components: int = 4,
        max_iter: int = 100,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        random_state: Optional[int] = 42,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

        self.weights_: Optional[np.ndarray] = None  # (K,)
        self.means_: Optional[np.ndarray] = None  # (K, D)
        self.covariances_: Optional[np.ndarray] = None  # (K, D, D)
        self.log_likelihood_history_: List[float] = []
        self.converged_: bool = False

    def _estimate_log_gaussian_prob(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log N(x_i | mu_k, Sigma_k) for all i and k.
        Returns: matrix of shape (N, K).
        """
        N, D = X.shape
        K = self.n_components
        log_prob = np.empty((N, K))

        for k in range(K):
            mu_k = self.means_[k]
            cov_k = self.covariances_[k] + self.reg_covar * np.eye(D)

            try:
                L = np.linalg.cholesky(cov_k)
            except np.linalg.LinAlgError:
                # Add diagonal jitter if not positive definite
                cov_k += (self.reg_covar * 100.0) * np.eye(D)
                L = np.linalg.cholesky(cov_k)

            # log determinant = 2 * sum(log(diag(L)))
            log_det = 2.0 * np.sum(np.log(np.diag(L)))

            diff = X - mu_k  # (N, D)
            # Solve L * y = diff.T -> y = L^-1 * diff.T
            y = solve_triangular(L, diff.T, lower=True)  # (D, N)
            quad_form = np.sum(y ** 2, axis=0)  # (N,)

            log_prob[:, k] = -0.5 * (D * np.log(2.0 * np.pi) + log_det + quad_form)

        return log_prob

    def _e_step(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Expectation Step:
        Compute responsibilities gamma_ik and total log-likelihood.
        Returns:
            log_resp: (N, K) log responsibilities
            resp: (N, K) responsibilities
            log_likelihood: float
        """
        # Weighted log-probabilities: ln(pi_k) + ln N(x | mu_k, Sigma_k)
        log_weighted = np.log(self.weights_ + 1e-15) + self._estimate_log_gaussian_prob(X)
        # Per-sample log-likelihood: logsumexp over K
        sample_log_prob = logsumexp(log_weighted, axis=1)  # (N,)
        total_log_likelihood = float(np.sum(sample_log_prob))

        # Posterior responsibilities: gamma_ik = exp(log_weighted - sample_log_prob)
        log_resp = log_weighted - sample_log_prob[:, np.newaxis]
        resp = np.exp(log_resp)

        return log_resp, resp, total_log_likelihood

    def _m_step(self, X: np.ndarray, resp: np.ndarray) -> None:
        """
        Maximization Step:
        Update mixing weights, means, and covariance matrices.
        """
        N, D = X.shape
        K = self.n_components

        # Effective number of points assigned to each component
        Nk = np.sum(resp, axis=0) + 1e-10  # (K,)

        # 1. Update weights
        self.weights_ = Nk / N

        # 2. Update means
        self.means_ = (resp.T @ X) / Nk[:, np.newaxis]

        # 3. Update covariances
        for k in range(K):
            diff = X - self.means_[k]  # (N, D)
            weighted_diff = resp[:, k : k + 1] * diff  # (N, D)
            cov_k = (weighted_diff.T @ diff) / Nk[k]
            # Regularize
            self.covariances_[k] = cov_k + self.reg_covar * np.eye(D)

    def fit(self, X: np.ndarray) -> "HandWrittenGMM":
        """Fit the GMM parameters using the EM algorithm."""
        X = np.asarray(X, dtype=float)
        N, D = X.shape
        rng = np.random.RandomState(self.random_state)

        # Initialize means via k-means or random choice
        init_indices = rng.choice(N, size=self.n_components, replace=False)
        self.means_ = X[init_indices].copy()
        self.weights_ = np.full(self.n_components, 1.0 / self.n_components)

        # Initialize covariances to empirical covariance of data
        emp_cov = np.cov(X, rowvar=False) + self.reg_covar * np.eye(D)
        self.covariances_ = np.tile(emp_cov, (self.n_components, 1, 1))

        self.log_likelihood_history_ = []
        prev_ll = -np.inf

        for it in range(self.max_iter):
            # E-step
            _, resp, current_ll = self._e_step(X)
            self.log_likelihood_history_.append(current_ll)

            # Check convergence
            if it > 0 and abs(current_ll - prev_ll) < self.tol:
                self.converged_ = True
                break

            prev_ll = current_ll

            # M-step
            self._m_step(X, resp)

        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the per-sample log-likelihood under the fitted GMM.
        Returns: (N,) vector where lower values denote unusual transactions.
        """
        X = np.asarray(X, dtype=float)
        log_weighted = np.log(self.weights_ + 1e-15) + self._estimate_log_gaussian_prob(X)
        return logsumexp(log_weighted, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute component posterior responsibilities."""
        _, resp, _ = self._e_step(X)
        return resp
