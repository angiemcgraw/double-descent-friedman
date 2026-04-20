"""
Kernel Ridge Regression for double descent experiments.

Complexity controls the regularization parameter alpha on a log scale:
    alpha = 10 ** (-complexity / 10)

As complexity increases, alpha shrinks toward zero. The interpolation
threshold occurs when alpha is small enough that the regularized kernel
matrix (K + alpha * I) becomes nearly singular — the effective number of
parameters approaches n_samples and test error spikes.

Kernel: RBF (Gaussian) with bandwidth set via median heuristic, which
adapts to the scale of the data without manual tuning. This gives an
infinite-dimensional feature space so the only knob is alpha.

Uses sklearn's KernelRidge, which solves the dual form:
    alpha_vec = (K + alpha * I)^{-1} y

Numerical stability:
    At very small alpha the kernel matrix becomes nearly singular,
    producing NaN/inf predictions (especially on Friedman2/3 which
    have larger output scales). We clamp predictions and report the
    effective degrees of freedom: trace(K (K + alpha I)^{-1}), which
    is the KRR analog of "number of parameters".

Last updated: April 2026
"""

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from models.base_model import BaseModel


class KernelRidgeRegression(BaseModel):
    def __init__(self, alpha=1.0, kernel="rbf", gamma=None):
        """
        Parameters
        ----------
        alpha : float
            Regularization strength. Smaller values -> less regularization ->
            closer to interpolation.
        kernel : str
            Kernel type. Default "rbf" (Gaussian).
        gamma : float or None
            RBF bandwidth parameter. If None, set automatically via the
            median heuristic during fit().
        """
        super().__init__(complexity=alpha)

        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.model_ = None
        self.effective_dof_ = np.nan   # effective degrees of freedom

    def _median_heuristic(self, X):
        """
        Set gamma = 1 / (2 * median(||x_i - x_j||^2)).

        The median heuristic (Garreau et al., 2017) is a standard,
        data-adaptive way to choose the RBF bandwidth without
        cross-validation. It keeps the kernel matrix well-conditioned
        across different datasets and scales.
        """
        dists = euclidean_distances(X, squared=True)
        triu_idx = np.triu_indices_from(dists, k=1)
        median_dist = np.median(dists[triu_idx])
        if median_dist == 0:
            median_dist = 1.0
        return 1.0 / (2.0 * median_dist)

    def _compute_effective_dof(self, X, gamma):
        """
        Effective degrees of freedom for KRR:
            df(alpha) = trace( K (K + alpha I)^{-1} )
                      = sum_i  lambda_i / (lambda_i + alpha)

        where lambda_i are the eigenvalues of K. This is the KRR
        analog of "number of parameters": when alpha is large, df ~ 0
        (underfitting); when alpha -> 0, df -> n (interpolation).
        """
        K = rbf_kernel(X, gamma=gamma)
        eigvals = np.linalg.eigvalsh(K)          # real, sorted ascending
        eigvals = np.maximum(eigvals, 0.0)        # clamp numerical negatives
        dof = np.sum(eigvals / (eigvals + self.alpha))
        return dof

    def fit(self, X, y):
        # Compute gamma from data if not provided
        if self.gamma is None:
            gamma = self._median_heuristic(X)
        else:
            gamma = self.gamma
        self.gamma_ = gamma    # store for later use

        self.model_ = KernelRidge(
            alpha=self.alpha,
            kernel=self.kernel,
            gamma=gamma
        )
        self.model_.fit(X, y)

        # Effective degrees of freedom (KRR's "param count")
        self.effective_dof_ = self._compute_effective_dof(X, gamma)

        # ── Stability check ──────────────────────────────────────
        y_pred = self.model_.predict(X)
        if not np.all(np.isfinite(y_pred)):
            n_bad = np.sum(~np.isfinite(y_pred))
            print(f"  ** WARNING: {n_bad}/{len(y_pred)} non-finite train "
                  f"predictions at alpha={self.alpha:.2e}")
            # Clamp so downstream MSE computation doesn't produce NaN
            y_pred = np.nan_to_num(y_pred, nan=0.0,
                                   posinf=np.finfo(np.float64).max / 2,
                                   neginf=-np.finfo(np.float64).max / 2)

        train_mse = np.mean((y - y_pred) ** 2)
        print(
            f"     alpha={self.alpha:.2e} | gamma={gamma:.4f} "
            f"| eff_dof={self.effective_dof_:.1f}/{X.shape[0]} "
            f"| train_loss={train_mse:.6f}"
        )

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        preds = self.model_.predict(X)

        # Clamp non-finite predictions so MSE stays real
        if not np.all(np.isfinite(preds)):
            preds = np.nan_to_num(preds, nan=0.0,
                                  posinf=np.finfo(np.float64).max / 2,
                                  neginf=-np.finfo(np.float64).max / 2)
        return preds

    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Compute and print train/test MSE. Call after fit().
        """
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        train_pred = self.predict(X_train)
        test_pred  = self.predict(X_test)

        train_mse = np.mean((y_train - train_pred) ** 2)
        test_mse  = np.mean((y_test  - test_pred)  ** 2)

        print(f"     alpha={self.alpha:.2e} | eff_dof={self.effective_dof_:.1f} "
              f"| train_mse={train_mse:.6f} | test_mse={test_mse:.6f}")

        return {"train_mse": train_mse, "test_mse": test_mse}
