"""
Polynomial Regression Model with Optional Ridge Regularization
==============================================================

This class implements a polynomial regression model that:
- Expands input features into a higher-dimensional polynomial space
- Scales features for numerical stability
- Optionally applies ridge regression (L2 regularization)
- Allows control over model complexity by limiting number of features (p)

Used to study how increasing model capacity affects performance,
which is key to observing the double descent phenomenon.
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class PolynomialRegression:
    def __init__(self, degree=3, alpha=None, n_features=None):
        self.degree = degree
        self.alpha = alpha        
        self.n_features = n_features
        self.poly = PolynomialFeatures(degree=degree, include_bias=True)
        self.scaler = StandardScaler()
        self.coeffs = None
        self.p_used = None

    """Fits the model using polynomial features and optional ridge/OLS regression."""
    def fit(self, X, y):
        
        X_poly = self.poly.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_poly)

        p_full = X_scaled.shape[1]

        
        p = min(self.n_features, p_full) if self.n_features is not None else p_full
        self.p_used = p

        X_sub = X_scaled[:, :p]

        if self.alpha is not None:
            
            self.coeffs = np.linalg.solve(
                X_sub.T @ X_sub + self.alpha * np.eye(p),
                X_sub.T @ y
            )
        else:
            
            self.coeffs, _, _, _ = np.linalg.lstsq(X_sub, y, rcond=None)
            
      """Generates predictions using learned coefficients."""
    def predict(self, X):
        X_poly = self.poly.transform(X)
        X_scaled = self.scaler.transform(X_poly)
        return X_scaled[:, :self.p_used] @ self.coeffs
    
    """Returns number of learned parameters (model coefficients)."""
    def count_params(self):
        return len(self.coeffs) if self.coeffs is not None else np.nan
