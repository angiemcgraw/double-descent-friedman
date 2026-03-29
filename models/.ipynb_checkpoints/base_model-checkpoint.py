"""
Angie McGraw
Last updated: March 28th, 2026
"""

class BaseModel:
    def __init__(self, complexity):
        self.complexity = complexity

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError