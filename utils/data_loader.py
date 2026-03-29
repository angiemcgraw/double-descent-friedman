"""
Angie McGraw
Last updated: March 28th, 2026
"""

import numpy as np
import os

# Use the project root as base
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def load_dataset(name="friedman1"):
    path = os.path.join(DATA_DIR, f"{name}.npz")
    with np.load(path) as data:
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]
    return X_train, X_test, y_train, y_test