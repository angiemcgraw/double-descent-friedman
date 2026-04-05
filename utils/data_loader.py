"""
Loads data using the pre-generated Friedman datasets (normalized train/test 
splits from .npz files in /data/.

Angie McGraw
Last updated: March 28th, 2026
"""

import numpy as np
import os

# Use the project root as base
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def load_dataset(name="friedman1"):
    """
    Load a pre-generated Friedman dataset.

    Args:
        name: str
            options: "friedman1", "friedman2", "friedman3"

    Returns:
        X_train, X_test, y_train, y_test: normalized numpy arrays
    """
    path = os.path.join(DATA_DIR, f"{name}.npz")
    with np.load(path) as data:
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]
    return X_train, X_test, y_train, y_test