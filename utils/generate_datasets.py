"""
Same as the generate_datasets.ipynb.
To generate the .npz files, generate_datasets.ipynb, not generate_datasets.py.

Angie McGraw
Last Updated: March 28th, 2026
"""

import os
import numpy as np
from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

N_TRAIN = 100
N_TEST = 1000
NOISE = 1.0
seed_map = {"friedman1":42, "friedman2":52, "friedman3":62}

def generate(save_npz=True):
    n_total = N_TRAIN + N_TEST
    datasets = {}
    for name, func in [("friedman1", make_friedman1), ("friedman2", make_friedman2), ("friedman3", make_friedman3)]:
        base_seed = seed_map[name]
        data_seed = base_seed + 100
        if name == "friedman1":
            X, y = func(n_samples=n_total, n_features=10, noise=NOISE, random_state=data_seed)
        else:
            X, y = func(n_samples=n_total, noise=NOISE, random_state=data_seed)
        split_seed = base_seed + 200
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=N_TRAIN, test_size=N_TEST, random_state=split_seed)
        datasets[name] = (X_train, X_test, y_train, y_test)
        if save_npz:
            np.savez(os.path.join(DATA_DIR, f"{name}.npz"), X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    return datasets