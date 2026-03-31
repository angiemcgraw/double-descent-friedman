"""
Shared plotting. 

Angie McGraw
Last updated: March 31st, 2026
"""

import matplotlib.pyplot as plt
import os

FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
#FIGURE_DIR = "../figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

def plot_double_descent(complexities, train_errors, test_errors, model_name="Model", filename=None):
    """
    Plots train/test errors vs. model complexity.

    MSE (should either be normalized MSE, log-scale MSE, or relative error (e.g. divided
    by variance of y).
    """
    plt.figure(figsize=(6,4))
    plt.plot(complexities, train_errors, marker='o', label="Train Error", alpha=0.7)
    plt.plot(complexities, test_errors, marker='x', label="Test Error", alpha=0.7)
    plt.title(f"Double Descent: {model_name}")
    plt.xlabel("Model Complexity")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()
    if filename is None:
        filename = f"{model_name}_double_descent.png"
    plt.tight_layout()
    save_path = os.path.join(FIGURE_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Figure saved to {save_path}.")