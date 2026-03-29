"""
Angie McGraw
Last updated: March 28th, 2026
"""

import matplotlib.pyplot as plt
import os

FIGURE_DIR = "../figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

def save_2x2_double_descent(complexities_dict, train_errors_dict, test_errors_dict, filename="double_descent_2x2.png"):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    for i, model_name in enumerate(complexities_dict.keys()):
        axs[i].plot(complexities_dict[model_name], train_errors_dict[model_name], label="Train Error", marker='o')
        axs[i].plot(complexities_dict[model_name], test_errors_dict[model_name], label="Test Error", marker='x')
        axs[i].set_title(model_name)
        axs[i].set_xlabel("Model Complexity")
        axs[i].set_ylabel("MSE")
        axs[i].grid(True)
        axs[i].legend()

    plt.tight_layout()
    save_path = os.path.join(FIGURE_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 2x2 double descent figure to {save_path}")