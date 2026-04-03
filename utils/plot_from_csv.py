"""
Usage:
(Default: log) python3 -m utils.plot_from_csv --model nn --dataset friedman1
python3 -m utils.plot_from_csv --model nn --dataset friedman1 --linear
"""
import argparse
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def smooth(y, window=5):
    return np.convolve(y, np.ones(window)/window, mode='same')

def plot_from_csv(csv_path, model_name, log_scale=True):
    df = pd.read_csv(csv_path)

    train_col = "train_mse_mean" if "train_mse_mean" in df.columns else "train_mse"
    test_col = "test_mse_mean" if "test_mse_mean" in df.columns else "test_mse"
    params_col = "params_mean" if "params_mean" in df.columns else "params"

    x = df[params_col].values
    train_errors = df[train_col].values
    test_errors = df[test_col].values

    idx = np.argsort(x)
    x, train_errors, test_errors = x[idx], train_errors[idx], test_errors[idx]

    if log_scale:
        train_errors = np.maximum(train_errors, 1e-8)
        test_errors = np.maximum(test_errors, 1e-8)

    trim = 2
    train_smooth = smooth(train_errors, window=5)[trim:-trim]
    test_smooth = smooth(test_errors, window=5)[trim:-trim]
    x = x[trim:-trim]

    plt.figure(figsize=(8, 4))
    plt.plot(x, train_smooth, linewidth=2, label="Train Error")
    plt.plot(x, test_smooth, linewidth=2, label="Test Error")

    if log_scale:
        plt.yscale('log')
        plt.ylabel("MSE (log scale)")
    else:
        plt.ylabel("MSE")

    plt.title(f"Double Descent: {model_name}")
    plt.xlabel("Number of Parameters")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()

    scale_tag = "log" if log_scale else "linear"
    out_path = csv_path.replace("_metrics_summary.csv", f"_double_descent_{scale_tag}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved to {out_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["nn", "polynomial", "randomfeature", "kernelridge"])
    parser.add_argument("--dataset", required=True, choices=["friedman1", "friedman2", "friedman3"])
    parser.add_argument("--linear", action="store_true", help="Use linear y-axis instead of log.")
    args = parser.parse_args()

    csv_path = f"figures/{args.model}/{args.dataset}_metrics_summary.csv"
    plot_from_csv(csv_path, model_name=f"{args.model}_{args.dataset}", log_scale=not args.linear)