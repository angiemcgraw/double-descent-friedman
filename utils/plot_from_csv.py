"""
Reads from a pre-generated metrics_summary.csv (the run.py will produce this).
Produces the double descent plot. Options:
- Default: log-scale y-axis, log-scale x-axis
- Linear-scale y-axis
- Has conditional arguments for the NN, i.e. the Adam vs SGD

Usage (from project root):
(Default: log) python3 -m utils.plot_from_csv --model nn --dataset friedman1
python3 -m utils.plot_from_csv --model nn --dataset friedman1 --linear
python3 -m utils.plot_from_csv --model nn --dataset friedman1 --linear --condition adam_corruption0.15

Output: 
    {csv_dir}/{dataset}_double_descent_{log|linear}.png

Angie McGraw
Last updated: April 5th, 2026
"""
import argparse
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-white')

def smooth(y, window=11):
    """
    Simple moving average.
    "same" so that the output length matches the input length.
    """
    return np.convolve(y, np.ones(window)/window, mode='same')

def _load_and_prepare(csv_path, log_scale):
    """
    Loads a metrics_summary CSV and return smoothed train/test curves.

    Handles both summary CSVs (params_mean, train_mse_mean) and per-run
    CSVs (params, train_mse) by checking column names.

    What it does:
    - Sorting by parameter count.
    - Log floor (if log-scale) to avoid log(0) situation.
    - Moving average smoothing (window=11)
    - Edge trim (trim=5) to remove boundary artifacts from convolution.

    Returns:
        x: smoothed and trimmed parameter counts
        train_smooth: smoothed train MSE
        test_smooth: smoothed test MSE
        test_std_smooth: smoothed test MSE std (or None, if unavailable)
    """
    df = pd.read_csv(csv_path)

    train_col = "train_mse_mean" if "train_mse_mean" in df.columns else "train_mse"
    test_col = "test_mse_mean" if "test_mse_mean" in df.columns else "test_mse"
    params_col = "params_mean" if "params_mean" in df.columns else "params"
    std_col = "test_mse_std" if "test_mse_std" in df.columns else None

    x = df[params_col].values
    train_errors = df[train_col].values
    test_errors = df[test_col].values
    test_std = df[std_col].values if std_col else None

    idx = np.argsort(x)
    x = x[idx]
    train_errors = train_errors[idx]
    test_errors = test_errors[idx]
    if test_std is not None:
        test_std = test_std[idx]

    if log_scale:
        train_errors = np.maximum(train_errors, 1e-8)
        test_errors = np.maximum(test_errors, 1e-8)

    trim = 5 if len(x) > 20 else 0
    window = 11 if len(x) > 20 else 1     # window=1 means no smoothing
    train_smooth = smooth(train_errors, window=window)
    test_smooth = smooth(test_errors, window=window)
    if trim > 0:
        train_smooth = train_smooth[trim:-trim]
        test_smooth = test_smooth[trim:-trim]
        x = x[trim:-trim]
        test_std_smooth = smooth(test_std, window=11)[trim:-trim] if test_std is not None else None
    else:
        test_std_smooth = smooth(test_std, window=11) if test_std is not None else None

    return x, train_smooth, test_smooth, test_std_smooth

def plot_from_csv(csv_path, model_name, log_scale=True):
    """
    Output is a double descent plot from a metrics_summary.csv.
    Saves to the same direcotry as the .csv.
    """
    x, train_smooth, test_smooth, _ = _load_and_prepare(csv_path, log_scale)
    
    plt.figure(figsize=(8, 4))
    plt.plot(x, test_smooth, linewidth=2.0, linestyle='-', color='#3A5FA0', label="Test error")
    plt.plot(x, train_smooth, linewidth=1.8, linestyle='-', color='#f0051b', label="Train error")

    if log_scale:
        plt.yscale('log')
        plt.ylabel("MSE (log scale)")
    else:
        plt.ylabel("MSE")

    plt.title(f"Double descent: {model_name}")
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xscale('log')
    plt.xlabel("Number of parameters (log scale)")
    plt.grid(False)
    plt.legend(loc='upper right', fontsize=7, frameon=False)
    plt.tight_layout()

    scale_tag = "log" if log_scale else "linear"
    out_path = csv_path.replace("_metrics_summary.csv", 
                                f"_double_descent_{scale_tag}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved to {out_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["nn", "polynomial", "randomfeature", "kernelridge"])
    parser.add_argument("--dataset", required=True, choices=["friedman1", "friedman2", "friedman3"])
    parser.add_argument("--linear", action="store_true", help="Use linear y-axis instead of log.")
    # NN specific
    parser.add_argument("--condition", type=str, default=None,
                        help="Subdirectory condition e.g. adam_corruption0.15")
    args = parser.parse_args()

    if args.condition:
        csv_path = f"figures/{args.model}/{args.condition}/{args.dataset}_metrics_summary.csv"
    else:
        csv_path = f"figures/{args.model}/{args.dataset}_metrics_summary.csv"
        
    model_name = f"{args.model}_{args.dataset}"

    plot_from_csv(csv_path, 
                  model_name=model_name, 
                  log_scale=not args.linear)