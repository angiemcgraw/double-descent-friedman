"""
Reads from a pre-generated metrics_summary.csv (the run.py will produce this).
Produces the double descent plot. Options:
- Default: log-scale y-axis
- Linear-scale y-axis
- Linear-scale y-axis, Annotated
- Has conditional arguments for the NN, i.e. the Adam vs SGD

Usage (from project root):
(Default: log) python3 -m utils.plot_from_csv --model nn --dataset friedman1
python3 -m utils.plot_from_csv --model nn --dataset friedman1 --linear
python3 -m utils.plot_from_csv --model nn --dataset friedman1 --linear --annotated
python3 -m utils.plot_from_csv --model nn --dataset friedman1 --linear --condition adam_corruption0.15

Output: 
    {csv_dir}/{dataset}_double_descent_{log|linear}.png
    {csv_dir}/{dataset}_double_descent_{log|linear}_annotated.png (if --annotated).

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

def find_threshold(x, train_errors, eps=0.01):
    """
    Estimate the interpolation threshold: first param count where train error
    drops below eps.

    eps is epsilon - used as a floor to prevent numerical issues. Small 
    positive number. Prevention of log(0) situation.
    """
    below = np.where(train_errors < eps)[0]
    if len(below) == 0:
        return None
    return x[below[0]]

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

    trim = 5
    train_smooth = smooth(train_errors, window=11)[trim:-trim]
    test_smooth = smooth(test_errors, window=11)[trim:-trim]
    x = x[trim:-trim]
    test_std_smooth = smooth(test_std, window=11)[trim:-trim] if test_std is not None else None

    return x, train_smooth, test_smooth, test_std_smooth

def plot_from_csv(csv_path, model_name, log_scale=True):
    """
    Output is a double descent plot from a metrics_summary.csv.
    Saves to the same direcotry as the .csv.
    """
    x, train_smooth, test_smooth, _ = _load_and_prepare(csv_path, log_scale)
    
    plt.figure(figsize=(8, 4))
    plt.plot(x, test_smooth, linewidth=2.0, linestyle='-', color='#3A5FA0', label="Test error")
    plt.plot(x, train_smooth, linewidth=1.8, linestyle='--', color='#f0051b', label="Train error")

    if log_scale:
        plt.yscale('log')
        plt.ylabel("MSE (log scale)")
    else:
        plt.ylabel("MSE")

    plt.title(f"Double descent: {model_name}")
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel("Number of parameters")
    plt.grid(False)
    plt.legend(loc='upper right', fontsize=7, frameon=False)
    plt.tight_layout()

    scale_tag = "log" if log_scale else "linear"
    out_path = csv_path.replace("_metrics_summary.csv", 
                                f"_double_descent_{scale_tag}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved to {out_path}.")

def plot_from_csv_annotated(csv_path, model_name, log_scale=True):
    """
    Outputs an annotated double descent plot from a metrics_summary.csv.

    Annotations (inspired by Nakkiran et al. 2019):
    - Critical regime shading (orange) between interpolation threshold and peak.
    - Dashed interpolation threshold line with an arrow pointing to it.
    - Classical regime, modern regime text labels.
    - Shaded +/-1 std band around the test error (if the std is available).

    find_threshold() to detect the interpolation threshold - uses train error.

    Annotations are drawn in situations where train error reaches near-zero.

    Saves to the same directory as the CSV.
    """
    x, train_smooth, test_smooth, test_std_smooth = _load_and_prepare(csv_path, log_scale)
    threshold = find_threshold(x, train_smooth)

    fig, ax = plt.subplots(figsize=(9, 5))
    ymax = test_smooth.max() * 1.25

    if threshold is not None:
        peak_idx = np.argmax(test_smooth)
        peak_x = x[peak_idx]
        min_width = (x[-1] - x[0]) * 0.05     # at least 5% of x range
        region_lo = threshold
        region_hi = max(peak_x, threshold + (x[-1] - x[0]) * 0.02)
        ax.axvspan(region_lo, region_hi, alpha=0.15, color='#E8925A', zorder=1)
        ax.axvline(threshold, color='#8B4513', linestyle='--',
                   linewidth=1.2, zorder=2)
        print(f"threshold: {threshold}, peak_x: {peak_x}, region_lo: {region_lo}, region_hi: {region_hi}")

    if test_std_smooth is not None:
        lower = np.maximum(test_smooth - test_std_smooth, 1e-8) if log_scale else test_smooth - test_std_smooth
        ax.fill_between(x, lower, test_smooth + test_std_smooth,
                        alpha=0.07, color='#3A5FA0', zorder=0)

    ax.plot(x, test_smooth, linewidth=2.0, linestyle='-',
            color='#3A5FA0', label='Test error', zorder=2)
    ax.plot(x, train_smooth, linewidth=1.8, linestyle='--',
            color='#f0051b', label='Train error', zorder=2)

    if threshold is not None:
        peak_idx = np.argmax(test_smooth)
        peak_x = x[peak_idx]
        region_hi = max(threshold, peak_x)

        classical_x = x[0] + (threshold - x[0]) * 0.30
        ax.text(classical_x, ymax * 0.97,
                'Classical regime:\nBias-variance tradeoff',
                ha='center', va='top', fontsize=8.5, color='#333333')

        modern_x = region_hi + (x[-1] - region_hi) * 0.45
        ax.text(modern_x, ymax * 0.97,
                'Modern regime',
                ha='center', va='top', fontsize=8.5, color='#333333')

        ax.annotate('Interpolation\nthreshold',
                    xy=(threshold, test_smooth[np.argmin(np.abs(x - threshold))] * 0.5),
                    xytext=(threshold + (x[-1] - x[0]) * 0.08,
                            test_smooth.max() * 0.45),
                    arrowprops=dict(arrowstyle='->', color='#8B4513', lw=1.2),
                    fontsize=8.5, color='#8B4513', ha='left')
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('MSE (log scale)', fontsize=11)
    else:
        ax.set_ylabel('MSE', fontsize=11)

    ax.set_ylim(top=ymax)
    ax.set_xlabel('Number of parameters', fontsize=11)
    ax.set_title(f'Double descent: {model_name}', fontsize=12, pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()

    scale_tag = "log" if log_scale else "linear"
    out_path = csv_path.replace("_metrics_summary.csv",
                                f"_double_descent_{scale_tag}_annotated.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved to {out_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["nn", "polynomial", "randomfeature", "kernelridge"])
    parser.add_argument("--dataset", required=True, choices=["friedman1", "friedman2", "friedman3"])
    parser.add_argument("--linear", action="store_true", help="Use linear y-axis instead of log.")
    parser.add_argument("--annotated", action="store_true", help="Produce annotated version,")
    # NN specific
    parser.add_argument("--condition", type=str, default=None,
                        help="Subdirectory condition e.g. adam_corruption0.15")
    args = parser.parse_args()

    if args.condition:
        csv_path = f"figures/{args.model}/{args.condition}/{args.dataset}_metrics_summary.csv"
    else:
        csv_path = f"figures/{args.model}/{args.dataset}_metrics_summary.csv"
        
    model_name = f"{args.model}_{args.dataset}"

    if args.annotated:
        plot_from_csv_annotated(csv_path, model_name=model_name,
                                log_scale=not args.linear)
    else:
        plot_from_csv(csv_path, model_name=f"{args.model}_{args.dataset}", 
                      log_scale=not args.linear)