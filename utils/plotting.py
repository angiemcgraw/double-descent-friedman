"""
Shared plotting. 

Produces log-scale plots after each run. 

plot_from_csv.py for customizable plotting.

Angie McGraw
Last updated: March 31st, 2026
"""

import matplotlib.pyplot as plt
import os
import numpy as np

FIGURE_DIR = os.path.dirname(os.path.dirname(__file__))
#FIGURE_DIR = "../figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

def smooth(y, window=5):
    """
    Simple moving average.
    "same" is used so that the output length is the same as the input length.
    """
    return np.convolve(y, np.ones(window)/window, mode='same')

def plot_double_descent(complexities, train_errors, test_errors, 
                        param_counts=None, model_name="Model", filename=None, threshold=None):
    """
    Plots train/test errors vs. model complexity.

    Works for model types:
    1. Neural network: pass param_counts to use as x-axis
    2. Polynomial regression: complexities = polynomial degree
    3. Random features: complexities = number of random features
    4. Kernel ridge: complexities = 1/lambda (inverse regularization strength)

    MSE (should either be normalized MSE, log-scale MSE, or relative error (e.g. divided
    by variance of y).

    Args:
        complexities: list of complexity values (always required, used as x-axis if no param_counts)
        train_errors: list of train MSEs
        test_errors: list of test MSEs
        param_counts: optional list of parameter counts (NN only) - used as x-axis
        model_name: string label for title and filename
        filename: override output filename
        threshold: optional x-value to draw a vertical interpolation threshold line
            NN: pass n_train; for polynomial: pass n_train; for random features: pass n_train; for kernel ridge: None
    """
    plt.figure(figsize=(8,4))

    x = param_counts if param_counts is not None else complexities
    xlabel = "Number of Parameters (log scale)" if param_counts is not None else "Model Complexity"

    x = np.array(x)
    train_errors = np.array(train_errors)
    test_errors = np.array(test_errors)

    # Sort by x
    idx = np.argsort(x)
    x = x[idx]
    train_errors = train_errors[idx]
    test_errors = test_errors[idx]

    # Clip outliers (but preserve peak)
    clip_val = np.percentile(test_errors, 95)
    train_errors = np.clip(train_errors, None, clip_val)
    test_errors = np.clip(test_errors, None, clip_val)

    # Avoid log(0) (undefined)
    eps = 1e-8
    train_errors = np.maximum(train_errors, eps)
    test_errors = np.maximum(test_errors, eps)

    # Smooth curves (we have stochastic training and noise, so smoothing is helpful)
    train_errors_smooth = smooth(train_errors, window=5)
    test_errors_smooth = smooth(test_errors, window=5)
    
    plt.plot(x, train_errors_smooth, linewidth=2, label="Train Error")
    plt.plot(x, test_errors_smooth, linewidth=2, label="Test Error")

    if threshold is not None:
        plt.axvline(x=threshold, color='gray', linestyle='--',
                    alpha=0.6, label=f"Interpolation threshold ({threshold})")

    plt.xscale('log')
    #plt.yscale('log')
    plt.title(f"Double Descent: {model_name}")
    plt.xlabel(xlabel)
    plt.ylabel("MSE")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if filename is None:
        filename = f"{model_name}_double_descent.png"
        
    save_path = os.path.join(FIGURE_DIR, filename)
    print(save_path)
    try:
        os.makedirs(FIGURE_DIR, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Successfully saved to: {save_path}")
    except Exception as e:
        print(f"Original save failed: {e}")
    #plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"[INFO] Figure saved to {save_path}.")