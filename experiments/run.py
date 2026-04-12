"""
Main entry point for all models and datasets.
- Run multiple seeds per complexity.
- Averages results across runs, since double descent is a statistical phenomenon.
- Produces smooth double descent curves.

Operations: 
1. Sweep over complexity (e.g. hidden units for neural networks) values (this is
the looping variable).
2. For each complexity, we train the model.
3. After traiing, we call count_params to get the actual parameter count.
4. The parameter count is stored in params_mean in the CSV.
5. plot_from_csv reads params_mean and uses it as the x-axis

Usage (from the base directory: /double-descent-friedman)
python3 -m experiments.run --model nn --dataset friedman1
python3 -m experiments.run --model nn --dataset friedman1 --optimizer adam --corruption 0.15
python3 -m experiments.run --model polynomial --dataset friedman1
python3 -m experiments.run --model nn --dataset friedman1 --output-dir adam_corruption0.15

Models supported:
1. Polynomial Regression (polynomial)
2. Random Feature Regression (randomfeature)
3. Kernel Ridge Regresion (kernelridge)
4. Neural Network (nn)

NN-specific CLI arguments:
--optimizer: "adam" (default) or "sgd"
--corruption: label corruption rate (default 0.15, set 0.0 for no corruption)
These arguments are ignored for the other models in this study.

Output CSVs:
{dataset}_metrics_summary.csv: mean/std of train/test MSE and param counts
{dataset}_metrics_per_run.csv: long-format per-run results

Output directory structure:
For non-NN models,
figures/{model}/
For NN models,
figures/nn/{optimizer}_corruption{rate}/ 

Angie McGraw
Last updated: April 5th, 2026
"""

import numpy as np
import argparse
import logging
import pandas as pd
import os
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import torch
from joblib import Parallel, delayed
from utils.data_loader import load_dataset
from utils.plotting import plot_double_descent

# Import models
#from models.polynomial import PolynomialRegression
#from models.randomfeature import RandomFeatureRegression
#from models.kernelridge import KernelRidgeRegression
from models.neural_network import NeuralNetwork

logging.basicConfig(level=logging.INFO)
torch.set_num_threads(1)

# Set number of runs per complexity
N_RUNS = 20     # each complexity is an average of N_RUNS independent trainings

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_model(model_name, complexity, input_dim, n_samples, optimizer="adam"):
    """
    Initialize a model for a given complexity level.

    For the NN, complexity = number of hidden units in the single hidden layer.
    The interpolation threshold occurs at approximately: critical_hidden = n_samples // (input_dim + 2)
    which corresponds to params ~ n_samples.

    Args:
        model_name: str
            options: "nn", "polynomial", "randomfeature", "kernelridge"
        complexity: int
            model complexity
        input_dim: int
            number of input features
        n_samples: int
            number of training samples
        optimizer: str
            options (for nn only): "adam" or "sgd"
    """
    if model_name == "nn":
        lr = 1e-3 if optimizer == "adam" else 0.01
        return NeuralNetwork(
            input_dim=input_dim,
            complexity=complexity,
            epochs=10000,
            #epochs=3000,     # for initial runs
            lr=lr,
            optimizer=optimizer
        )
    #elif model_name == "polynomial:
        #return PolynomialRegression(degree=complexity)
    #elif model_name == "randomfeature":
        #return RandomFeatureRegresssion(n_features=complexity)
    #elif model_name == "kernelridge":
        #alpha = 10 ** (-complexity / 10)
        #reutrn KernelRidgeRegression(alpha=alpha)
    else:
        raise ValueError(f"Unknown model: {model_name}.")

"""
The double descent peak happens near the point where the number of parameters = the number
of training samples. We want high resolution here to be able to capture that.
"""
def get_complexities(model_name, input_dim, n_samples):
    """
    Returns the complexity grid for a given model and dataset.

    For the NN, complexity = number of hidden units in the single hidden layer. 
    The grid is constructed with fine resolution around the interpolation 
    threshold (critical_hidden), where the double descent peak is expected.

        Parameter count for a single hidden layer MLP: 
            total_params = hidden * (input_dim + 2) + 1
            where: 
                input_dim * hidden = input-to-hidden weights
                hidden = hidden biases
                hidden * 1 = hidden-to-output weights
                1 = output bias

        Interpolation threshold occurs when total_params ~ n_samples.
        Solving for hidden gives the critcal point: 
            critical_hidden = n_samples // (input_dim + 2)

        e.g. Friedman1: critical_hidden = 500 // 12 = 41, total_params ~ 505
             Friedman2: critical_hidden = 500 // 6 = 83, total_params ~ 499
    """
    if model_name == "nn":
        critical_hidden = max(1, n_samples // (input_dim + 2))
        small = list(range(1, critical_hidden + 20))
        medium = list(range(critical_hidden, critical_hidden * 2, 1))
        large = list(range(critical_hidden * 2, 400, 10))    

        return sorted(set(small + medium + large))
        
    #elif model_name == "polynomial:
        #return list(range(1, 30))
    #elif model_name == "randomfeature":
    #elif model_name == "kernelridge":
    else:
        raise ValueError(f"Unknown model: {model_name}.")

def count_params(model):
    """
    Count trainable parameters. 
    """
    if hasattr(model, "model"):
        return sum(p.numel() for p in model.model.parameters())
    else: 
        return np.nan

def extract_hyperparameters(model_name, complexity, model):
    """
    Extract hyperparameters from a fitted model for logging to CSV.
    If the field is not applicable to the given model, the field is 
    set as np.nan.
    """
    if model_name == "nn":
        return {
            "learning_rate": getattr(model, "lr", np.nan),
            "batch_size": getattr(model, "batch_size", np.nan),
            "epochs": getattr(model, "epochs", np.nan),
            "alpha": np.nan,
            "degree": np.nan,
            "n_features": np.nan
        }
    elif model_name == "polynomial":
        return {
            "learning_rate": np.nan,
            "batch_size": np.nan,
            "epochs": np.nan,
            "alpha": np.nan,
            "degree": np.nan,
            "n_features": np.nan
        }
    elif model_name == "randomfeature":
        return {
            "learning_rate": np.nan,
            "batch_size": np.nan,
            "epochs": np.nan,
            "alpha": np.nan,
            "degree": np.nan,
            "n_features": np.nan
        }
    elif model_name == "kernelridge":
        return {
            "learning_rate": np.nan,
            "batch_size": np.nan,
            "epochs": np.nan,
            "alpha": np.nan,
            "degree": np.nan,
            "n_features": np.nan
        }
    else:
        return {
            "learning_rate": np.nan,
            "batch_size": np.nan,
            "epochs": np.nan,
            "alpha": np.nan,
            "degree": np.nan,
            "n_features": np.nan
        }

def run_seed(seed, complexity, X_train, y_train, X_test, y_test, model_name,
            optimizer="adam", corruption=0.15):
    """
    Train one model at a given complexity and seed, return train/test MSE.

    Label corruption (Nakkiran et al., 2019), applicable to NN implementation
        Add label noise to training set only (Nakkiran et al.) - test 
        labels remain clean throughout. 
        
        What it does: Randomly replace 15% of training labels with values 
        drawn from the training label distribution. 
        
        This causes the interpolation threshold peak without raising the noise 
        floor as much as Gaussian noise does. 

        Without label corruption, Adam finds smooth minima that generalize
        resonably well everwhere, so the peak gets suppressed.

    Args:
        seed: int
            random seed (unique per run and per complexity)
        complexity: int
            model complexity
        X_train: array
            training features
        y_train: array
            training labels (clean)
        X_test: array
            test features
        y_test: array
            test labels (clean)
        model_name: str
            model identifier
        optimizer: str
            "adam" or "sgd" (just for NN)
        corruption: float
            label corruption rate (just for NN, 0.0 = no corruption)
    """
    set_seed(seed)

    # NN-specific label corruption (Nakkiran et al.)
    label_corruption = 0.0
    y_train_noisy = y_train.copy()

    if model_name == "nn" and corruption > 0:
        # Corrupt n% of labels randomly (helps with producing the interpolation peak)
        n = len(y_train)
        label_corruption = corruption
        corrupt_idx = np.random.choice(n, size=int(label_corruption * n), replace=False)
        y_train_noisy[corrupt_idx] = np.random.choice(y_train, size=len(corrupt_idx))    

    model = get_model(model_name, complexity, X_train.shape[1], X_train.shape[0], optimizer)
    model.fit(X_train, y_train_noisy)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse = np.mean((y_train_noisy - y_train_pred) ** 2)
    test_mse = np.mean((y_test - y_test_pred) ** 2)
    n_params = count_params(model)

    # Extract hyperparameters dynamically
    hyperparams = extract_hyperparameters(model_name, complexity, model)

    return {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "params": n_params,
        "label_corruption": label_corruption,
        **hyperparams
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["nn", "polynomial", "randomfeature", "kernelridge"])
    parser.add_argument("--dataset", required=True, choices=["friedman1", "friedman2", "friedman3"])
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory. Defaults to figures/{model}/")
    # NN-specific
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam",
                        help="Optimizer for nn model only.")
    parser.add_argument("--corruption", type=float, default=0.00,
                        help="Label corruption rate for nn model only (0.0 = no corruption).")
    args = parser.parse_args()
    
    model_name = args.model
    dataset_name = args.dataset
    logging.info(f"Running {model_name} on {dataset_name}.")

    # Load dataset
    X_train, X_test, y_train, y_test = load_dataset(dataset_name)

    complexities = get_complexities(model_name, X_train.shape[1], X_train.shape[0])

    # Prepare summary dictionary
    summary_data = {k: [] for k in [
        "complexity", "params_mean", "params_std",
        "train_mse_mean", "train_mse_std",
        "test_mse_mean", "test_mse_std",
        "learning_rate", "batch_size", "epochs",
        "alpha", "degree", "n_features", "label_corruption"
    ]}

    long_data = []

    # ---------------------------------------------------------------
    # Main Loop (average results across runs)
    for c in tqdm(complexities, desc=f"{model_name}-{dataset_name}"):

        # default: "loky" uses separate processes, requires pickling
        # "threading" uses threads in the same process, no pickling
        with tqdm_joblib(tqdm(desc="runs", total=N_RUNS, leave=False)):
            #results = Parallel(n_jobs=2, backend="threading")(
            results = Parallel(n_jobs=2, backend="loky")(
                delayed(run_seed)(
                    # Ensure that we are not reusing seeds
                    # Ensures different randomness per complexity
                    # Seeding is independent across runs, independent across complexities
                    42 + run + 1000 * c, 
                    c,
                    X_train, y_train,
                    X_test, y_test,
                    model_name,
                    args.optimizer,
                    args.corruption
                )
                for run in range(N_RUNS)
            )

        keys = results[0].keys()
        aggregated = {k: [r[k] for r in results] for k in keys}

        # Summary statistics
        summary_data["complexity"].append(c)
        summary_data["params_mean"].append(np.nanmean(aggregated["params"]))
        summary_data["params_std"].append(np.nanstd(aggregated["params"]))
        summary_data["train_mse_mean"].append(np.mean(aggregated["train_mse"]))
        summary_data["train_mse_std"].append(np.std(aggregated["train_mse"]))
        summary_data["test_mse_mean"].append(np.mean(aggregated["test_mse"]))
        summary_data["test_mse_std"].append(np.std(aggregated["test_mse"]))

        # Hyperparameters (take first run)
        for hp in ["learning_rate", "batch_size", "epochs", "alpha", "degree", "n_features", "label_corruption"]:
            summary_data[hp].append(aggregated[hp][0])

        # Long-format per-run data
        for run_idx in range(N_RUNS):
            row = {"complexity": c, "run": run_idx}
            for k in keys:
                row[k] = aggregated[k][run_idx]
            long_data.append(row)

    # ---------------------------------------------------------------

    # Check
    print(f"Min train error: {min(summary_data['train_mse_mean']):.4f}")     # we expect this to be close to 0 for large models
    print(f"Min test error: {min(summary_data['test_mse_mean']):.4f}")

    # Save summary CSV
    if args.output_dir:
        model_dir = args.output_dir
    else:
        model_dir = os.path.join("figures", model_name)
    os.makedirs(model_dir, exist_ok=True)
    summary_csv = os.path.join(model_dir, f"{dataset_name}_metrics_summary.csv")
    pd.DataFrame(summary_data).to_csv(summary_csv, index=False)
    print(f"[INFO] Summary metrics saved to {summary_csv}.")

    # Save long-format per-run CSV
    long_csv = os.path.join(model_dir, f"{dataset_name}_metrics_per_run.csv")
    pd.DataFrame(long_data).to_csv(long_csv, index=False)
    print(f"[INFO] Per-run metrics saved to {long_csv}.")

    # Plotting
    plot_double_descent(
        summary_data["complexity"],
        summary_data["train_mse_mean"],
        summary_data["test_mse_mean"],
        param_counts=summary_data["params_mean"],
        model_name=f"{model_name}_{dataset_name}",
        filename=os.path.join(model_dir.replace("figures/", ""), f"{dataset_name}_double_descent.png"),
        threshold=X_train.shape[0]
    )

if __name__ == "__main__":
    main()