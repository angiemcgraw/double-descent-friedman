"""
Main entry point for all models and datasets.

Usage (from the base directory: /double-descent-friedman)
python3 -m experiments.run --model nn --dataset friedman1
python3 -m experiments.run --model polynomial --dataset friedman1

Models supported:
1. Polynomial Regression
2. Random Feature Regression
3. Kernel Ridge Regresion
4. Neural Network

Angie McGraw
Last updated: March 28th, 2026
"""

import numpy as np
import argparse
import logging
from tqdm import tqdm

from utils.data_loader import load_dataset
from utils.plotting import plot_double_descent

# Import models
#from models.polynomial import PolynomialRegression
#from models.randomfeature import RandomFeatureRegression
#from models.kernelridge import KernelRidgeRegression
from models.neural_network import NeuralNetwork

logging.basicConfig(level=logging.INFO)

def get_model(model_name, complexity, input_dim):
    if model_name == "nn":
        return NeuralNetwork(input_dim=input_dim, complexity=complexity, epochs=300)
    #elif model_name == "polynomial:
        #return PolynomialRegression(degree=complexity)
    #elif model_name == "randomfeature":
    #elif model_name == "kernelridge":
    else:
        raise ValueError(f"Unknown model: {model_name}.")

def get_complexities(model_name):
    """
    Complexity grids.
    """
    if model_name == "nn":
        return list(range(1, 25)) + [30, 40, 50, 75, 100, 200, 500, 1000]
    #elif model_name == "polynomial:
        #return list(range(1, 30))
    #elif model_name == "randomfeature":
    #elif model_name == "kernelridge":
    else:
        raise ValueError(f"Unknown model: {model_name}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["nn", "polynomial", "randomfeature", "kernelridge"])
    parser.add_argument("--dataset", required=True, choices=["friedman1", "friedman2", "friedman3"])
    args = parser.parse_args()

    model_name = args.model
    dataset_name = args.dataset
    logging.info(f"Running {model_name} on {dataset_name}.")

    # Load dataset
    X_train, X_test, y_train, y_test = load_dataset(dataset_name)

    complexities = get_complexities(model_name)
    train_errors = []
    test_errors = []

    for c in tqdm(complexities, desc=f"{model_name}-{dataset_name}"):
        model = get_model(model_name, c, X_train.shape[1])
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_mse = np.mean((y_train - y_train_pred) ** 2)
        test_mse = np.mean((y_test - y_test_pred) ** 2)

        train_errors.append(train_mse)
        test_errors.append(test_mse)

    # Plot double descent curves
    plot_double_descent(
        complexities,
        train_errors,
        test_errors,
        model_name = f"{model_name}_{dataset_name}"
    )

if __name__ == "__main__":
    main()