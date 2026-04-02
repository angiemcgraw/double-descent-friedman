import pandas as pd
from utils.plotting import plot_double_descent

def plot_from_csv(csv_path, model_name, threshold=None):
    df = pd.read_csv(csv_path)

    complexities = df["complexity"].values
    train_errors = df["train_mse"].values
    test_errors = df["test_mse"].values
    param_counts = df["params"].values

    plot_double_descent(
        complexities,
        train_errors,
        test_errors,
        param_counts=param_counts,
        model_name=model_name,
        filename=f"{model_name}_double_descent.png",
        threshold=threshold
    )

if __name__ == "__main__":
    plot_from_csv(
        "figures/nn/friedman1_metrics.csv",
        model_name="nn_friedman1",
        threshold=200
    )