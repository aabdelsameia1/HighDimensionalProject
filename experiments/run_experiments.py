import sys
import os

# Add the project root to sys.path so that imports work correctly.
project_root = os.path.join(os.path.dirname(__file__), '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt

# Import your project modules.
from data.synthetic_data import generate_synthetic_data
from methods.lasso_cd import lasso_coordinate_descent
from methods.lasso_fista import lasso_fista
from methods.lasso_ista import lasso_ista
from methods.square_root_lasso import square_root_lasso
from utils.plotting import plot_coefficient_evolution  # A custom plotting function.

# Define the configuration directly in code.
# Modify these parameters to adjust your experiments.
config = {
    "dataset": {
        "type": "synthetic",      # Only synthetic is supported here.
        "n_samples": 100,
        "n_features": 1000,
        "n_informative": 20,
        "noise": 20,
        "random_state": 42
    },
    "algorithms": {
        # "lasso_cd": {
        #     "lam": 0.1,
        #     "max_iter": 5000,
        #     "tol": 1e-4,
        #     "threshold": 1e-2  # Threshold for feature selection.
        # },
        "lasso_fista": {
            "lam": 0.1,
            "max_iter": 5000,
            "tol": 1e-4,
            "threshold": 0.1
        },
        # "lasso_ista": {
        #     "lam": 0.1,
        #     "max_iter": 100000,
        #     "tol": 1e-4,
        #     "threshold": 0.1
        # },
        # "square_root_lasso": {
        #     "tau": 0.2,
        #     "max_outer_iter": 250,
        #     "tol": 1e-4,
        #     "threshold": 0.0  # For square-root lasso, you might want to use exact zero.
        # }
    }
}

def run_experiment(algo_name, X_scaled, y_centered, config_algo):
    """Run the selected LASSO algorithm using the provided configuration."""
    print(f"Running experiment for {algo_name}...")
    if algo_name == "lasso_cd":
        beta_est, beta_history = lasso_coordinate_descent(
            X_scaled, y_centered,
            lam=config_algo["lam"],
            max_iter=config_algo["max_iter"],
            tol=config_algo["tol"]
        )
    elif algo_name == "lasso_fista":
        beta_est, beta_history = lasso_fista(
            X_scaled, y_centered,
            lam=config_algo["lam"],
            max_iter=config_algo["max_iter"],
            tol=config_algo["tol"]
        )
    elif algo_name == "lasso_ista":
        beta_est, beta_history = lasso_ista(
            X_scaled, y_centered,
            lam=config_algo["lam"],
            max_iter=config_algo["max_iter"],
            tol=config_algo["tol"]
        )
    elif algo_name == "square_root_lasso":
        beta_est, sigma_est, beta_history = square_root_lasso(
            X_scaled, y_centered,
            tau=config_algo["tau"],
            max_outer_iter=config_algo["max_outer_iter"],
            tol=config_algo["tol"]
        )
    else:
        raise ValueError("Unknown algorithm")
    
    beta_history_arr = np.array(beta_history)
    return beta_est, beta_history_arr

def evaluate_and_plot(beta_est, beta_history_arr, true_informative_indices, X_scaled, config_algo, algo_name):
    """
    Evaluate the selected features compared to the ground truth and create plots.
    
    Uses a threshold defined in config_algo to select features, prints out results,
    and calls a custom plotting function to visualize coefficient evolution.
    """
    threshold = config_algo.get("threshold", 1e-2)
    selected_indices = np.where(beta_est > threshold)[0]
    n_features = X_scaled.shape[1]
    n_selected = len(selected_indices)
    n_not_selected = n_features - n_selected
    
    # Print summary.
    print(f"\n{algo_name} results:")
    print("Number of features selected (nonzero coefficients):", n_selected)
    print("Indices of selected features:", selected_indices)
    common_features = set(selected_indices).intersection(set(true_informative_indices))
    fraction = len(common_features) / len(true_informative_indices) if len(true_informative_indices) > 0 else 0
    print("Recovered informative features (intersection):", sorted(common_features))
    print("Fraction of ground truth features recovered: {}/{} = {:.2f}".format(
        len(common_features), len(true_informative_indices), fraction))
    
    # Plot the coefficient evolution using the custom plotting function.
    plot_coefficient_evolution(beta_history_arr, selected_indices, n_features)

def main():
    # Data Generation.
    data_config = config["dataset"]
    X_data, y_data, true_coef = generate_synthetic_data(
        n_samples=data_config["n_samples"],
        n_features=data_config["n_features"],
        n_informative=data_config["n_informative"],
        noise=data_config["noise"],
        random_state=data_config["random_state"]
    )
    true_informative_indices = np.where(true_coef != 0)[0]
    
    # Data Standardization.
    X_mean = np.mean(X_data, axis=0)
    X_std = np.std(X_data, axis=0)
    X_scaled = (X_data - X_mean) / X_std
    y_centered = y_data - np.mean(y_data)
    
    # Loop over each algorithm defined in the configuration.
    for algo_name, algo_config in config["algorithms"].items():
        beta_est, beta_history_arr = run_experiment(algo_name, X_scaled, y_centered, algo_config)
        evaluate_and_plot(beta_est, beta_history_arr, true_informative_indices, X_scaled, algo_config, algo_name)

if __name__ == '__main__':
    main()
