# data/synthetic_data.py
import numpy as np
from sklearn.datasets import make_regression

def generate_synthetic_data(n_samples=100, n_features=1000, n_informative=20, noise=3, random_state=42):
    """
    Generates synthetic regression data using make_regression.

    Arguments:
      n_samples    : Number of samples.
      n_features   : Total number of features.
      n_informative: Number of informative (non-zero) features.
      noise        : Noise level.
      random_state : Random seed for reproducibility.

    Returns:
      X_data   : Feature matrix of shape (n_samples, n_features).
      y_data   : Target vector of shape (n_samples,).
      true_coef: The true coefficient vector (n_features,).
    """
    X_data, y_data, true_coef = make_regression(n_samples=n_samples,
                                                n_features=n_features,
                                                n_informative=n_informative,
                                                noise=noise,
                                                coef=True,
                                                random_state=random_state)
    return X_data, y_data, true_coef