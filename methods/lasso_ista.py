# methods/lasso_ista.py
import numpy as np

def soft_thresholding(v, threshold):
    """
    Vectorized soft-thresholding operator.
    
    For each element in v, computes:
        sign(v_i) * max(|v_i| - threshold, 0)
    
    Parameters:
        v         : Input numpy array.
        threshold : The threshold value.
    
    Returns:
        A numpy array after applying soft-thresholding.
    """
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)

def lasso_ista(X, y, lam, L=None, max_iter=5000, tol=1e-4):
    """
    Solves the LASSO regression problem using the ISTA algorithm.
    
    Objective:
         minimize (1/(2n)) * ||y - Xb||^2 + lam * ||b||_1
    
    Parameters:
        X        : Standardized design matrix (n_samples x n_features)
        y        : Centered target vector (n_samples,)
        lam      : Regularization parameter (lambda)
        L        : Lipschitz constant for the gradient of the smooth part.
                   If None, computed as L = (1/n) * (spectral norm of X)^2.
        max_iter : Maximum number of iterations.
        tol      : Convergence tolerance (change in b).
    
    Returns:
        b         : The estimated coefficient vector (n_features,)
        b_history : List of coefficient vectors from each iteration.
    """
    n, p = X.shape
    # Compute the Lipschitz constant if not provided.
    if L is None:
        L = np.linalg.norm(X, 2)**2 / n  # spectral norm squared divided by n
    
    b = np.zeros(p)  # initialize b to zeros
    b_history = []

    for iteration in range(max_iter):
        b_old = b.copy()
        # Compute the gradient of the smooth part:
        # f(b) = (1/(2n)) * ||y - Xb||^2  =>  grad f(b) = (1/n) * X^T (Xb - y)
        grad = (X.T @ (X @ b - y)) / n
        
        # Take a gradient descent step and then apply the soft-thresholding operator.
        b = soft_thresholding(b - (1 / L) * grad, lam / L)
        
        b_history.append(b.copy())
        
        # Check for convergence.
        if np.linalg.norm(b - b_old) < tol:
            print("ISTA converged at iteration", iteration)
            break

    return b, b_history