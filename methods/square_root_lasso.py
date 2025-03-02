# methods/square_root_lasso.py
import numpy as np
# Import our coordinate descent LASSO solver.
from methods.lasso_cd import lasso_coordinate_descent
from methods.lasso_fista import lasso_fista

def square_root_lasso(X, y, tau, max_outer_iter=100, tol=1e-4):
    """
    Solves the square-root LASSO (scaled LASSO) problem via alternating minimization.
    
    We aim to solve:
    
        minimize_{b, sigma > 0} { sigma/2 + (1/(2*sigma*n))*||y - Xb||_2^2 + tau * ||b||_1 }.
    
    The algorithm alternates between:
      1. Updating b by solving:
             min_b { (1/(2n))*||y - Xb||_2^2 + (tau * sigma)*||b||_1 }
         using lasso_coordinate_descent (with effective lambda = tau * sigma).
      2. Updating sigma as:
             sigma = ||y - Xb||_2 / sqrt(n)
    
    Parameters:
        X             : Design matrix (n_samples x n_features). (Assumed standardized.)
        y             : Target vector (n_samples,). (Assumed centered.)
        tau           : Regularization parameter for the square-root LASSO.
        max_outer_iter: Maximum number of outer iterations.
        tol           : Convergence tolerance for relative changes in sigma and b.
    
    Returns:
        b       : Estimated coefficient vector.
        sigma   : Estimated noise level.
        history : List of tuples (sigma, b) for each outer iteration.
    """
    n, p = X.shape
    # Initialize sigma with a natural guess: the norm of y divided by sqrt(n)
    sigma = np.linalg.norm(y) / np.sqrt(n)
    b = np.zeros(p)
    history = []
    
    for outer_iter in range(max_outer_iter):
        sigma_old = sigma
        b_old = b.copy()
        
        # --- Step 1: Update b ---
        # Solve the LASSO subproblem:
        #   min_b { (1/(2n))*||y - Xb||_2^2 + (tau * sigma)*||b||_1 }.
        # Here we call our lasso_coordinate_descent with effective lambda = tau * sigma.
        b, _ = lasso_fista(X, y, lam=tau * sigma, max_iter=5000, tol=1e-6)
        
        # --- Step 2: Update sigma ---
        residual = y - X @ b
        sigma = np.linalg.norm(residual) / np.sqrt(n)
        
        history.append((sigma, b.copy()))
        
        # Check convergence: we use relative changes in sigma and in b.
        sigma_change = np.abs(sigma - sigma_old) / (sigma_old + 1e-8)
        b_change = np.linalg.norm(b - b_old) / (np.linalg.norm(b_old) + 1e-8)
        
        if sigma_change < tol and b_change < tol:
            print("Square-root LASSO converged at outer iteration", outer_iter)
            break
            
    return b, sigma, history