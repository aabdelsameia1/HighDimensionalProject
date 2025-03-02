# methods/lasso_fista.py
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

def lasso_fista(X, y, lam, L=None, max_iter=5000, tol=1e-4):
    """
    Solves the LASSO regression problem using the FISTA algorithm.
    
    Objective:
         minimize (1/(2n)) * ||y - Xb||^2 + lam * ||b||_1
    
    FISTA updates:
         b^{(k+1)} = S_{lam/L}(y^{(k)} - (1/L)*grad_f(y^{(k)}))
         t_{k+1} = (1 + sqrt(1+4*t_k^2))/2
         y^{(k+1)} = b^{(k+1)} + ((t_k - 1)/t_{k+1})*(b^{(k+1)} - b^{(k)})
    
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
        b_history : List of coefficient vectors (one per iteration)
    """
    n, p = X.shape
    # Compute the Lipschitz constant if not provided.
    if L is None:
        L = np.linalg.norm(X, 2)**2 / n

    # Initialize variables.
    b = np.zeros(p)         # b^{(0)}
    yk = b.copy()           # y^{(0)} = b^{(0)}
    t = 1.0                 # t_0 = 1
    b_history = []

    for iteration in range(max_iter):
        b_old = b.copy()

        # Compute gradient at yk.
        grad = (X.T @ (X @ yk - y)) / n

        # ISTA step on the extrapolated point yk.
        b = soft_thresholding(yk - (1.0 / L) * grad, lam / L)

        # Update momentum parameter.
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t**2))

        # Update the extrapolated point.
        yk = b + ((t - 1) / t_new) * (b - b_old)

        # Update t.
        t = t_new

        b_history.append(b.copy())

        # Check convergence.
        if np.linalg.norm(b - b_old, ord=2) < tol:
            print("FISTA converged at iteration", iteration)
            break

    return b, b_history
