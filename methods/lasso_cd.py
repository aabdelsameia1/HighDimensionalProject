# methods/lasso_cd.py
import numpy as np

def soft_thresholding(rho, lam):
    """
    Soft-thresholding operator.
    Returns:
      rho - lam if rho > lam,
      rho + lam if rho < -lam,
      0 otherwise.
    """
    if rho > lam:
        return rho - lam
    elif rho < -lam:
        return rho + lam
    else:
        return 0.0

def lasso_coordinate_descent(X, y, lam, max_iter=5000, tol=1e-4):
    """
    Solves the LASSO regression problem using coordinate descent.
    
    Arguments:
      X       : The standardized design matrix (n_samples x n_features)
      y       : The centered target vector (n_samples,)
      lam     : Regularization parameter (lambda)
      max_iter: Maximum number of iterations (complete passes over all features)
      tol     : Convergence tolerance (change in beta)
      
    Returns:
      beta         : The estimated coefficient vector (n_features,)
      beta_history : History of beta estimates (list, one per iteration)
    """
    n, p = X.shape
    beta = np.zeros(p)
    beta_history = []

    for iteration in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            # Compute the partial residual for feature j:
            residual = y - (X @ beta) + beta[j] * X[:, j]
            # Compute the correlation (rho) of feature j with the residual.
            rho = np.dot(X[:, j], residual) / n

            # Compute normalization factor (squared norm of feature j)
            norm_j = np.sum(X[:, j] ** 2) / n

            # Update beta[j] using the soft-thresholding operator.
            beta[j] = soft_thresholding(rho, lam) / norm_j

        beta_history.append(beta.copy())

        if np.linalg.norm(beta - beta_old, ord=2) < tol:
            print("Converged at iteration", iteration)
            break

    return beta, beta_history



# methods/lasso_cd_torch_optimized.py

# methods/lasso_cd_torch_numpy.py

# import numpy as np
# import torch

# def soft_thresholding(rho, lam):
#     """
#     Soft-thresholding operator.

#     Given a scalar `rho` and a regularization parameter `lam`, this operator shrinks
#     the value towards zero. Specifically, it returns:
#       - rho - lam  if rho > lam,
#       - rho + lam  if rho < -lam,
#       - 0          otherwise.
    
#     This is a key step in promoting sparsity in the LASSO regression solution.
    
#     Args:
#         rho (float): The input scalar.
#         lam (float): The regularization parameter (lambda).
        
#     Returns:
#         float: The thresholded value.
    
#     More info: https://en.wikipedia.org/wiki/Soft_thresholding
#     """
#     if rho > lam:
#         return rho - lam
#     elif rho < -lam:
#         return rho + lam
#     else:
#         return 0.0

# def lasso_coordinate_descent(X, y, lam, max_iter=5000, tol=1e-4, device=None):
#     """
#     Solves the LASSO regression problem using coordinate descent optimized with PyTorch.
    
#     This function accepts NumPy arrays as input and returns NumPy arrays.
#     Internally, it converts the inputs to PyTorch tensors, uses incremental residual 
#     updates, and automatically selects the best device (MPS for macOS if available).
    
#     The LASSO problem is formulated as:
#         minimize (1/(2n)) * ||y - Xβ||² + lam * ||β||₁
#     where X is the standardized design matrix and y is the centered target vector.
    
#     Args:
#         X (np.ndarray or torch.Tensor): Standardized design matrix (n_samples x n_features).
#         y (np.ndarray or torch.Tensor): Centered target vector (n_samples,).
#         lam (float): Regularization parameter (lambda).
#         max_iter (int): Maximum number of full passes over all features.
#         tol (float): Convergence tolerance for the L2 norm change in beta.
#         device (torch.device, optional): The device to run computations on.
    
#     Returns:
#         beta (np.ndarray): The estimated coefficient vector (n_features,).
#         beta_history (list of np.ndarray): History of beta estimates per iteration.
    
#     Further reading:
#       - Coordinate Descent: https://arxiv.org/abs/1509.09079
#       - scikit-learn LASSO: https://scikit-learn.org/stable/modules/linear_model.html#lasso
#     """
#     # Convert inputs from NumPy arrays to PyTorch tensors if needed.
#     if isinstance(X, np.ndarray):
#         X = torch.from_numpy(X).float()
#     if isinstance(y, np.ndarray):
#         y = torch.from_numpy(y).float()
    
#     # Device selection: use provided device, or auto-select (MPS on Mac if available, otherwise CPU)
#     if device is None:
#         if torch.backends.mps.is_available():
#             device = torch.device("mps")
#         else:
#             device = torch.device("cpu")
    
#     # Move tensors to the chosen device.
#     X = X.to(device)
#     y = y.to(device)
    
#     n, p = X.shape
#     beta = torch.zeros(p, device=device, dtype=X.dtype)
#     beta_history = []
    
#     # Initial residual: r = y - Xβ. With beta = 0, r starts as y.
#     r = y.clone()
    
#     # Precompute the squared norm (averaged over samples) for each feature.
#     norm_j = torch.sum(X * X, dim=0) / n  # Shape: (p,)
    
#     for iteration in range(max_iter):
#         beta_old = beta.clone()
#         for j in range(p):
#             # Compute the partial residual for feature j.
#             # Since r = y - Xβ (with the j-th term not yet added), we add back beta[j]*X[:, j]
#             r_partial = r + beta[j] * X[:, j]
            
#             # Compute the correlation (rho) between X[:, j] and the partial residual.
#             rho = torch.dot(X[:, j], r_partial) / n
            
#             # Save the old value of beta[j] for updating the residual later.
#             beta_j_old = beta[j].item()
            
#             # Compute the updated beta[j] using the soft-thresholding operator.
#             if norm_j[j].item() != 0:
#                 beta_j_new = soft_thresholding(rho.item(), lam) / norm_j[j].item()
#             else:
#                 beta_j_new = 0.0
            
#             # Update the residual vector:
#             # delta = (new beta[j] - old beta[j]) is the change in coefficient.
#             delta = beta_j_new - beta_j_old
#             r = r - delta * X[:, j]
            
#             # Update the coefficient for feature j.
#             beta[j] = beta_j_new
        
#         beta_history.append(beta.clone())
        
#         # Check convergence: if the L2 norm of the change in beta is below tol, break.
#         if torch.norm(beta - beta_old, p=2) < tol:
#             print("Converged at iteration", iteration)
#             break
    
#     # Convert the final beta and beta_history back to NumPy arrays.
#     beta_np = beta.cpu().detach().numpy()
#     beta_history_np = [b.cpu().detach().numpy() for b in beta_history]
    
#     return beta_np, beta_history_np