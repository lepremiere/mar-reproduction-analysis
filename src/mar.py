import numpy as np
import scipy.linalg
from scipy.optimize import minimize

from src.base import Estimator
from src.var import compute_gammas, yule_walker_var, burg_var, ols_var

# Note: Paper Def A2 defines vec(A) as stacking columns.
ORDER = 'F'
TOL = 1e-9
ITERS = 1000

def unvec_X(x, M):
    T = x.shape[0]
    return x.reshape(T, M, M, order=ORDER)

def vec_X(X):
    T = X.shape[0]
    return X.reshape(T, -1, order=ORDER)

def vec(A):
    return A.reshape(-1, order=ORDER)

def unvec(A, M):
    return A.reshape(M, M, order=ORDER)

def _normalize_ab(A, B):
    """Enforces ||A||_F = 1 and scales B correspondingly."""
    norm_A = np.linalg.norm(A, "fro")
    A = A / norm_A
    B = B * norm_A
    return A, B

def _solve_nkp(Phi, m, n):
    """
    Decomposes the transition matrix Phi (size mn x mn) into the closest 
    MAR components A (m x m) and B (n x n) such that Phi approx B (x) A.
    """
    # Reshape to separate the dimensions of A and B
    Phi_reshaped = Phi.reshape(m, n, m, n, order=ORDER)
    
    # Rearrange (Permute) to group A-indices and B-indices
    Phi_permuted = Phi_reshaped.transpose(0, 2, 1, 3)
    
    # Flatten to create the Rearrangement Matrix
    R_Phi = Phi_permuted.reshape(m * m, n * n, order=ORDER)
    
    # SVD 
    U, S, Vt = np.linalg.svd(R_Phi)
    sigma = S[0]
    scale = np.sqrt(sigma)
    
    # Reshape back to matrices
    A_hat = U[:, 0].reshape(m, m, order='F') * scale
    B_hat = Vt[0, :].reshape(n, n, order='F') * scale
    
    return _normalize_ab(A_hat, B_hat)

def vec_MAR_yw(X):
    """Estimate via Standard VAR Yule-Walker then Project."""
    T, m, n = X.shape
    X_vec = vec_X(X)
    Phi_hat = yule_walker_var(X_vec)
    return _solve_nkp(Phi_hat, m, n)

def vec_MAR_burg(X):
    """Estimate via Multivariate Burg then Project."""
    T, m, n = X.shape
    X_vec = vec_X(X)
    Phi_hat = burg_var(X_vec)
    return _solve_nkp(Phi_hat, m, n)

def vec_MAR_ols(X):
    """Estimate via Multivariate OLS then Project."""
    T, m, n = X.shape
    X_vec = vec_X(X)
    Phi_hat = ols_var(X_vec)
    return _solve_nkp(Phi_hat, m, n)

def predict_mar(X: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    T, m, n = X.shape
    X_hat = np.zeros_like(X)
    X_hat[0] = X[0]
    
    for t in range(1, T):
        X_hat[t] = A @ X[t - 1] @ B.T
    
    return X_hat

def yule_walker_MAR(X):
    T, m, n = X.shape
    dim = m * n
    
    # Compute autocovariances
    X_vec = vec_X(X)
    Gamma_0, Gamma_1 = compute_gammas(X_vec)
    
    # Permutation Logic for MAR YW
    indices = np.arange(dim)
    i_grid, j_grid = np.meshgrid(indices, indices, indexing='ij')
    row_map = (j_grid // m) * m + (i_grid // n)
    col_map = (i_grid % n) * m + (j_grid % m)
    
    Gamma_0_kron = Gamma_0[row_map, col_map]
    Gamma_1_kron = Gamma_1[row_map, col_map]
    
    # Optimization specific to MAR YW
    def objective(params):
        split = m * m
        A_flat, B_flat = params[:split], params[split:]
        A, B = unvec(A_flat, m), unvec(B_flat, n)
        A, B = _normalize_ab(A, B)
        
        A_B = np.kron(A, np.eye(n))
        B_B = np.kron(B.T, np.eye(m))
        
        Prediction = A_B @ Gamma_0_kron @ B_B
        residual = np.linalg.norm(Gamma_1_kron - Prediction, 'fro') # (4)
        
        return residual
    
    x0 = np.concatenate([vec(np.eye(m)), vec(np.eye(n))])
    
    # Run optimization
    result = minimize(
        objective, x0, method='L-BFGS-B', options={'maxiter': ITERS, 'ftol': TOL}
    )

    A_est = unvec(result.x[:m*m], m)
    B_est = unvec(result.x[m*m:], n)
    return _normalize_ab(A_est, B_est) # Enforce ||A||_F = 1

def burg_MAR(X):
    """Iterative MAR Burg Estimator."""
    T, m, n = X.shape
    
    # Initialize
    A, B = _normalize_ab(np.eye(m), np.eye(n))
    
    # Residuals
    f = X[1:] # t=1..T-1
    b = X[:-1] # t=0..T-2
    
    f_T = f.transpose(0, 2, 1)
    b_T = b.transpose(0, 2, 1)
    
    for _ in range(ITERS):
        A_prev, B_prev = A.copy(), B.copy()
        
        # Precompute products
        BTB = B.T @ B
        
        # --- UPDATE A via Eq. (7) ---       
        Numerator_A = np.sum(f @ B @ b_T + b @ B @ f_T, axis=0)
        Denominator_A = np.sum(b @ BTB @ b_T + f @ BTB @ f_T, axis=0)
        
        A = Numerator_A @ scipy.linalg.pinv(Denominator_A)
        A = A / np.linalg.norm(A, 'fro') # Normalize A
        
        # Precompute products
        A_T = A.T
        ATA = A_T @ A
        
        # --- UPDATE B via Eq. (8) ---        
        Numerator_B = np.sum(b_T @ A_T @ f + f_T @ A_T @ b, axis=0)
        Denominator_B = np.sum(b_T @ ATA @ b + f_T @ ATA @ f, axis=0)
        
        B = (scipy.linalg.pinv(Denominator_B) @ Numerator_B).T
        
        # Abort if converged
        if np.linalg.norm(A - A_prev) + np.linalg.norm(B - B_prev) < TOL:
            break
            
    return _normalize_ab(A, B)

def lse_mar(X):
    """Least Squares Estimator for MAR."""
    T, m, n = X.shape
    
    # Initialize
    A, B = _normalize_ab(np.random.randn(m, m), np.random.randn(n, n))
    
    # Residuals
    f = X[1:] # t=1..T-1
    b = X[:-1] # t=0..T-2
    
    # Precompute Transposes
    f_T = f.transpose(0, 2, 1)
    b_T = b.transpose(0, 2, 1)
    
    for _ in range(ITERS):
        A_prev, B_prev = A.copy(), B.copy()
        
        # Precompute products
        BTB = B.T @ B
        
        # --- UPDATE A ---
        Numerator_A = np.sum(f @ B @ b_T, axis=0)
        Denominator_A = np.sum(b @ BTB @ b_T, axis=0)
        
        A = Numerator_A @ scipy.linalg.pinv(Denominator_A)
        A = A / np.linalg.norm(A, 'fro') # Normalize A
        
        # Precompute products
        ATA = A.T @ A
        
        # --- UPDATE B ---
        Numerator_B = np.sum(f_T @ A @ b, axis=0)        
        Denominator_B = np.sum(b_T @ ATA @ b, axis=0)
        
        B = Numerator_B @ scipy.linalg.pinv(Denominator_B)
        
        # Abort if converged
        if np.linalg.norm(A - A_prev) + np.linalg.norm(B - B_prev) < TOL:
            break
        
    return A, B

class MAR_Estimator(Estimator):
    def __init__(self, mode: str='yw'):
        self.A = None
        self.B = None
        self.mode = mode.lower()

    def fit(self, X: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], float]:
        if len(X.shape) != 3:
            X = unvec_X(X, int(np.sqrt(X.shape[1])))
        if self.mode == 'yw':
            self.A, self.B = yule_walker_MAR(X)
        elif self.mode == 'burg':
            self.A, self.B = burg_MAR(X)
        elif self.mode == 'lse':
            self.A, self.B = lse_mar(X)
        elif self.mode == 'vec_yw':
            self.A, self.B = vec_MAR_yw(X)
        elif self.mode == 'vec_burg':
            self.A, self.B = vec_MAR_burg(X)
        elif self.mode == 'vec_ols':
            self.A, self.B = vec_MAR_ols(X)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Check stationarity
        rho_A = np.max(np.abs(np.linalg.eigvals(self.A)))
        rho_B = np.max(np.abs(np.linalg.eigvals(self.B)))
        spectral_norm = rho_A * rho_B
        if spectral_norm >= 1.0:
            print(
            f"Warning: Estimated MAR ({self.mode}) model is not stationary. ",
            f"Shape for A = {self.A.shape} and B = {self.B.shape}. "
            f"Spectral norms: rho(A)={rho_A:.4f}, rho(B)={rho_B:.4f}, ",
            f"rho(A)*rho(B)={spectral_norm:.4f}."
        )
                
        return (self.A, self.B), spectral_norm

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) != 3:
            X = unvec_X(X, int(np.sqrt(X.shape[1])))
        if self.A is None or self.B is None:
            raise ValueError("Model is not fitted yet.")
        X_hat = predict_mar(X, self.A, self.B)
        return vec_X(X_hat)

if __name__ == "__main__":
    pass