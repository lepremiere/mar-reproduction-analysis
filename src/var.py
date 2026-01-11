import numpy as np
from scipy.linalg import pinv

from src.base import Estimator

def compute_gammas(Y: np.ndarray):
    T, m = Y.shape
    Gamma_0 = (Y.T @ Y) / T
    Gamma_1 = (Y[1:].T @ Y[:-1]) / (T - 1)
    return Gamma_0, Gamma_1

def yule_walker_var(Y: np.ndarray) -> np.ndarray:
    Gamma0, Gamma1 = compute_gammas(Y)
    return Gamma1 @ pinv(Gamma0)

def burg_var(Y: np.ndarray) -> np.ndarray:
    f, b = Y[1:], Y[:-1]
    numer = 2.0 * (f.T @ b)
    denom = (f.T @ f) + (b.T @ b)
    return numer @ pinv(denom)

def ols_var(Y: np.ndarray) -> np.ndarray:    
    return np.linalg.lstsq(Y[:-1], Y[1:], rcond=None)[0].T

def predict_var(Y: np.ndarray, phi: np.ndarray) -> np.ndarray:
    Y_hat = np.zeros_like(Y)
    Y_hat[0] = Y[0]
    Y_hat[1:] = (phi @ Y[:-1].T).T
    return Y_hat

class VAR_Estimator(Estimator):
    def __init__(self, mode: str='ols'):
        self.phi = None
        self.mode = mode.lower()

    def fit(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        if self.mode == 'ols':
            self.phi = ols_var(X)
        elif self.mode == 'yw':
            self.phi = yule_walker_var(X)
        elif self.mode == 'burg':
            self.phi = burg_var(X)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Check stationarity
        eigvals = np.linalg.eigvals(self.phi)
        spectral_norm = np.max(np.abs(eigvals))
        if spectral_norm >= 1:
            print(
                f"Warning: Estimated VAR ({self.mode}) model is not stationary. ",
                f"Shape for phi = {self.phi.shape}. "
                f"Spectral norm: {spectral_norm:.3f}",
            )
            
        return self.phi, spectral_norm

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.phi is None:
            raise ValueError("Model is not fitted yet.")
        return predict_var(X, self.phi)
    
if __name__ == "__main__":
    pass