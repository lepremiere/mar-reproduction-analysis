import numpy as np

def MAE(X_true: np.ndarray, X_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(X_true - X_pred), axis=None))

def RMSE(X_true: np.ndarray, X_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((X_true - X_pred) ** 2, axis=None)))

def SMAPE(X_true: np.ndarray, X_pred: np.ndarray, eps=1e-8) -> float:
    num = np.abs(X_true - X_pred)
    den = (np.abs(X_true) + np.abs(X_pred)) + eps
    return float(100 * np.mean(2 * num / den, axis=None))

if __name__ == "__main__":
    pass