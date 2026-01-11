import numpy as np

class Estimator:
    def fit(self, X: np.ndarray) -> tuple[tuple | np.ndarray, float]:
        raise NotImplementedError("fit method not implemented")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("predict method not implemented")
    
    def fit_predict(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        params, spectral_norm = self.fit(X)
        Y_hat = self.predict(X)
        return Y_hat, spectral_norm
    
if __name__ == "__main__":
    pass