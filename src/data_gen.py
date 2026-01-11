import numpy as np

def generate_synthetic_data(
        T: int=200,  m: int=10,  
        seed: int | np.random.Generator=42, 
        style: str="VAR", force_symmetry: bool=False
    ):
    if isinstance(seed, int):
        rng = np.random.default_rng(seed)
    else:
        rng = seed
        
    # Not sure if the authors used VAR(1) or MAR(1) to generate data
    if style == "MAR":
        A = rng.standard_normal((m, m))
        B = rng.standard_normal((m, m))
        if force_symmetry:
            A = 1/2 * (A + A.T)
            B = 1/2 * (B + B.T)
        phi_r = np.kron(B, A)
    elif style == "VAR":
        phi_r = rng.standard_normal((m**2, m**2))
    else:
        raise ValueError(f"Unknown style: {style}")

    m = m**2
    
    # Rescale to ensure spectral radius < 1
    eigvals = np.linalg.eigvals(phi_r)
    rho = np.max(np.abs(eigvals))
    phi = phi_r / (rho + 1.0)
    assert np.max(np.abs(np.linalg.eigvals(phi))) < 1.0
    
    # Construct covariance matrix for noise
    S = rng.standard_normal((m, m))
    S_s = 1/2 * (S + S.T)
    D, Q = np.linalg.eigh(S_s)
    D = np.diag(np.abs(D))  # Ensure non-negative eigen
    Sigma = Q @ D @ Q.T
    assert np.allclose(Sigma, Sigma.T)
    assert np.all(np.linalg.eigvals(Sigma) >= 0)
    
    # Generate noise
    epsilon = rng.multivariate_normal(
        mean=np.zeros(m), cov=Sigma, size=T
    )

    # Generate the VAR(1) process
    X = np.zeros((T, m))
    X[0] = epsilon[0]
    for t in range(1, T):
        X[t] = phi @ X[t-1] + epsilon[t]

    return phi, X

if __name__ == "__main__":
    pass