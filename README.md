# MAR Reproduction Analysis

Python implementation and reproduction study of Matrix Autoregressive (MAR) and Vector Autoregressive (VAR) model estimation methods.

## Overview

This repository reproduces the experiments from the paper:

> Kołodziejski, K. (2025). Estimation methods of Matrix-valued AR model. doi: 10.48550/arXiv.2505.15220

The implementation includes multiple estimation algorithms for both MAR(1) and VAR(1) models, along with simulation studies comparing their forecasting performance across different data-generating processes and dimensions. The reproduction analysis is documented in [paper.pdf](paper.pdf).

## Contents

### Estimation Methods

**MAR(1) Estimators:**
- `yule_walker_MAR` - Direct numerical optimization of Yule-Walker equations
- `burg_MAR` - Iterative Burg algorithm for MAR
- `lse_mar` - Least Squares Estimation
- `vec_MAR_yw`, `vec_MAR_burg`, `vec_MAR_ols` - Projection-based methods (estimate as VAR, project to MAR structure)

**VAR(1) Estimators:**
- `ols_var` - Ordinary Least Squares
- `yule_walker_var` - Yule-Walker equations
- `burg_var` - Multivariate Burg algorithm

## Repository Structure

```
.
├── src/
│   ├── base.py           # Base estimator interface
│   ├── mar.py            # MAR model implementations
│   ├── var.py            # VAR model implementations
│   ├── data_gen.py       # Synthetic data generation
│   └── metrics.py        # Evaluation metrics (MAE, RMSE, SMAPE)
├── notebooks/
│   ├── kronecker_energy.ipynb  # Kronecker product analysis
│   ├── plots.ipynb             # Visualization of results
│   └── tables.ipynb            # Result tabulation
├── results/
│   ├── experiment_results_MAR.csv
│   ├── experiment_results_MAR_symmetric.csv
│   ├── experiment_results_VAR.csv
│   └── experiment_results_VAR_flawed.csv
├── simulation_study.py   # Main simulation script with parallel execution
├── permutation_test.py   # Kronecker product permutation validation
├── paper.pdf             # Reproduction analysis paper
└── requirements.txt      # Python dependencies
```

## Installation

```bash
# Clone the repository
git clone https://github.com/lepremiere/mar-reproduction-analysis.git
cd mar-reproduction-analysis

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- numpy
- pandas
- scipy
- scikit-learn
- pingouin
- matplotlib

## Usage

### Running Simulation Studies

Execute the main simulation study with parallel processing:

```bash
python simulation_study.py
```

This will:
1. Generate synthetic VAR/MAR data with varying dimensions (m = 2 to 10)
2. Train models with different time series lengths (T = 100 to 500)
3. Compare all estimation methods across 100 trials
4. Save results to CSV files in the `results/` directory

### Reproducing Paper Figures and Tables

To reproduce the figures and tables from the paper, run the corresponding Jupyter notebooks:

**For paper figures:**
```bash
jupyter notebook notebooks/plots.ipynb
```
Run all cells in [plots.ipynb](notebooks/plots.ipynb) to generate the visualization figures from the paper.

**For paper tables:**
```bash
jupyter notebook notebooks/tables.ipynb
```
Run all cells in [tables.ipynb](notebooks/tables.ipynb) to generate the result tables from the paper.

**For Kronecker energy figure:**
```bash
jupyter notebook notebooks/kronecker_energy.ipynb
```
Run all cells in [kronecker_energy.ipynb](notebooks/kronecker_energy.ipynb) to generate the Kronecker energy analysis figure from the paper.

## Experimental Settings

The simulation study evaluates models under four scenarios:

1. **VAR (Flawed)**: VAR-generated data with incorrect MAR prediction on training data
2. **VAR (Correct)**: VAR-generated data with proper test set prediction
3. **MAR**: MAR-generated data
4. **MAR (Symmetric)**: MAR-generated data with symmetric transition matrices

## Results

Results are stored in CSV format with columns:
- `Estimator`: Model name and estimation method
- `m`: Base dimension
- `Dim`: Actual vector dimension (m²)
- `T`: Training sample size
- `MAE`, `RMSE`, `SMAPE`: Forecasting errors
- `Spectral Norm`: Largest eigenvalue magnitude (stationarity indicator)
- `Time Execution`: Computation time
- `p-values`: Multivariate normality test results

View detailed analysis in the Jupyter notebooks under `notebooks/`.

## API Reference

### Synthetic Data Generation

```python
from src.data_gen import generate_synthetic_data

# Generate MAR-style data
phi, X = generate_synthetic_data(
    T=200,              # Time series length
    m=5,                # Dimension (creates 5×5 = 25 variables)
    style="MAR",        # "MAR" or "VAR"
    force_symmetry=False,  # Force symmetric transition matrices
    seed=42
)
```

**Parameters:**
- `T`: Time series length
- `m`: Base dimension (actual dimension will be m²)
- `style`: "VAR" or "MAR" 
- `force_symmetry`: Whether to generate symmetric transition matrices
- `seed`: Random seed (int or numpy Generator)

**Returns:** `(phi, X)` where phi is the transition matrix and X is the generated data

### Model Estimation

```python
from src.mar import MAR_Estimator
from src.var import VAR_Estimator

# MAR estimation
mar_model = MAR_Estimator(mode='yw')  # 'yw', 'burg', 'lse', 'vec_yw', 'vec_burg'
(A, B), spectral_norm = mar_model.fit(X)
X_hat = mar_model.predict(X)

# VAR estimation
var_model = VAR_Estimator(mode='ols')  # 'ols', 'yw', 'burg'
phi, spectral_norm = var_model.fit(Y)
Y_hat = var_model.predict(Y)
```

**Estimator Interface:**

Both `MAR_Estimator` and `VAR_Estimator` follow the same interface:

```python
estimator.fit(X)           # Returns (params, spectral_norm)
estimator.predict(X)       # Returns predictions
estimator.fit_predict(X)   # Convenience method combining both
```

### Metrics

```python
from src.metrics import MAE, RMSE, SMAPE

mae = MAE(X_true, X_pred)      # Mean Absolute Error
rmse = RMSE(X_true, X_pred)    # Root Mean Square Error  
smape = SMAPE(X_true, X_pred)  # Symmetric MAPE (%)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.