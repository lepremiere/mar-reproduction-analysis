import numpy as np
import pandas as pd
import multiprocessing

from src.base import Estimator
from src.var import VAR_Estimator
from src.mar import MAR_Estimator

from time import perf_counter
from pingouin import multivariate_normality
from src.data_gen import generate_synthetic_data
from src.metrics import MAE, RMSE, SMAPE


def run_experiment(
        T: int,
        m: int,
        estimator: Estimator,
        trials: int=100,
        n_test: int=100,
        style: str="VAR",
        seed: int | np.random.Generator=42,
        flawed: bool=False,
        symmetric: bool=False
    ) -> dict:
    if isinstance(seed, int):
        rng = np.random.default_rng(seed)
    else:
        rng = seed
        
    stds, maes, rmses, smapes, ps, spectral_norms, ts = [], [], [], [], [], [], []
    for _ in range(trials):
        # Generate synthetic data
        phi, Y = generate_synthetic_data(
            T=T + n_test, m=m, seed=rng, style=style, force_symmetry=symmetric
        )

        # Split into training and testing sets
        Y_train = Y[:-n_test]
        Y_test = Y[-n_test:]
        
        # Fit the estimator and make predictions
        t0 = perf_counter()
        Y_train_hat, spectral_norm = estimator.fit_predict(Y_train)
        t1 = perf_counter()
        
        # Predict on test set
        if flawed:
            Y_test_hat = Y_train_hat[-n_test:]
        else:
            Y_test_hat = estimator.predict(Y_test)
        
        # Check for multivariate normality
        residuals = Y_test - Y_test_hat
        _, p, _ = multivariate_normality(residuals, alpha=0.05)
        
        # Calculate evaluation metrics
        stds.append(np.std(Y))
        maes.append(MAE(Y_test, Y_test_hat))
        rmses.append(RMSE(Y_test, Y_test_hat))
        smapes.append(SMAPE(Y_test, Y_test_hat))
        spectral_norms.append(spectral_norm)
        ts.append(t1 - t0)
        ps.append(p)
        
    res = {
        "Std": np.array(stds),
        "MAE": np.array(maes),
        "RMSE": np.array(rmses),
        "SMAPE": np.array(smapes),
        "Spectral Norm": np.array(spectral_norms),
        "Time": np.array(ts),
        "p-values": np.array(ps)
    }
    return res


def run_single_trial(args):
    """
    Unpacks arguments and runs a single experiment configuration.
    Returns the configuration metadata and the result.
    """
    estimator_name, estimator_obj, m, T, n_trials, seed, style, flawed, symmetric = args
    
    # Run the existing experiment function
    res = run_experiment(
        T=T, m=m,
        estimator=estimator_obj,
        trials=n_trials,
        seed=seed,
        style=style,
        flawed=flawed,
        symmetric=symmetric
    )
    
    # Calculate means immediately to return lightweight data
    mae = np.mean(res["MAE"])
    rmse = np.mean(res["RMSE"])
    smape = np.mean(res["SMAPE"])
    spectral_norm = np.mean(res["Spectral Norm"])
    time_avg = np.mean(res["Time"])
    p = np.quantile(res["p-values"], 0.5)
    
    # Return metadata, the full result (for dataframe), and the summary metrics
    return (
        estimator_name, m, T, res, 
        mae, rmse, smape, spectral_norm, 
        time_avg, p
    )

# --- 2. Main Execution Block ---
if __name__ == "__main__":
    # Configuration
    SEED = 42
    n_trials = 1
    ms = range(2, 5, 1)
    n_trains = range(100, 501, 100)
    settings = [
        {"style": "VAR", "flawed": True, "symmetric": False},
        # {"style": "VAR", "flawed": False, "symmetric": False},
        # {"style": "MAR", "flawed": False, "symmetric": False},
        # {"style": "MAR", "flawed": False, "symmetric": True}
    ]
    
    for setting in settings:
        style = setting["style"]
        flawed = setting["flawed"]
        symmetric = setting["symmetric"]
        print(f"\nRunning experiments for style={style}, flawed={flawed}, symmetric={symmetric}\n")
        
        # Define estimators to test
        estimators = {
            "MAR(1) Yule-Walker": MAR_Estimator(mode='yw'),
            "MAR(1) Burg": MAR_Estimator(mode='burg'),
            "MAR(1) LSE": MAR_Estimator(mode='lse'),
            "VAR(1) Yule-Walker": VAR_Estimator(mode='yw'),
            "VAR(1) Burg": VAR_Estimator(mode='burg'),
            "vec MAR(1) Yule-Walker": MAR_Estimator(mode='vec_yw'),
            "vec MAR(1) Burg": MAR_Estimator(mode='vec_burg'),
        }

        # Prepare the list of tasks
        # We flatten the nested loops into a single list of arguments
        tasks = []
        for name, estimator in estimators.items():
            for m in ms:
                for T in n_trains:
                    # Pack all arguments into a tuple
                    tasks.append((name, estimator, m, T, n_trials, SEED, style, flawed, symmetric))

        print(f"Starting {len(tasks)} experiments across {multiprocessing.cpu_count()} cores...")
        print(
            f"{'Estimator':<25} | {'m':<4} | {'Dim':<4} | {'T':<4} | ",
            f"{'MAE':<9} | {'RMSE':<8} | {'SMAPE':<8} | {'Spectral Norm':<8} | ",
            f"{'p-value':<8} | {'Time (s)'}"
        )
        print("-" * 124)

        results_storage = {}
        
        # Use Pool to execute in parallel
        with multiprocessing.Pool() as pool:
            for result in pool.imap_unordered(run_single_trial, tasks):
                name, m, T, res_dict, mae, rmse, smape, spectral_norm, t, p = result
                
                # Print progress as it completes
                print(
                    f"{name:<25} | {m:<4} | {m**2:<4} | {T:<4} | "
                    f"{mae:<8.4f} | {rmse:<8.4f} | {smape:<10.4f} | ",
                    f"{spectral_norm:<12.4f} | {p:<8.4f} | {t:.4f}"
                )
                
                # Store full results for DataFrame creation
                results_storage[(name, m, T)] = res_dict

        # --- 3. Save Results ---
        print("\nProcessing results into CSV...")
        dfs = []
        for (name, m, T), res in results_storage.items():
            df = pd.DataFrame({
                "Estimator": name,
                "m": m,
                "Dim": m**2,
                "T": T,
                "Std": res["Std"],
                "MAE": res["MAE"],
                "RMSE": res["RMSE"],
                "SMAPE": res["SMAPE"],
                "Spectral Norm": res["Spectral Norm"],
                "Time Execution": res["Time"],
                "p-values": res["p-values"]
            })
            dfs.append(df)

        final_df = pd.concat(dfs, ignore_index=True)
        # Sort to make the CSV readable since parallel execution scrambled the order
        final_df = final_df.sort_values(by=['Estimator', 'm', 'T'])
        appendix = "_flawed" if flawed else ""
        appendix += "_symmetric" if symmetric else ""
        final_df.to_csv(f"results/experiment_results_{style}{appendix}.csv", index=False)
        print("Done. Saved to experiment_results.csv")