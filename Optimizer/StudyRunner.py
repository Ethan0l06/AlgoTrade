# /StrategyLab/Optimizer/StudyRunner.py

import optuna
import pandas as pd
import multiprocessing
import math
from typing import Dict, Any, Callable

# --- All Project Imports ---
from StrategyLab.Config.BacktestConfig import BacktestConfig
from StrategyLab.Backtester.BacktestRunner import BacktestRunner

n_cpus = multiprocessing.cpu_count()
n_jobs = max(1, math.floor(n_cpus * 0.6))

def run_optimization(
    data_dict: Dict[str, pd.DataFrame],
    config: BacktestConfig,
    strategy_function: Callable,
    n_trials: int,
    metric_to_optimize: str = "sharpe_ratio",
) -> optuna.study.Study:
    """
    Runs an Optuna optimization study on a given strategy function.

    Args:
        data_dict (Dict[str, pd.DataFrame]): Dictionary containing the raw data,
                                             e.g., {"ltf_data": df1, "htf_data": df2}.
        config (BacktestConfig): A fixed configuration object for the backtest.
        strategy_function (Callable): The function that defines parameters and
                                      generates signals. It must accept a trial
                                      and the data_dict as arguments.
        n_trials (int): The number of optimization trials to run.
        metric_to_optimize (str): The metric from the analysis results to maximize.
                                  e.g., "sharpe_ratio", "roi_pct", "profit_factor".

    Returns:
        optuna.study.Study: The completed Optuna study object.
    """

    def objective(trial: optuna.trial.Trial) -> float:
        try:
            # 1. Generate the data with signals using the strategy function
            #    which contains the trial's parameters.
            signal_data = strategy_function(trial, data_dict)

            # 2. Run the backtest with the generated data and the FIXED config
            runner = BacktestRunner(config=config, data=signal_data)
            analysis = runner.run()

            # 3. Get the performance metric
            metric = analysis.metrics.get(metric_to_optimize, -1.0)

            # Ensure we return a valid float (handle NaN or None)
            return metric if pd.notna(metric) else -1.0

        except Exception as e:
            # Print the error to see what went wrong during a trial
            print(f"Trial failed with error: {e}")
            # Returning -1.0 ensures that failed trials are scored poorly
            return -1.0

    # Create the study and optimize
    # Using a pruner helps to cut unpromising trials early.
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(
        objective, n_trials=n_trials, n_jobs=n_jobs
    )  # n_jobs=-1 uses all CPU cores

    print("\n--- Optimization Finished ---")
    print(f"Best Metric ({metric_to_optimize}): {study.best_value}")
    for key, value in study.best_params.items():
        if key != metric_to_optimize:
            print(f"{key}: {value}")
    return study
