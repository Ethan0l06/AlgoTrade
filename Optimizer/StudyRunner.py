# /AlgoTrade/Optimizer/StudyRunner.py

import optuna
import pandas as pd
import multiprocessing
import math
import numpy as np
from typing import Dict, Any, Callable, Union

# --- All Project Imports ---
from AlgoTrade.Config.BacktestConfig import BacktestConfig
from AlgoTrade.Backtester.BacktestRunner import BacktestRunner
from AlgoTrade.Analysis.BacktestAnalysis import BacktestAnalysis

n_cpus = multiprocessing.cpu_count()
n_jobs = max(1, math.floor(n_cpus * 0.8))


def _default_objective(analysis: BacktestAnalysis) -> float:
    """
    Default objective function that maximizes Sharpe while penalizing low trade counts.
    """
    sharpe = analysis.metrics.get("sharpe_ratio", -1.0)
    total_trades = analysis.metrics.get("total_trades", 0)

    # Penalize trials with fewer than 10 trades
    if total_trades < 10:
        return -1.0

    # Use log to reward higher trade counts with diminishing returns
    return sharpe * np.log1p(total_trades)


def run_optimization(
    data_dict: Dict[str, pd.DataFrame],
    config: BacktestConfig,
    strategy_function: Callable,
    n_trials: int,
    objective_function: Callable = _default_objective,
) -> optuna.study.Study:
    """
    Runs an Optuna optimization study on a given strategy function.

    Args:
        data_dict (Dict[str, pd.DataFrame]): Dictionary containing the raw data.
        config (BacktestConfig): A fixed configuration object for the backtest.
        strategy_function (Callable): The function that defines parameters and
                                      generates signals.
        n_trials (int): The number of optimization trials to run.
        objective_function (Callable): A function that takes a BacktestAnalysis
                                     object and returns a float score to be maximized.
                                     If None, a default score is used.

    Returns:
        optuna.study.Study: The completed Optuna study object.
    """

    def objective(trial: optuna.trial.Trial) -> float:
        try:
            # 1. Generate the data with signals using the trial's parameters.
            signal_data = strategy_function(trial, data_dict)

            if signal_data.empty:
                return -1.0  # Return poor score if no data

            # 2. Run the backtest
            runner = BacktestRunner(config=config, data=signal_data)
            analysis = runner.run()

            # 3. Calculate the objective score using the provided function
            if not analysis.metrics:
                return -1.0  # No trades were made

            return objective_function(analysis)

        except optuna.exceptions.TrialPruned:
            raise  # Re-raise the pruning exception
        except Exception as e:
            print(f"Trial failed with error: {e}")
            return -1.0  # Failed trials are scored poorly

    # Create the study and optimize
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    print("\n--- Optimization Finished ---")
    print(f"Best Objective Score: {study.best_value}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    return study
