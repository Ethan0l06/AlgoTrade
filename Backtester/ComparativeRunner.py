# /TradingBacktester/comparative_run.py

import pandas as pd
import copy
from typing import Dict

# --- All Project Imports ---
# Assuming your data preparation produces the final DataFrame
# from StrategyLab.Utils.DataManager import DataManager
from StrategyLab.Config.BacktestConfig import BacktestConfig
from StrategyLab.Backtester.BacktestRunner import BacktestRunner
from StrategyLab.Analysis.BacktestAnalysis import BacktestAnalysis


def print_comparison_report(results_dict: Dict[str, BacktestAnalysis]):
    """
    Generates and prints a comprehensive ("wholesome") report comparing
    the performance of all position sizing methods.

    Args:
        results_dict (Dict[str, BacktestAnalysis]): A dictionary where keys are
                                                    method names and values are the
                                                    completed analysis objects.
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "Comprehensive Backtest Comparison Report")
    print("=" * 80)
    print(
        f"Period: [{results_dict["PercentBalance"].results_df.index[0].date()}] -> [{results_dict["PercentBalance"].results_df.index[-1].date()}]"
    )
    report_data = []

    # Define the order and formatting for metrics
    metrics_to_display = {
        "initial_balance": "${:,.2f}",
        "final_balance": "${:,.2f}",
        "roi_pct": "{:.2f}%",
        "total_profits": "${:,.2f}",
        "total_losses": "${:,.2f}",
        "sharpe_ratio": "{:.3f}",
        "sortino_ratio": "{:.3f}",
        "calmar_ratio": "{:.3f}",
        "max_drawdown_equity": "{:.2%}",
        "total_trades": "{:d}",
        "total_good_trades": "{:d}",
        "total_bad_trades": "{:d}",
        "avg_pnl_pct_good_trades": "{:.3f}%",
        "avg_pnl_pct_bad_trades": "{:.3f}%",
        "global_win_rate": "{:.2%}",
        "profit_factor": "{:.2f}",
    }

    for method, analysis in results_dict.items():
        if not analysis.metrics:
            # Handle cases where a method resulted in no trades
            row = {"Method": method, **{metric: "N/A" for metric in metrics_to_display}}
        else:
            row = {"Method": method}
            for metric, fmt in metrics_to_display.items():
                value = analysis.metrics.get(
                    metric, 0
                )  # Default to 0 if metric is missing
                row[metric] = fmt.format(value) if pd.notna(value) else "N/A"

        report_data.append(row)

    if not report_data:
        print("No results to display.")
        return

    # Create and print the DataFrame
    report_df = pd.DataFrame(report_data).set_index("Method")

    # Transpose for better readability if there are many metrics
    print(report_df.T)
    print("=" * 80)


def run_comparative_analysis(
    base_config: BacktestConfig, data: pd.DataFrame
) -> Dict[str, BacktestAnalysis]:
    """
    Orchestrates multiple backtest runs, one for each position sizing method.

    Args:
        base_config (BacktestConfig): The base configuration with common settings.
        data (pd.DataFrame): The prepared DataFrame with signals.

    Returns:
        Dict[str, BacktestAnalysis]: A dictionary containing the analysis object for each method.
    """
    methods_to_compare = [
        "PercentBalance",
        "FixedAmount",
        "AtrVolatility",
        "KellyCriterion",
        "AtrBands",
    ]

    all_results = {}

    for method in methods_to_compare:
        print("\n" + "-" * 50)
        print(f"--- Running Backtest for: {method} ---")
        print("-" * 50)

        # Create a deep copy to ensure settings don't leak between runs
        current_config = copy.deepcopy(base_config)
        current_config.position_sizing_method = method

        # Instantiate and run the backtester
        runner = BacktestRunner(config=current_config, data=data)
        analysis = runner.run()

        # Store the completed analysis object
        all_results[method] = analysis

    return all_results
