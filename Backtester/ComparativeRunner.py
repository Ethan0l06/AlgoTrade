import pandas as pd
import copy
from typing import Dict, List
from AlgoTrade.Config.BacktestConfig import BacktestConfig
from AlgoTrade.Sizing.BaseSizer import BaseSizer
from AlgoTrade.Sizing.FixedAmountSizer import FixedAmountSizer
from AlgoTrade.Sizing.PercentBalanceSizer import PercentBalanceSizer
from AlgoTrade.Sizing.AtrBandsSizer import AtrBandsSizer

# Import other sizers as they are created
from AlgoTrade.Backtester.BacktestRunner import BacktestRunner
from AlgoTrade.Analysis.BacktestAnalysis import BacktestAnalysis


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
        f"Period: [{results_dict["PercentBalanceSizer"].results_df.index[0].date()}] -> [{results_dict["PercentBalanceSizer"].results_df.index[-1].date()}]"
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

    # Define the list of strategy objects to compare
    sizers_to_compare: List[BaseSizer] = [
        PercentBalanceSizer(percent=0.1),  # 10% of balance
        FixedAmountSizer(amount=10.0),  # $100 margin
        AtrBandsSizer(
            risk_pct=0.01,
            atr_multiplier=2.0,
            risk_reward_ratio=1.5,
            leverage=base_config.leverage,
        ),
    ]

    all_results = {}

    for sizer in sizers_to_compare:
        method_name = sizer.__class__.__name__
        print(f"\n--- Running Backtest for: {method_name} ---")

        current_config = copy.deepcopy(base_config)
        current_config.sizing_strategy = sizer  # Assign the sizer object

        runner = BacktestRunner(config=current_config, data=data)
        analysis = runner.run()
        all_results[method_name] = analysis

    return all_results
