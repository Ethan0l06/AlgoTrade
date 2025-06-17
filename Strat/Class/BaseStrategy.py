from abc import ABC, abstractmethod
import pandas as pd
from AlgoTrade.Config.BacktestConfig import BacktestConfig
from AlgoTrade.Backtester.BacktestRunner import BacktestRunner
from AlgoTrade.Backtester import ComparativeRunner as cr


class BaseStrategy(ABC):
    """An abstract base class for all trading strategies."""

    def __init__(self, config: BacktestConfig):
        self.config = config

    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """
        The core logic of the strategy. This method should load data,
        compute indicators, and return a DataFrame with a 'signal' column.
        """
        raise NotImplementedError("Each strategy must implement this method.")

    def run_single(
        self,
        plot_interactive_trades: bool = False,
        generate_quantstats_report: bool = False,
    ):
        """Runs a single backtest with the strategy's signals."""
        print("\\n--- Running Single Backtest ---")
        signal_df = self.generate_signals()
        runner = BacktestRunner(config=self.config, data=signal_df)
        analysis = runner.run()
        analysis.print_metrics()
        if plot_interactive_trades:
            analysis.plot_interactive_trades()
        if generate_quantstats_report:
            analysis.generate_quantstats_report()
        return analysis

    def run_comparative(self):
        """Runs a comparative analysis across all sizing methods."""
        print("\\n--- Running Comparative Analysis ---")
        signal_df = self.generate_signals()
        results = cr.run_comparative_analysis(self.config, signal_df)
        cr.print_comparison_report(results)
        return results
