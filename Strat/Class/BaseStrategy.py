from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from AlgoTrade.Config.BacktestConfig import BacktestConfig
from AlgoTrade.Backtester.BacktestRunner import BacktestRunner
from AlgoTrade.Backtester import ComparativeRunner as cr


class BaseStrategy(ABC):
    """Enhanced abstract base class for all trading strategies."""

    def __init__(self, config: BacktestConfig):
        self.config = config

    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """
        The core logic of the strategy. This method should load data,
        compute indicators, and return a DataFrame with a 'signal' column.

        For simple strategies, you can use the helper methods:
        1. Load your data
        2. Call self.setup_indicators(df)
        3. Call self._apply_vectorized_signals(df)
        4. Return df
        """
        raise NotImplementedError("Each strategy must implement this method.")

    # === NEW ENHANCED METHODS ===

    def setup_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Override this method to add indicators to your DataFrame.
        Use IndicatorFactory or any other method you prefer.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with indicators added

        Example:
            return IndicatorFactory(df).add_sma(20).add_rsi(14).get_data()
        """
        return df  # Default: no indicators

    def generate_signal_conditions(self, df: pd.DataFrame) -> np.ndarray:
        """
        Override this method to define your signal logic using vectorized operations.

        Args:
            df: DataFrame with OHLCV data and indicators

        Returns:
            np.array with signals: 1 (long), -1 (short), 0 (no signal)

        Example:
            long_cond = (df['close'] > df['SMA_20']) & (df['RSI_14'] < 70)
            short_cond = (df['close'] < df['SMA_20']) & (df['RSI_14'] > 30)
            return VectorizedSignals.safe_signals(long_cond, short_cond)
        """
        return np.zeros(len(df))  # Default: no signals

    def configure_sizing(self):
        """
        Override this method to configure position sizing.
        The returned sizer will be applied to the config automatically.

        Returns:
            BaseSizer instance or None

        Example:
            return RiskBasedSizer(risk_percent=0.01)
        """
        return None  # Default: use config's existing sizer

    # === HELPER METHODS ===

    def _apply_vectorized_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper method that applies vectorized signal generation safely.
        Call this in your generate_signals() method after adding indicators.

        Args:
            df: DataFrame with indicators

        Returns:
            DataFrame with 'signal' column added
        """
        # Generate signals using the vectorized method
        signals = self.generate_signal_conditions(df)

        # Apply proper shift to prevent lookahead bias
        df["signal"] = pd.Series(signals, index=df.index).shift(1).fillna(0)

        return df

    def _apply_sizing_config(self):
        """
        Helper method to apply sizing configuration to the config.
        Call this before running backtest if you want to use configure_sizing().
        """
        sizer = self.configure_sizing()
        if sizer is not None:
            self.config.sizing_strategy = sizer

    # === EXISTING METHODS (UNCHANGED) ===

    def run_single(
        self,
        generate_quantstats_report: bool = False,
        apply_sizing_config: bool = True,
    ):
        """Runs a single backtest with the strategy's signals."""
        print("\\n--- Running Single Backtest ---")

        # Apply sizing configuration if requested
        if apply_sizing_config:
            self._apply_sizing_config()

        signal_df = self.generate_signals()
        runner = BacktestRunner(config=self.config, data=signal_df)
        analysis = runner.run()
        analysis.print_metrics()

        if generate_quantstats_report:
            analysis.generate_quantstats_report()

        return analysis

    def run_comparative(self, apply_sizing_config: bool = True):
        """Runs a comparative analysis across all sizing methods."""
        print("\n--- Running Comparative Analysis ---")

        # Apply sizing configuration if requested
        if apply_sizing_config:
            self._apply_sizing_config()

        signal_df = self.generate_signals()
        results = cr.run_comparative_analysis(self.config, signal_df)
        cr.print_comparison_report(results)
        return results
