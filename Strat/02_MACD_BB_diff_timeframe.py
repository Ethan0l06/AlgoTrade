import pandas as pd
import numpy as np
from typing import Dict, Callable
import optuna

# --- All Project Imports ---
from AlgoTrade.Utils.DataManager import DataManager
from AlgoTrade.Config.BacktestConfig import BacktestConfig
from AlgoTrade.Config.Enums import TradingMode
from AlgoTrade.Strat.Class.BaseStrategy import BaseStrategy
from AlgoTrade.Factories.IndicatorFactory import IndicatorFactory
from AlgoTrade.Sizing.AtrBandsSizer import AtrBandsSizer
from AlgoTrade.Optimizer.StudyRunner import run_optimization


class MacdBBStrategy(BaseStrategy):
    """
    A strategy that uses MACD on a higher timeframe (HTF) to determine the trend
    and Bollinger Bands on a lower timeframe (LTF) for entry signals.

    Signal Logic:
    - HTF (1hr): Trend is bullish if MACD line > Signal line. Bearish if MACD < Signal.
    - LTF (15m):
        - Buy: If HTF trend is bullish and price crosses below the lower Bollinger Band.
        - Sell: If HTF trend is bearish and price crosses above the upper Bollinger Band.
    """

    def __init__(
        self,
        config: BacktestConfig,
        symbol_ltf: str,
        tframe_ltf: str,
        symbol_htf: str,
        tframe_htf: str,
        start_date: str,
        params: Dict = None,
    ):
        super().__init__(config)
        self.dm = DataManager(name="bitget")
        self.symbol_ltf = symbol_ltf
        self.tframe_ltf = tframe_ltf
        self.symbol_htf = symbol_htf
        self.tframe_htf = tframe_htf
        self.start_date = start_date

        # Default or user-provided parameters
        self.params = params or {
            "bb_period": 20,
            "bb_std_dev": 2.0,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
        }

    def generate_signals(self) -> pd.DataFrame:
        # 1. Load Data
        ltf_data = self.dm.from_local(self.symbol_ltf, self.tframe_ltf, self.start_date)
        htf_data = self.dm.from_local(self.symbol_htf, self.tframe_htf, self.start_date)

        # 2. Calculate Indicators
        # LTF (15m) Data - Bollinger Bands
        ltf_factory = IndicatorFactory(ltf_data)
        df_ltf = (
            ltf_factory.add_bollinger_bands(
                period=self.params["bb_period"],
                nbdevup=self.params["bb_std_dev"],
                nbdevdn=self.params["bb_std_dev"],
            )
            .add_macd(
                fastperiod=self.params["macd_fast"],
                slowperiod=self.params["macd_slow"],
                signalperiod=self.params["macd_signal"],
            )
            .get_data()
        )

        # HTF (1h) Data - MACD for trend
        htf_factory = IndicatorFactory(htf_data)
        df_htf = htf_factory.add_macd(
            fastperiod=self.params["macd_fast"],
            slowperiod=self.params["macd_slow"],
            signalperiod=self.params["macd_signal"],
        ).get_data()

        # 3. Define Trend from HTF data
        df_htf["trend"] = np.where(
            (df_htf["MACD"] > df_htf["MACD_signal"]) & (df_htf["MACD"] > 0),
            1,
            np.where(
                (df_htf["MACD"] < df_htf["MACD_signal"]) & (df_htf["MACD"] < 0), -1, 0
            ),
        )
        df_htf["trend"] = pd.Series(df_htf["trend"]).shift(1).fillna(0)

        # 4. Merge dataframes
        df_htf_resampled = df_htf.reindex(df_ltf.index, method="ffill")
        merged = df_ltf.join(df_htf_resampled, rsuffix="_htf").dropna()

        # 5. Generate Signals
        bb_period = self.params["bb_period"]
        # 1. MACD crosses OVER its signal line
        macd_crossover = (merged["MACD"] > merged["MACD_signal"]) & \
                        (merged["MACD"].shift(1) <= merged["MACD_signal"].shift(1))

        # 2. MACD crosses BELOW its signal line
        macd_cross_below = (merged["MACD"] < merged["MACD_signal"]) & \
                        (merged["MACD"].shift(1) >= merged["MACD_signal"].shift(1))
        buy_signal = (
            (merged["trend"] == 1)
            & (merged["close"] > merged[f"BB_UPPER_{bb_period}"])
            & (macd_crossover)
        )
        sell_signal = (
            (merged["trend"] == -1)
            & (merged["close"] < merged[f"BB_LOWER_{bb_period}"])
            & (macd_cross_below)
        )

        signal = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))

        merged["signal"] = pd.Series(signal, index=merged.index).shift(1).fillna(0)
        # if signal is 0, check macd
        return merged


def create_optimizer_strategy_function(
    trial: optuna.trial.Trial, data_dict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    This function is a wrapper for Optuna. It defines the search space
    and generates signals based on the trial parameters.
    """
    # 1. Define Search Space for Parameters
    params = {
        "bb_period": trial.suggest_int("bb_period", 10, 50),
        "bb_std_dev": trial.suggest_float("bb_std_dev", 1.5, 3.5),
        "macd_fast": trial.suggest_int("macd_fast", 5, 20),
        "macd_slow": trial.suggest_int("macd_slow", 21, 60),
        "macd_signal": trial.suggest_int("macd_signal", 5, 20),
    }

    # 2. Extract data from the dictionary
    ltf_data = data_dict["ltf_data"]
    htf_data = data_dict["htf_data"]

    # 3. Generate signals using the same logic as the main class
    ltf_factory = IndicatorFactory(ltf_data.copy())
    df_ltf = ltf_factory.add_bollinger_bands(
        period=params["bb_period"],
        nbdevup=params["bb_std_dev"],
        nbdevdn=params["bb_std_dev"],
    ).get_data()

    htf_factory = IndicatorFactory(htf_data.copy())
    df_htf = htf_factory.add_macd(
        fastperiod=params["macd_fast"],
        slowperiod=params["macd_slow"],
        signalperiod=params["macd_signal"],
    ).get_data()

    df_htf["trend"] = np.where(df_htf["MACD"] > df_htf["MACD_signal"], 1, -1)
    df_htf["trend"] = pd.Series(df_htf["trend"]).shift(1).fillna(0)

    df_htf_resampled = df_htf.reindex(df_ltf.index, method="ffill")
    merged = df_ltf.join(df_htf_resampled, rsuffix="_htf").dropna()

    bb_period = params["bb_period"]
    buy_signal = (merged["trend"] == 1) & (
        merged["close"] < merged[f"BB_LOWER_{bb_period}"]
    )
    sell_signal = (merged["trend"] == -1) & (
        merged["close"] > merged[f"BB_UPPER_{bb_period}"]
    )

    signal = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))
    merged["signal"] = pd.Series(signal, index=merged.index).shift(1).fillna(0)

    return merged


def main():
    # --- Part 1: Run Comparative Analysis ---
    if True:
        print("\n" + "=" * 50 + "\n--- Starting Comparative Analysis ---\n" + "=" * 50)
        config_for_comparison = BacktestConfig(
            initial_balance=30.0,
            leverage=10,
            trading_mode=TradingMode.CROSS,
            enable_trailing_stop=True,
            breakeven_trigger_pct=0.05,  # e.g., 0.005 for 0.5% profit
            breakeven_sl_pct=0.001,  # Move SL to 0.1% profit
            midpoint_trigger_pct=0.5,  # e.g., 0.5 for 50% to TP
            midpoint_tp_extension_pct=0.5,  # e.g., 0.5 to extend TP by 50%
            midpoint_sl_adjustment_pct=0.3,  # e.g., 0.3 to lock in 30% of profit
        )

        strategy_comparison = MacdBBStrategy(
            config=config_for_comparison,
            symbol_ltf="ADA/USDT:USDT",
            tframe_ltf="15m",
            symbol_htf="ADA/USDT:USDT",
            tframe_htf="1h",
            start_date="2024-01-01",
        )
        strategy_comparison.run_comparative()

    # --- Part 2: Run Single ---
    if False:
        print("\n" + "=" * 50 + "\n--- Starting Single Backtest ---\n" + "=" * 50)
        leverage = 10
        sizer = AtrBandsSizer(
            risk_pct=0.01,
            atr_multiplier=2.0,
            risk_reward_ratio=1.5,
            leverage=leverage,
        )
        config_for_single = BacktestConfig(
            initial_balance=30.0,
            leverage=leverage,
            trading_mode=TradingMode.CROSS,
            sizing_strategy=sizer,
            enable_trailing_stop=True,
            breakeven_trigger_pct=0.05,  # e.g., 0.005 for 0.5% profit
            breakeven_sl_pct=0.001,  # Move SL to 0.1% profit
            midpoint_trigger_pct=0.5,  # e.g., 0.5 for 50% to TP
            midpoint_tp_extension_pct=0.5,  # e.g., 0.5 to extend TP by 50%
            midpoint_sl_adjustment_pct=0.3,  # e.g., 0.3 to lock in 30% of profit
        )
        strategy_single = MacdBBStrategy(
            config=config_for_single,
            symbol_ltf="ADA/USDT:USDT",
            tframe_ltf="15m",
            symbol_htf="ADA/USDT:USDT",
            tframe_htf="1h",
            start_date="2024-01-01",
        )
        analysis = strategy_single.run_single(generate_quantstats_report=True)
        analysis.results_df.to_csv(
            "D:/ComputerScience/Trading/Quant2/optimizer_studies/results_single.csv"
        )

    if False:
        # --- Part 2: Run Optimizer ---
        print(
            "\n" + "=" * 50 + "\n--- Starting Parameter Optimization ---\n" + "=" * 50
        )
        # Use a specific sizer for optimization runs
        atr_sizer_for_opt = AtrBandsSizer(
            risk_pct=0.02, atr_multiplier=2.0, risk_reward_ratio=1.5, leverage=10
        )
        config_for_optimizer = BacktestConfig(
            initial_balance=3000.0,
            leverage=10,
            trading_mode=TradingMode.CROSS,
            sizing_strategy=atr_sizer_for_opt,
        )

        # Load data once for the optimizer to pass to all trials
        dm = DataManager(name="bitget")
        data_for_opt = {
            "ltf_data": dm.from_local("ADA/USDT:USDT", "15m", "2024-01-01"),
            "htf_data": dm.from_local("ADA/USDT:USDT", "1h", "2024-01-01"),
        }

        # Run the optimization
        study = run_optimization(
            data_dict=data_for_opt,
            config=config_for_optimizer,
            strategy_function=create_optimizer_strategy_function,
            n_trials=10,  # Number of trials for the optimizer
            metric_to_optimize="sharpe_ratio",
        )

        # You can now analyze the 'study' object for best parameters, etc.
        print(f"\nBest Sharpe Ratio: {study.best_value}")
        print(f"Best Parameters: {study.best_params}")


if __name__ == "__main__":
    main()
