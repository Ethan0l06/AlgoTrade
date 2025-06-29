import pandas as pd
import numpy as np
from typing import Dict
import optuna

# --- All Project Imports ---
from AlgoTrade.Utils.DataManager import DataManager
from AlgoTrade.Config.BacktestConfig import BacktestConfig
from AlgoTrade.Config.Enums import TradingMode
from AlgoTrade.Strat.Class.BaseStrategy import BaseStrategy
from AlgoTrade.Factories.IndicatorFactory import IndicatorFactory
from AlgoTrade.Sizing.AtrBandsSizer import AtrBandsSizer
from AlgoTrade.Optimizer.StudyRunner import run_optimization
from AlgoTrade.Analysis.BacktestAnalysis import BacktestAnalysis

class MacdBBStrategy(BaseStrategy):
    """
    A strategy that uses MACD on a higher timeframe (HTF) to determine the trend
    and Bollinger Bands on a lower timeframe (LTF) for entry signals.
    The signal logic can be switched between mean-reversion and trend-following.
    """

    def __init__(
        self,
        config: BacktestConfig,
        symbol_ltf: str,
        tframe_ltf: str,
        symbol_htf: str,
        tframe_htf: str,
        start_date: str,
        params: Dict,
    ):
        super().__init__(config)
        self.dm = DataManager(name="bitget")
        self.symbol_ltf = symbol_ltf
        self.tframe_ltf = tframe_ltf
        self.symbol_htf = symbol_htf
        self.tframe_htf = tframe_htf
        self.start_date = start_date
        self.params = params

    def generate_signals(self) -> pd.DataFrame:
        # 1. Load Data
        ltf_data = self.dm.from_local(self.symbol_ltf, self.tframe_ltf, self.start_date)
        htf_data = self.dm.from_local(self.symbol_htf, self.tframe_htf, self.start_date)

        # 2. Calculate Indicators
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

        htf_factory = IndicatorFactory(htf_data)
        df_htf = htf_factory.add_macd(
            fastperiod=self.params["macd_fast"],
            slowperiod=self.params["macd_slow"],
            signalperiod=self.params["macd_signal"],
        ).get_data()

        # 3. Define Trend from HTF data
        df_htf["trend"] = np.where(
            (df_htf["MACD"] > df_htf["MACD_signal"]),
            1,
            np.where((df_htf["MACD"] < df_htf["MACD_signal"]), -1, 0),
        )
        df_htf["trend"] = pd.Series(df_htf["trend"]).shift(1).fillna(0)

        # 4. Merge dataframes
        df_htf_resampled = df_htf.reindex(df_ltf.index, method="ffill")
        merged = df_ltf.join(df_htf_resampled, rsuffix="_htf").dropna()

        # 5. Generate Signals based on the chosen mode
        bb_period = self.params["bb_period"]
        signal_mode = self.params.get(
            "signal_mode", "mean_reversion"
        )  # Default to mean_reversion

        if signal_mode == "mean_reversion":
            buy_signal = (merged["trend"] == 1) & (
                merged["close"] < merged[f"BB_LOWER_{bb_period}"]
            )
            sell_signal = (merged["trend"] == -1) & (
                merged["close"] > merged[f"BB_UPPER_{bb_period}"]
            )
        elif signal_mode == "trend_following":
            buy_signal = (merged["trend"] == 1) & (
                merged["close"] > merged[f"BB_UPPER_{bb_period}"]
            )
            sell_signal = (merged["trend"] == -1) & (
                merged["close"] < merged[f"BB_LOWER_{bb_period}"]
            )
        else:
            buy_signal, sell_signal = pd.Series(False, index=merged.index), pd.Series(
                False, index=merged.index
            )

        signal = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))
        merged["signal"] = pd.Series(signal, index=merged.index).shift(1).fillna(0)
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
        # "signal_mode": trial.suggest_categorical(
        #     "signal_mode", ["mean_reversion", "trend_following"]
        # ),
    }

    # 2. This function re-uses the main strategy's logic
    # It requires a dummy config and instance to call generate_signals
    dummy_config = BacktestConfig()
    strategy_instance = MacdBBStrategy(
        config=dummy_config,
        symbol_ltf="ADA/USDT:USDT",
        tframe_ltf="15m",
        symbol_htf="ADA/USDT:USDT",
        tframe_htf="1h",
        start_date="2024-01-01",
        params=params,
    )
    # The data is already loaded and passed in via data_dict, so we manually assign it
    strategy_instance.dm.from_local = lambda symbol, timeframe, start_date: data_dict[
        "ltf_data" if timeframe == "15m" else "htf_data"
    ]

    return strategy_instance.generate_signals()


def main():
    # --- Control Panel ---
    RUN_COMPARATIVE_ANALYSIS = False
    RUN_SINGLE_BACKTEST = True
    RUN_OPTIMIZATION = False

    # --- Common Strategy Parameters ---
    symbol_ltf = "TRX/USDT:USDT"
    tframe_ltf = "15m"
    symbol_htf = "TRX/USDT:USDT"
    tframe_htf = "1h"
    start_date = "2024-01-01"

    # --- Part 1: Run Comparative Analysis ---
    if RUN_COMPARATIVE_ANALYSIS:
        print("\n" + "=" * 50 + "\n--- Starting Comparative Analysis ---\n" + "=" * 50)
        config_for_comparison = BacktestConfig(
            initial_balance=30.0,
            leverage=10,
            trading_mode=TradingMode.CROSS,
        )
        # Using default params for comparison run
        default_params = {
            "bb_period": 20,
            "bb_std_dev": 2.0,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "signal_mode": "mean_reversion",
        }
        strategy_comparison = MacdBBStrategy(
            config=config_for_comparison,
            symbol_ltf=symbol_ltf,
            tframe_ltf=tframe_ltf,
            symbol_htf=symbol_htf,
            tframe_htf=tframe_htf,
            start_date=start_date,
            params=default_params,
        )
        strategy_comparison.run_comparative()

    # --- Part 2: Run Single Backtest ---
    if RUN_SINGLE_BACKTEST:
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
            breakeven_trigger_pct=0.05,  # e.g., 0.005 for 0.5% profit
            breakeven_sl_pct=0.001,  # Move SL to 0.1% profit
            midpoint_trigger_pct=0.5,  # e.g., 0.5 for 50% to TP
            midpoint_tp_extension_pct=0.5,  # e.g., 0.5 to extend TP by 50%
            midpoint_sl_adjustment_pct=0.3,  # e.g., 0.3 to lock in 30% of profit
        )
        # Best params from a previous optimization run
        best_params = {
            "bb_period": 14,
            "bb_std_dev": 3,
            "macd_fast": 6,
            "macd_slow": 14,
            "macd_signal": 6,
            "signal_mode": "trend_following",
        }
        strategy_single = MacdBBStrategy(
            config=config_for_single,
            symbol_ltf=symbol_ltf,
            tframe_ltf=tframe_ltf,
            symbol_htf=symbol_htf,
            tframe_htf=tframe_htf,
            start_date=start_date,
            params=best_params,
        )
        strategy_single.run_single(generate_quantstats_report=True)

    # --- Part 3: Run Optimizer ---
    if RUN_OPTIMIZATION:
        def trade_sharpe_pf_score(analysis: BacktestAnalysis) -> float:
            """
            A custom objective function that scores a backtest based on a balance of
            Sharpe Ratio, Profit Factor, and the number of trades.

            Args:
                analysis (BacktestAnalysis): The completed analysis object from a backtest run.

            Returns:
                float: The calculated score to be maximized by Optuna.
            """
            metrics = analysis.metrics

            sharpe = metrics.get("sharpe_ratio", -1.0)
            profit_factor = metrics.get("profit_factor", 0.0)
            total_trades = metrics.get("total_trades", 0)

            # --- Hard Constraints ---
            # We define a baseline for an acceptable strategy. If a trial fails
            # these checks, it's given a score of -1.0 to be discarded.
            MINIMUM_TRADES = 25
            MINIMUM_PROFIT_FACTOR = 1.1  # Must be at least 10% profitable
            MINIMUM_SHARPE_RATIO = 0.1  # Must have a positive risk-adjusted return

            if (
                total_trades < MINIMUM_TRADES
                or profit_factor < MINIMUM_PROFIT_FACTOR
                or sharpe < MINIMUM_SHARPE_RATIO
            ):
                return -1.0

            # --- Scoring Logic ---
            # The score is a product of the three metrics, with log scaling for
            # profit factor and total trades to ensure balanced contributions.
            # A positive Sharpe is the primary driver.
            # We use np.log1p (log(1+x)) to handle values gracefully.

            # We scale profit_factor so that PF=1.1 gives a small score, and it grows from there.
            pf_score = np.log1p(profit_factor - 1)

            # We scale trades so that more trades are better, but with diminishing returns.
            trades_score = np.log1p(total_trades)

            score = sharpe * pf_score * trades_score

            # Ensure the final score is a valid number
            if not np.isfinite(score):
                return -1.0

            return score
        print(
            "\n" + "=" * 50 + "\n--- Starting Parameter Optimization ---\n" + "=" * 50
        )
        atr_sizer_for_opt = AtrBandsSizer(
            risk_pct=0.01, atr_multiplier=2.0, risk_reward_ratio=1.5, leverage=10
        )
        config_for_optimizer = BacktestConfig(
            initial_balance=30.0,
            leverage=10,
            trading_mode=TradingMode.CROSS,
            sizing_strategy=atr_sizer_for_opt,
            breakeven_trigger_pct=0.05,  # e.g., 0.005 for 0.5% profit
            breakeven_sl_pct=0.001,  # Move SL to 0.1% profit
            midpoint_trigger_pct=0.5,  # e.g., 0.5 for 50% to TP
            midpoint_tp_extension_pct=0.5,  # e.g., 0.5 to extend TP by 50%
            midpoint_sl_adjustment_pct=0.3,  # e.g., 0.3 to lock in 30% of profit
        )

        dm = DataManager(name="bitget")
        data_for_opt = {
            "ltf_data": dm.from_local(symbol_ltf, tframe_ltf, start_date),
            "htf_data": dm.from_local(symbol_htf, tframe_htf, start_date),
        }

        study = run_optimization(
            data_dict=data_for_opt,
            config=config_for_optimizer,
            strategy_function=create_optimizer_strategy_function,
            n_trials=100,
            objective_function=trade_sharpe_pf_score,
        )

        print(f"\nBest Custom Score: {study.best_value}")
        print(f"Best Parameters: {study.best_params}")


if __name__ == "__main__":
    main()
