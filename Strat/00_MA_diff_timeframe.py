from typing_extensions import Literal
import pandas as pd
import numpy as np
import talib as ta

# --- Our Imports ---
from AlgoTrade.Utils.DataManager import DataManager
from AlgoTrade.Config.BacktestConfig import BacktestConfig
from AlgoTrade.Backtester.BacktestRunner import BacktestRunner
from AlgoTrade.Optimizer.StudyRunner import run_optimization


def pre_process_data(df1: pd.DataFrame, df2: pd.DataFrame):
    df1 = df1.copy()
    df1["YYYYMMDD"] = df1.index.strftime("%Y%m%d")
    df1["datetime"] = df1.index
    df2 = df2.copy()
    df2["YYYYMMDD"] = df2.index.strftime("%Y%m%d")
    df2["datetime"] = df2.index
    print("Data preprocessed.")
    return df1, df2


def compute_indicators_df1(
    df1: pd.DataFrame,
    period_df1: int,
):
    df1["MA"] = ta.SMA(df1["close"], timeperiod=period_df1)
    df1["STDDEV"] = ta.STDDEV(df1["close"], timeperiod=14)
    df1["ATR"] = ta.ATR(df1["high"], df1["low"], df1["close"], timeperiod=14)
    df1["SD_ATR_Spread"] = df1["STDDEV"] - df1["ATR"]
    print("Indicators computed (df1).")
    return df1


def compute_indicators_df2(df2: pd.DataFrame, period_df2: int):
    df2["MA"] = ta.SMA(df2["close"], timeperiod=period_df2)
    df2["STDDEV"] = ta.STDDEV(df2["close"], timeperiod=14)
    df2["ATR"] = ta.ATR(df2["high"], df2["low"], df2["close"], timeperiod=14)
    df2["SD_ATR_Spread"] = df2["STDDEV"] - df2["ATR"]
    print("Indicators computed (df2).")
    return df2


def merge_data(df1: pd.DataFrame, df2: pd.DataFrame):
    # Merge with suffixes to clearly identify which columns come from which timeframe
    merged_df = pd.merge(
        df1,
        df2,
        on="YYYYMMDD",
        how="inner",
        suffixes=("_small", "_large"),  # More descriptive suffixes
    )
    merged_df.dropna(inplace=True)
    merged_df = merged_df.rename(
        columns={
            "datetime_small": "datetime",
            "close_small": "close",
            "open_small": "open",
            "high_small": "high",
            "low_small": "low",
            "volume_small": "volume",
        }
    )
    # Set index to datetime instead of date to maintain timestamp information
    merged_df.set_index("datetime", inplace=True)
    print("Data merged.")
    return merged_df


def compute_signals(merged_df: pd.DataFrame):
    merged_df["signal"] = 0
    # Convert numpy array to pandas Series before shifting
    trend = np.where(merged_df["close_large"] > merged_df["MA_large"], 1, -1)
    merged_df["trend"] = pd.Series(trend).shift(1).fillna(0)

    # Create signal array
    signal = np.where(
        (merged_df["trend"] == 1)
        & (merged_df["close"] > merged_df["MA_small"])
        & (merged_df["SD_ATR_Spread_small"] > 0),
        1,
        np.where(
            (merged_df["trend"] == -1)
            & (merged_df["close"] < merged_df["MA_small"])
            & (merged_df["SD_ATR_Spread_small"] > 0),
            -1,
            0,
        ),
    )
    # Convert signal array to pandas Series before shifting
    merged_df["signal"] = pd.Series(signal, index=merged_df.index).shift(1).fillna(0)
    print("Signal computed.")
    return merged_df


def run_single_backtest(data: pd.DataFrame, enable_plot: bool = False):
    print("\n--- Running Single Backtest ---")

    # Create a configuration instance
    config = BacktestConfig(
        initial_balance=1000.0,
        trading_mode="Cross",
        leverage=10,
        position_sizing_method="AtrVolatility",
        risk_per_trade_pct=0.02,
        atr_period=14,
        atr_multiplier=2.5,
        exit_on_signal_0=False,
    )

    # 1. Run the backtest
    runner = BacktestRunner(config=config, data=data)
    analysis = runner.run()  # The run method now returns the analysis object
    if enable_plot:
        analysis.plot_all()
    # 2. Print metrics and plot equity
    analysis.print_metrics()
    return analysis


def run_optimizer(data: pd.DataFrame):
    print("\n--- Running Optimization Study ---")
    study = run_optimization(data, n_trials=100)

    print("\n--- Optimization Finished ---")
    print(f"Best Sharpe Ratio: {study.best_value:.2f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")


def test_strategy(data: pd.DataFrame):
    folder_path = "D:/ComputerScience/Trading/Quant2/AlgoTrade/Strat/strat_check"
    data.to_csv(f"{folder_path}/data.csv")
    print(f"Data saved to {folder_path}/data.csv")


def main(plot: bool = False):
    MODE: Literal["single", "optimize", "test-strategy"] = "test-strategy"

    dm = DataManager(name="bitget")
    ohlcv_data_s = dm.from_local("ADA/USDT:USDT", "15m", "2024-12-01")
    ohlcv_data_l = dm.from_local("ADA/USDT:USDT", "4h", "2025-01-01")
    ohlcv_data_s, ohlcv_data_l = pre_process_data(ohlcv_data_s, ohlcv_data_l)
    ohlcv_data_s, ohlcv_data_l = compute_indicators(ohlcv_data_s, ohlcv_data_l, 55, 3)
    ohlcv_data = merge_data(ohlcv_data_s, ohlcv_data_l)
    ohlcv_data = compute_signals(ohlcv_data)

    if MODE == "single":
        analysis = run_single_backtest(ohlcv_data, False)
    elif MODE == "optimize":
        run_optimizer(ohlcv_data)
    elif MODE == "test-strategy":
        test_strategy(ohlcv_data)
