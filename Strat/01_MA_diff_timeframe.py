from typing_extensions import Literal
import pandas as pd
import numpy as np
import talib as ta

# --- Our Imports ---
from AlgoTrade.Utils.DataManager import DataManager
from AlgoTrade.Config.BacktestConfig import BacktestConfig
from AlgoTrade.Backtester.BacktestRunner import BacktestRunner
from AlgoTrade.Backtester import ComparativeRunner as cr


def pre_process_data(df1: pd.DataFrame, df2: pd.DataFrame):
    df1["datetime"] = df1.index
    df1.reset_index(drop=True, inplace=True)
    df2["datetime"] = df2.index
    df2.reset_index(drop=True, inplace=True)
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
    trend = np.where(df2["close"] > df2["MA"], 1, -1)
    df2["trend"] = pd.Series(trend).shift(1).fillna(0)
    print("Indicators computed (df2).")
    return df2


def merge_htf_into_ltf(df_ltf, df_htf, suffix):
    df_ltf = df_ltf.copy()
    df_htf = df_htf.copy()

    df_ltf["datetime"] = pd.to_datetime(df_ltf["datetime"])
    df_htf["datetime"] = pd.to_datetime(df_htf["datetime"])

    df_ltf.set_index("datetime", inplace=True)
    df_htf.set_index("datetime", inplace=True)

    df_htf = df_htf.shift(1)

    df_htf_resampled = df_htf.reindex(df_ltf.index, method="ffill")
    df_merged = df_ltf.join(df_htf_resampled, rsuffix=suffix)
    df_merged.dropna(inplace=True)
    print("Starting time before merge:", df_ltf.index[0])
    print("Starting time after merge:", df_merged.index[0])
    return df_merged.reset_index()


def compute_signals(merged_df: pd.DataFrame):
    merged_df["signal"] = 0
    # Create signal array
    signal = np.where(
        (merged_df["trend"] == 1)
        & (merged_df["close"] > merged_df["MA"])
        & (merged_df["SD_ATR_Spread"] > 0),
        1,
        np.where(
            (merged_df["trend"] == -1)
            & (merged_df["close"] < merged_df["MA"])
            & (merged_df["SD_ATR_Spread"] > 0),
            -1,
            0,
        ),
    )
    # Convert signal array to pandas Series before shifting
    merged_df["signal"] = pd.Series(signal, index=merged_df.index).shift(1).fillna(0)
    print("Signal computed.")
    return merged_df


def run_single_backtest(data: pd.DataFrame):
    config = BacktestConfig(
        initial_balance=100.0,
        trading_mode="Cross",
        leverage=100,
        position_sizing_method="PercentBalance",
        percent_balance_pct=0.1,  # <-- UPDATED
        exit_on_signal_0=True,
        # General SL/TP can be used as a fallback
        stop_loss_pct=0.02,
        take_profit_pct=0.02,
    )

    runner = BacktestRunner(config=config, data=data)
    analysis = runner.run()
    analysis.print_metrics()
    return analysis


def run_comparative_analysis(data: pd.DataFrame):
    base_config = BacktestConfig(
        initial_balance=100.0,
        trading_mode="Cross",
        leverage=50,
        exit_on_signal_0=True,
        # --- Parameters for all sizing methods ---
        # For PercentBalance
        percent_balance_pct=0.1,
        # For FixedAmount
        fixed_amount_size=20.0,  # $20 margin per trade
        # For AtrVolatility
        atr_volatility_risk_pct=0.02,
        atr_volatility_period=14,
        atr_volatility_multiplier=2.5,
        # For KellyCriterion
        kelly_criterion_lookback=50,
        kelly_criterion_fraction=0.5,
        # For AtrBands
        atr_bands_risk_pct=0.02,
        atr_bands_period=14,
        atr_bands_multiplier=2.0,
        atr_bands_risk_reward_ratio=1.5,
    )

    results = cr.run_comparative_analysis(base_config, data)
    cr.print_comparison_report(results)
    return results


def main():
    # Load data
    dm = DataManager(name="bitget")
    ohlcv_data_s = dm.from_local("ADA/USDT:USDT", "15m", "2024-12-01")
    ohlcv_data_l = dm.from_local("ADA/USDT:USDT", "4h", "2025-01-01")

    # Process data
    df1, df2 = pre_process_data(ohlcv_data_s, ohlcv_data_l)
    df1 = compute_indicators_df1(df1, 55)
    df2 = compute_indicators_df2(df2, 3)

    # Merge and compute signals
    merge_df = merge_htf_into_ltf(df1, df2, suffix="_4h")
    solution = compute_signals(merge_df)
    solution.set_index('datetime', inplace=True)

    # Run backtests
    print("\n=== Running Single Backtest ===")
    analysis = run_single_backtest(solution)

    print("\n=== Running Comparative Analysis ===")
    results = run_comparative_analysis(solution)


if __name__ == "__main__":
    main() 
