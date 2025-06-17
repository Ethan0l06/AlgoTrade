import pandas as pd
import numpy as np
from AlgoTrade.Utils.DataManager import DataManager
from AlgoTrade.Config.BacktestConfig import BacktestConfig
from AlgoTrade.Config.Enums import TradingMode
from AlgoTrade.Strat.Class.BaseStrategy import BaseStrategy
from AlgoTrade.Factories.IndicatorFactory import IndicatorFactory
from AlgoTrade.Config.BacktestConfig import BacktestConfig
from AlgoTrade.Sizing.FixedAmountSizer import FixedAmountSizer
from AlgoTrade.Sizing.PercentBalanceSizer import PercentBalanceSizer
from AlgoTrade.Sizing.AtrBandsSizer import AtrBandsSizer


class MaDiffTimeframeStrategy(BaseStrategy):
    def __init__(
        self,
        config: BacktestConfig,
        symbol_ltf: str,
        tframe_ltf: str,
        symbol_htf: str,
        tframe_htf: str,
        start_date: str,
    ):
        super().__init__(config)
        self.dm = DataManager(name="bitget")
        self.symbol_ltf = symbol_ltf
        self.tframe_ltf = tframe_ltf
        self.symbol_htf = symbol_htf
        self.tframe_htf = tframe_htf
        self.start_date = start_date

    def generate_signals(self) -> pd.DataFrame:
        ltf_data = self.dm.from_local(self.symbol_ltf, self.tframe_ltf, self.start_date)
        htf_data = self.dm.from_local(self.symbol_htf, self.tframe_htf, self.start_date)

        ltf_factory = IndicatorFactory(ltf_data)
        df1 = ltf_factory.add_sma(90).add_stddev(14).add_atr(14).get_data()
        df1["SD_ATR_Spread"] = df1["STDDEV_14"] - df1["ATR_14"]

        htf_factory = IndicatorFactory(htf_data)
        df2 = htf_factory.add_sma(14).get_data()
        df2["trend"] = np.where(df2["close"] > df2["SMA_14"], 1, -1)
        df2["trend"] = pd.Series(df2["trend"]).shift(1).fillna(0)

        # Merge logic (simplified from original for clarity)
        df2_resampled = df2.reindex(df1.index, method="ffill")
        merged = df1.join(df2_resampled, rsuffix="_htf").dropna()

        signal = np.where(
            (merged["trend"] == 1) & (merged["close"] > merged["SMA_90"]),
            1,
            np.where(
                (merged["trend"] == -1) & (merged["close"] < merged["SMA_90"]), -1, 0
            ),
        )
        merged["signal"] = pd.Series(signal, index=merged.index).shift(1).fillna(0)
        return merged


def main():
    # ============== Fixed Amount Sizer ==============
    fixed_sizer = FixedAmountSizer(amount=10.0)
    config_fixed_amount = BacktestConfig(
        initial_balance=30.0,
        leverage=10,
        trading_mode=TradingMode.CROSS,
        sizing_strategy=fixed_sizer,  # Pass the sizer object here
    )

    # ============== ATR Bands Sizer ==============
    leverage = 10
    atr_sizer = AtrBandsSizer(
        risk_pct=0.01,  # Risk 1% of total equity per trade.
        atr_multiplier=1.0,  # Set Stop-Loss at 2x ATR away from the entry price.
        risk_reward_ratio=1.5,  # Set Take-Profit at 1.5x the Stop-Loss distance.
        leverage=leverage,  # Pass the leverage to the sizer.
    )
    config_atr_bands = BacktestConfig(
        initial_balance=3000.0,
        leverage=leverage,
        trading_mode=TradingMode.CROSS,
        sizing_strategy=atr_sizer,  # Pass the configured sizer object.
    )

    # ============== Percent Balance Sizer ==============
    percent_sizer = PercentBalanceSizer(percent=0.05)  # Use 5% of balance for margin
    config_percent_balance = BacktestConfig(
        initial_balance=30.0,
        leverage=10,
        trading_mode=TradingMode.CROSS,
        sizing_strategy=percent_sizer,  # Pass the sizer object.
    )

    # ============== Backtest Config For Comparative Runner ==============
    backtest_configs = BacktestConfig(
        initial_balance=3000.0,
        leverage=10,
        trading_mode=TradingMode.CROSS,
    )

    strategy = MaDiffTimeframeStrategy(
        config=backtest_configs,
        symbol_ltf="ADA/USDT:USDT",
        tframe_ltf="15m",
        symbol_htf="ADA/USDT:USDT",
        tframe_htf="4h",
        start_date="2024-01-01",
    )

    # strategy.run_single()
    strategy.run_comparative()


if __name__ == "__main__":
    main()
