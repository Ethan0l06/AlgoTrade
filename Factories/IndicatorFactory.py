# AlgoTrade/Factories/IndicatorFactory.py

import pandas as pd
import talib as ta
from typing import Self


class IndicatorFactory:
    """
    A factory class for computing technical indicators on a DataFrame.
    Uses a fluent, chainable interface.
    """

    def __init__(self, data: pd.DataFrame, inplace: bool = False):
        """
        Initializes the factory with the OHLCV data.

        Args:
            data (pd.DataFrame): The input DataFrame with 'open', 'high', 'low', 'close' columns.
            inplace (bool): If True, modifies the original DataFrame.
                            If False (default), works on a copy.
        """
        self._df = data if inplace else data.copy()

    def add_sma(self, period: int, column: str = "close") -> Self:
        """Adds a Simple Moving Average (SMA) column."""
        self._df[f"SMA_{period}"] = ta.SMA(self._df[column], timeperiod=period)
        return self

    def add_ema(self, period: int, column: str = "close") -> Self:
        """Adds an Exponential Moving Average (EMA) column."""
        self._df[f"EMA_{period}"] = ta.EMA(self._df[column], timeperiod=period)
        return self

    def add_atr(self, period: int) -> Self:
        """Adds the Average True Range (ATR) indicator."""
        self._df[f"ATR_{period}"] = ta.ATR(
            self._df["high"], self._df["low"], self._df["close"], timeperiod=period
        )
        return self

    def add_stddev(self, period: int, column: str = "close") -> Self:
        """Adds the Standard Deviation indicator."""
        self._df[f"STDDEV_{period}"] = ta.STDDEV(self._df[column], timeperiod=period)
        return self

    def add_bollinger_bands(
        self,
        period: int,
        nbdevup: float = 2.0,
        nbdevdn: float = 2.0,
        column: str = "close",
    ) -> Self:
        """Adds Bollinger Bands (BBANDS) indicator."""
        upper, middle, lower = ta.BBANDS(
            self._df[column],
            timeperiod=period,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn,
            matype=0,  # SMA
        )
        self._df[f"BB_UPPER_{period}"] = upper
        self._df[f"BB_MIDDLE_{period}"] = middle
        self._df[f"BB_LOWER_{period}"] = lower
        return self

    def add_macd(
        self,
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9,
        column: str = "close",
    ) -> Self:
        """Adds Moving Average Convergence Divergence (MACD) indicators."""
        macd, macdsignal, macdhist = ta.MACD(
            self._df[column],
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod,
        )
        self._df["MACD"] = macd
        self._df["MACD_signal"] = macdsignal
        self._df["MACD_hist"] = macdhist
        return self

    def add_rsi(self, period: int, column: str = "close") -> Self:
        """Adds the Relative Strength Index (RSI) indicator."""
        self._df[f"RSI_{period}"] = ta.RSI(self._df[column], timeperiod=period)
        return self

    def add_volume_sma(self, period: int, column: str = "volume") -> Self:
        """Adds a Simple Moving Average (SMA) column for volume."""
        self._df[f"VOLUME_SMA_{period}"] = ta.SMA(self._df[column], timeperiod=period)
        return self
    
    def get_data(self) -> pd.DataFrame:
        """
        Returns the final DataFrame with all computed indicators.
        """
        print("Getting data...")
        return self._df
