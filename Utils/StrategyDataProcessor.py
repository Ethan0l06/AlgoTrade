import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
from AlgoTrade.Utils.DataManager import TIMEFRAMES_TYPE


class StrategyDataProcessor:
    """
    Prepares raw OHLCV data for trading strategies by adding technical indicators,
    features, and performing data cleaning and validation.
    """

    def __init__(self, sqlite_manager):
        self.sqlite_manager = sqlite_manager
        self.processed_cache = {}

    def look_available_assets(self):
        return self.sqlite_manager.get_available_symbols()

    def prepare_single_asset(
        self,
        symbol: str,
        timeframe: TIMEFRAMES_TYPE,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Prepare data for a single asset with technical indicators and features

        :param symbol: Trading pair symbol
        :param timeframe: Data timeframe
        :param start_date: Start date filter
        :param end_date: End date filter
        :param features: List of features to add ['sma', 'rsi', 'macd', etc.]
        :return: DataFrame with processed data
        """
        # Load raw data
        raw_data = self.sqlite_manager.load_ohlcv_data(
            symbol, timeframe, start_date, end_date
        )

        if raw_data.empty:
            print(f"No data found for {symbol} {timeframe}")
            return pd.DataFrame()

        print(f"Processing {symbol} {timeframe}: {len(raw_data)} records")

        # Clean and validate data
        data = self._clean_ohlcv_data(raw_data.copy())

        # Add basic features
        data = self._add_basic_features(data)

        # Clean final data
        data = self._final_cleanup(data)

        print(
            f"✅ Processed {symbol}: {len(data)} records, {len(data.columns)} features"
        )
        return data

    def _clean_ohlcv_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate OHLCV data"""
        # Remove any duplicate timestamps
        data = data[~data.index.duplicated(keep="first")]

        # Sort by index
        data = data.sort_index()

        # Check for negative or zero prices
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in data.columns:
                data[col] = data[col].replace(0, np.nan)
                data[col] = data[col].where(data[col] > 0, np.nan)

        # Check OHLC logic (high >= low, etc.)
        if all(col in data.columns for col in price_cols):
            # High should be >= Open, Close, Low
            data.loc[
                data["high"] < data[["open", "close", "low"]].max(axis=1), "high"
            ] = data[["open", "close", "low"]].max(axis=1)

            # Low should be <= Open, Close, High
            data.loc[
                data["low"] > data[["open", "close", "high"]].min(axis=1), "low"
            ] = data[["open", "close", "high"]].min(axis=1)

        # Forward fill missing values (small gaps)
        data = data.fillna(method="ffill", limit=5)

        # Remove remaining NaN rows
        data = data.dropna()

        return data

    def _add_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic price and volume features"""
        # Price features
        data["returns"] = data["close"].pct_change()
        data["log_returns"] = np.log(data["close"] / data["close"].shift(1))
        data["price_range"] = (data["high"] - data["low"]) / data["close"]
        data["body_size"] = abs(data["close"] - data["open"]) / data["close"]
        data["upper_shadow"] = (
            data["high"] - data[["open", "close"]].max(axis=1)
        ) / data["close"]
        data["lower_shadow"] = (
            data[["open", "close"]].min(axis=1) - data["low"]
        ) / data["close"]

        # Volume features
        if "volume" in data.columns:
            data["volume_change"] = data["volume"].pct_change()
            data["price_volume"] = data["close"] * data["volume"]

        # Time features
        data["hour"] = data.index.hour
        data["day_of_week"] = data.index.dayofweek
        data["is_weekend"] = (data.index.dayofweek >= 5).astype(int)

        return data

    def _final_cleanup(self, data: pd.DataFrame) -> pd.DataFrame:
        """Final data cleanup and validation"""
        # Remove infinite values
        data = data.replace([np.inf, -np.inf], np.nan)

        # Forward fill NaN values (limited)
        data = data.fillna(method="ffill", limit=3)

        # Remove remaining NaN rows
        initial_len = len(data)
        data = data.dropna()
        final_len = len(data)

        if initial_len != final_len:
            print(f"  Removed {initial_len - final_len} rows with missing values")

        return data
