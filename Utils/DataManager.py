import ccxt
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Literal
import os
from AlgoTrade.Config.Paths import DATA_DIR

EXCHANGES: Dict[str, Dict[str, Any]] = {
    "bitget": {
        "exchange_object": ccxt.bitget(config={"enableRateLimit": True}),
        "limit_size_request": 200,
    },
    "bybit": {
        "exchange_object": ccxt.bybit(config={"enableRateLimit": True}),
        "limit_size_request": 200,
    },
    "alpaca": {
        "exchange_object": ccxt.alpaca(config={"enableRateLimit": True}),
        "limit_size_request": 1000,
    },
}

TIMEFRAMES_TYPE = Literal[
    "1m", "2m", "5m", "15m", "30m", "1h", "2h", "4h", "12h", "1d", "1w", "1M"
]
TIMEFRAMES: Dict[str, Dict[str, Any]] = {
    "1m": {"timedelta": timedelta(minutes=1), "interval_ms": 60000},
    "2m": {"timedelta": timedelta(minutes=2), "interval_ms": 120000},
    "5m": {"timedelta": timedelta(minutes=5), "interval_ms": 300000},
    "15m": {"timedelta": timedelta(minutes=15), "interval_ms": 900000},
    "30m": {"timedelta": timedelta(minutes=30), "interval_ms": 1800000},
    "1h": {"timedelta": timedelta(hours=1), "interval_ms": 3600000},
    "2h": {"timedelta": timedelta(hours=2), "interval_ms": 7200000},
    "4h": {"timedelta": timedelta(hours=4), "interval_ms": 14400000},
    "12h": {"timedelta": timedelta(hours=12), "interval_ms": 43200000},
    "1d": {"timedelta": timedelta(days=1), "interval_ms": 86400000},
    "1w": {"timedelta": timedelta(weeks=1), "interval_ms": 604800000},
    "1M": {"timedelta": timedelta(days=30), "interval_ms": 2629746000},
}


class DataManager:
    """
    Manages downloading and loading OHLCV data for cryptocurrencies
    across various exchanges using the CCXT library.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.exchange = EXCHANGES[self.name]["exchange_object"]
        self._check_support()
        self.markets = None
        self.available_symbols = None

    def fetch_markets(self):
        self.markets = self.exchange.load_markets()
        self.available_symbols = list(self.markets.keys())
        return self.markets, self.available_symbols

    def fetch_symbol_markets_info(self, symbol: str) -> None:
        if not self.markets:
            self.fetch_markets()
        return self.markets[symbol]

    def fetch_symbol_markets_limits(self, symbol: str) -> None:
        if not self.markets:
            self.fetch_markets()
        return self.markets[symbol]["limits"]

    def fetch_symbol_ticker_info(self, symbol: str, params={}) -> None:
        return self.exchange.fetch_ticker(symbol, params)

    def download(
        self,
        symbol: str,
        timeframe: TIMEFRAMES_TYPE,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
        """
        Downloads OHLCV data for a given symbol and timeframe

        :param symbol: Trading pair symbol (e.g., 'BTC/USDT').
        :param timeframe: Timeframe for the OHLCV data.
        :param start_date: Start date for the data in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' format.
        :param end_date: End date for the data in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' format.
        """

        if not self.markets:
            self.fetch_markets()

        if symbol not in self.available_symbols:
            raise ValueError(
                f"The trading pair {symbol} either does not exist on {self.name} or the format is wrong. "
                f"Check with a print('your Ohlcv instance'.available_symbols)"
            )

        if timeframe not in TIMEFRAMES:
            raise ValueError(f"The timeframe {timeframe} is not supported.")

        date_format = "%Y-%m-%d" if timeframe == "1d" else "%Y-%m-%d %H:%M:%S"
        date_format_error_message = f"Dates need to be in the '{date_format}' format."

        if start_date is None:
            start_date = datetime(2017, 1, 1, 0, 0, 0)
        else:
            try:
                start_date = datetime.strptime(start_date, date_format)
            except ValueError:
                raise ValueError(date_format_error_message)

        if end_date is None:
            end_date = datetime.now()
        else:
            try:
                end_date = datetime.strptime(end_date, date_format)
            except ValueError:
                raise ValueError(date_format_error_message)

        ohlcv = self._get_ohlcv(symbol, timeframe, start_date, end_date)
        ohlcv = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        ohlcv["date"] = pd.to_datetime(ohlcv["timestamp"], unit="ms")
        ohlcv.set_index("date", inplace=True)
        ohlcv = ohlcv[~ohlcv.index.duplicated(keep="first")]
        del ohlcv["timestamp"]
        ohlcv = ohlcv.iloc[:-1]
        return ohlcv

    def _check_support(self) -> None:
        if self.name not in EXCHANGES:
            raise ValueError(f"The exchange {self.name} is not supported.")

    @staticmethod
    def _validate_date_format(date: Optional[str], timeframe: str) -> datetime:
        date_format = "%Y-%m-%d" if timeframe == "1d" else "%Y-%m-%d %H:%M:%S"

        try:
            return datetime.strptime(date, date_format)

        except ValueError:
            raise ValueError(
                f"The date '{date}' does not match the expected format '{date_format}'."
            )

    def _get_ohlcv(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> List[List[Any]]:
        current_date_ms = int(start_date.timestamp() * 1000)
        end_date_ms = int(end_date.timestamp() * 1000)
        ohlcv = []

        if self.name == "bitget":
            if ":" not in symbol:
                raise ValueError("Bitget Spot data not supported")

            while current_date_ms < end_date_ms:
                fetched_data = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=EXCHANGES[self.name]["limit_size_request"],
                    params={
                        "method": "publicMixGetV2MixMarketHistoryCandles",
                        "until": current_date_ms
                        + TIMEFRAMES[timeframe]["interval_ms"]
                        * EXCHANGES[self.name]["limit_size_request"],
                    },
                )

                if fetched_data:
                    ohlcv.extend(fetched_data)
                    print(
                        f"fetched ohlcv data for {symbol} ({timeframe}) from {datetime.fromtimestamp(current_date_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")}"
                    )
                else:
                    print(
                        f"fetched ohlcv data for {symbol} ({timeframe}) from {datetime.fromtimestamp(current_date_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")} (empty)"
                    )

                current_date_ms = min(
                    [
                        current_date_ms
                        + int(
                            0.5
                            * TIMEFRAMES[timeframe]["interval_ms"]
                            * EXCHANGES[self.name]["limit_size_request"]
                        ),
                        end_date_ms,
                    ]
                )

        else:
            while current_date_ms < end_date_ms:
                fetched_data = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_date_ms,
                    limit=EXCHANGES[self.name]["limit_size_request"],
                )
                if fetched_data:
                    ohlcv.extend(fetched_data)
                    current_date_ms = fetched_data[-1][0] + 1
                    print(
                        f"fetched ohlcv data for {symbol} from {datetime.fromtimestamp(current_date_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")}"
                    )
                else:
                    break

        return ohlcv

    def to_local(self, data: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Saves OHLCV data to a CSV file in the configured data directory."""
        exchange_dir = DATA_DIR / self.name
        exchange_dir.mkdir(exist_ok=True)

        clean_symbol = symbol.replace("/", "_").replace(":", "_")
        file_path = exchange_dir / f"{clean_symbol}_{timeframe}.csv"

        print(f"Exporting data to {file_path}...")
        data.to_csv(file_path)
        print("Export completed successfully.")

    def from_local(self, symbol: str, timeframe: str, start_date: str) -> pd.DataFrame:
        """Loads OHLCV data from a CSV file in the configured data directory."""
        exchange_dir = DATA_DIR / self.name
        clean_symbol = symbol.replace("/", "_").replace(":", "_")
        file_path = exchange_dir / f"{clean_symbol}_{timeframe}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"No data file found at {file_path}")

        print(f"Loading data from {file_path}...")
        data = pd.read_csv(file_path, index_col="date", parse_dates=True)

        if start_date:
            data = data.loc[start_date:]

        data.columns = [col.lower() for col in data.columns]
        data.sort_index(inplace=True)

        print(f"Load completed successfully. Data shape: {data.shape}")
        return data
