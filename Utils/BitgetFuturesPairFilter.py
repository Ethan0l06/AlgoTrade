import requests
import pandas as pd
from typing import List, Dict, Any
import time


class BitgetFuturesPairFilter:
    def __init__(self, data_manager):
        self.dm = data_manager
        self.markets = None
        self.symbols = None
        self.futures_symbols = None

    def initialize_markets(self):
        """Initialize market data from Bitget exchange"""
        if self.dm.name != "bitget":
            raise ValueError("This filter is specifically for Bitget exchange")

        self.markets, self.symbols = self.dm.fetch_markets()

        # Filter for futures pairs (XXX/USDT:USDT format)
        self.futures_symbols = [
            symbol for symbol in self.symbols if symbol.endswith("/USDT:USDT")
        ]

        print(f"Loaded {len(self.symbols)} total symbols from Bitget")
        print(f"Found {len(self.futures_symbols)} futures pairs")

    def get_available_futures_pairs(self) -> List[str]:
        """Get all available futures pairs on Bitget"""
        if not self.futures_symbols:
            self.initialize_markets()
        return self.futures_symbols.copy()

    def get_coingecko_trending_futures(self) -> List[str]:
        """Get trending coins from CoinGecko that are available as Bitget futures"""
        if not self.futures_symbols:
            self.initialize_markets()

        try:
            url = "https://api.coingecko.com/api/v3/search/trending"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                trending_futures = []
                for coin in data["coins"]:
                    symbol = coin["item"]["symbol"].upper()
                    futures_pair = f"{symbol}/USDT:USDT"
                    if futures_pair in self.futures_symbols:
                        trending_futures.append(futures_pair)
                return trending_futures
        except Exception as e:
            print(f"Error fetching trending coins: {e}")
        return []

    def get_top_market_cap_futures(self, limit: int = 50) -> List[str]:
        """Get top coins by market cap that are available as Bitget futures"""
        if not self.futures_symbols:
            self.initialize_markets()

        try:
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": limit,
                "page": 1,
                "sparkline": False,
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                market_cap_futures = []
                for coin in data:
                    symbol = coin["symbol"].upper()
                    futures_pair = f"{symbol}/USDT:USDT"
                    if futures_pair in self.futures_symbols:
                        market_cap_futures.append(futures_pair)
                return market_cap_futures
        except Exception as e:
            print(f"Error fetching market cap data: {e}")
        return []

    def get_high_volume_futures(self, top_n: int = 20) -> List[str]:
        """Get futures pairs with highest trading volume"""
        if not self.futures_symbols:
            self.initialize_markets()

        volume_data = []
        print(f"Checking volume for {len(self.futures_symbols)} futures pairs...")

        for i, symbol in enumerate(self.futures_symbols):
            try:
                ticker = self.dm.fetch_symbol_ticker_info(symbol)
                if ticker.get("quoteVolume"):
                    volume_data.append(
                        {
                            "symbol": symbol,
                            "volume": ticker["quoteVolume"],
                            "price": ticker.get("last", 0),
                        }
                    )

                # Rate limiting
                if i % 10 == 0:
                    time.sleep(1)
                    print(f"Processed {i+1} futures pairs...")

            except Exception as e:
                print(f"Error fetching ticker for {symbol}: {e}")
                continue

        # Sort by volume and return top N
        volume_data.sort(key=lambda x: x["volume"], reverse=True)
        return [item["symbol"] for item in volume_data[:top_n]]

    def get_defi_related_futures(self) -> List[str]:
        """Get DeFi protocol tokens available as Bitget futures"""
        if not self.futures_symbols:
            self.initialize_markets()

        defi_tokens = [
            "UNI",
            "SUSHI",
            "CAKE",
            "COMP",
            "AAVE",
            "MKR",
            "SNX",
            "CRV",
            "YFI",
            "1INCH",
            "ALPHA",
            "CREAM",
            "BADGER",
            "RUNE",
            "DYDX",
            "LINK",
            "GRT",
            "LDO",
            "RPL",
            "FXS",
            "CVX",
            "BAL",
            "SPELL",
        ]

        defi_futures = []
        for token in defi_tokens:
            futures_pair = f"{token}/USDT:USDT"
            if futures_pair in self.futures_symbols:
                defi_futures.append(futures_pair)
        return defi_futures

    def get_layer1_futures(self) -> List[str]:
        """Get Layer 1 blockchain tokens available as Bitget futures"""
        if not self.futures_symbols:
            self.initialize_markets()

        layer1_tokens = [
            "BTC",
            "ETH",
            "BNB",
            "ADA",
            "SOL",
            "DOT",
            "AVAX",
            "MATIC",
            "ATOM",
            "NEAR",
            "ALGO",
            "EGLD",
            "FLOW",
            "ICP",
            "HBAR",
            "VET",
            "FTM",
            "ONE",
            "ROSE",
            "KAVA",
            "CELO",
            "ZIL",
            "WAVES",
            "ICX",
        ]

        layer1_futures = []
        for token in layer1_tokens:
            futures_pair = f"{token}/USDT:USDT"
            if futures_pair in self.futures_symbols:
                layer1_futures.append(futures_pair)
        return layer1_futures

    def get_meme_futures(self) -> List[str]:
        """Get popular meme coins available as Bitget futures"""
        if not self.futures_symbols:
            self.initialize_markets()

        meme_tokens = [
            "DOGE",
            "SHIB",
            "PEPE",
            "FLOKI",
            "BONK",
            "WIF",
            "BOME",
            "MEME",
            "LADYS",
            "TURBO",
            "BABYDOGE",
            "ELON",
            "AKITA",
        ]

        meme_futures = []
        for token in meme_tokens:
            futures_pair = f"{token}/USDT:USDT"
            if futures_pair in self.futures_symbols:
                meme_futures.append(futures_pair)
        return meme_futures

    def get_filtered_futures_pairs(
        self,
        include_trending: bool = True,
        include_top_market_cap: bool = True,
        include_high_volume: bool = True,
        include_defi: bool = True,
        include_layer1: bool = True,
        include_meme: bool = False,
    ) -> Dict[str, List[str]]:
        """Get filtered futures pairs based on multiple criteria"""

        if not self.futures_symbols:
            self.initialize_markets()

        filtered_pairs = {}

        if include_trending:
            print("Getting trending futures pairs...")
            filtered_pairs["trending"] = self.get_coingecko_trending_futures()

        if include_top_market_cap:
            print("Getting top market cap futures pairs...")
            filtered_pairs["top_market_cap"] = self.get_top_market_cap_futures()

        if include_high_volume:
            print("Getting high volume futures pairs...")
            filtered_pairs["high_volume"] = self.get_high_volume_futures()

        if include_defi:
            print("Getting DeFi futures pairs...")
            filtered_pairs["defi"] = self.get_defi_related_futures()

        if include_layer1:
            print("Getting Layer 1 futures pairs...")
            filtered_pairs["layer1"] = self.get_layer1_futures()

        if include_meme:
            print("Getting meme coin futures pairs...")
            filtered_pairs["meme"] = self.get_meme_futures()

        return filtered_pairs

    def get_recommended_futures_pairs(self, max_pairs: int = 20) -> List[str]:
        """Get a final recommended list of futures pairs"""
        all_filtered = self.get_filtered_futures_pairs()

        # Combine all pairs and count occurrences
        pair_counts = {}
        for category, pairs in all_filtered.items():
            print(f"{category}: {len(pairs)} futures pairs")
            for pair in pairs:
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        # Sort by occurrence count (pairs appearing in multiple categories)
        recommended = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)

        print(f"\nTop {max_pairs} recommended futures pairs:")
        final_pairs = []
        for pair, count in recommended[:max_pairs]:
            print(f"{pair} (appears in {count} categories)")
            final_pairs.append(pair)

        return final_pairs

    def filter_by_base_currency(self, base_currencies: List[str]) -> List[str]:
        """Filter futures pairs by specific base currencies"""
        if not self.futures_symbols:
            self.initialize_markets()

        filtered_pairs = []
        for currency in base_currencies:
            futures_pair = f"{currency.upper()}/USDT:USDT"
            if futures_pair in self.futures_symbols:
                filtered_pairs.append(futures_pair)

        return filtered_pairs