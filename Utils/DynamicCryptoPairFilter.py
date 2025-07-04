import requests
import pandas as pd
from typing import List, Dict, Any, Optional
import time
from datetime import datetime, timedelta
import json


class DynamicCryptoPairFilter:
    """
    Dynamic crypto pair filter using multiple free data sources:
    - CoinGecko API (trending, market data)
    - DeFiLlama API (DeFi protocols, TVL)
    - CoinMarketCap API (free tier)
    - GitHub activity data
    - Social sentiment data
    - On-chain metrics
    """

    def __init__(self, data_manager, cache_duration_hours: int = 6):
        self.dm = data_manager
        self.markets = None
        self.symbols = None
        self.futures_symbols = None
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.cache = {}

    def initialize_markets(self):
        """Initialize market data from exchange"""
        if self.dm.name != "bitget":
            raise ValueError("This filter is specifically for Bitget exchange")

        self.markets, self.symbols = self.dm.fetch_markets()
        self.futures_symbols = [
            symbol for symbol in self.symbols if symbol.endswith("/USDT:USDT")
        ]

        print(f"Loaded {len(self.symbols)} total symbols from Bitget")
        print(f"Found {len(self.futures_symbols)} futures pairs")

    def _get_cached_or_fetch(self, cache_key: str, fetch_function, *args, **kwargs):
        """Cache mechanism to avoid repeated API calls"""
        now = datetime.now()

        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if now - cached_time < self.cache_duration:
                print(f"Using cached data for {cache_key}")
                return cached_data

        # Fetch new data
        try:
            data = fetch_function(*args, **kwargs)
            self.cache[cache_key] = (now, data)
            return data
        except Exception as e:
            print(f"Error fetching {cache_key}: {e}")
            # Return cached data if available, even if expired
            if cache_key in self.cache:
                return self.cache[cache_key][1]
            return None

    def _fetch_coingecko_trending(self) -> List[Dict]:
        """Fetch trending coins from CoinGecko"""
        url = "https://api.coingecko.com/api/v3/search/trending"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json().get("coins", [])

    def _fetch_coingecko_market_data(self, limit: int = 100) -> List[Dict]:
        """Fetch market cap data from CoinGecko"""
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
            "sparkline": False,
            "price_change_percentage": "24h,7d",
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    def _fetch_coingecko_gainers_losers(self) -> Dict:
        """Fetch top gainers and losers"""
        gainers_url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "percent_change_24h_desc",
            "per_page": 50,
            "page": 1,
        }
        gainers = requests.get(gainers_url, params=params, timeout=10).json()

        params["order"] = "percent_change_24h_asc"
        losers = requests.get(gainers_url, params=params, timeout=10).json()

        return {"gainers": gainers, "losers": losers}

    def _fetch_defillama_protocols(self) -> List[Dict]:
        """Fetch DeFi protocols from DeFiLlama"""
        url = "https://api.llama.fi/protocols"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()

    def _fetch_defillama_chains(self) -> List[Dict]:
        """Fetch blockchain data from DeFiLlama"""
        url = "https://api.llama.fi/v2/chains"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()

    def _fetch_social_sentiment(self) -> List[Dict]:
        """Fetch social sentiment data (using CoinGecko's social data)"""
        try:
            # Get trending coins with social metrics
            url = "https://api.coingecko.com/api/v3/search/trending"
            response = requests.get(url, timeout=10)
            trending_data = response.json()

            social_coins = []
            for coin in trending_data.get("coins", []):
                social_coins.append(
                    {
                        "symbol": coin["item"]["symbol"],
                        "name": coin["item"]["name"],
                        "market_cap_rank": coin["item"].get("market_cap_rank"),
                        "score": coin["item"].get("score", 0),
                    }
                )

            return social_coins
        except Exception as e:
            print(f"Error fetching social sentiment: {e}")
            return []

    def get_trending_futures(self) -> List[str]:
        """Get trending coins available as Bitget futures"""
        if not self.futures_symbols:
            self.initialize_markets()

        trending_data = self._get_cached_or_fetch(
            "coingecko_trending", self._fetch_coingecko_trending
        )

        if not trending_data:
            return []

        trending_futures = []
        for coin in trending_data:
            try:
                item = coin.get("item", {})
                symbol = item.get("symbol", "").upper()
                if symbol:
                    futures_pair = f"{symbol}/USDT:USDT"
                    if futures_pair in self.futures_symbols:
                        trending_futures.append(futures_pair)
            except Exception as e:
                continue

        return trending_futures

    def get_high_market_cap_futures(
        self, min_rank: int = 1, max_rank: int = 100
    ) -> List[str]:
        """Get high market cap coins as futures pairs"""
        if not self.futures_symbols:
            self.initialize_markets()

        market_data = self._get_cached_or_fetch(
            "coingecko_market_data", self._fetch_coingecko_market_data, max_rank
        )

        if not market_data:
            return []

        market_cap_futures = []
        for coin in market_data:
            try:
                market_cap_rank = coin.get("market_cap_rank")
                if market_cap_rank is not None and isinstance(market_cap_rank, int):
                    if min_rank <= market_cap_rank <= max_rank:
                        symbol = coin.get("symbol", "").upper()
                        if symbol:
                            futures_pair = f"{symbol}/USDT:USDT"
                            if futures_pair in self.futures_symbols:
                                market_cap_futures.append(futures_pair)
            except Exception as e:
                continue

        return market_cap_futures

    def get_momentum_futures(
        self, min_change_24h: float = 10.0
    ) -> Dict[str, List[str]]:
        """Get coins with high momentum (gainers/losers)"""
        if not self.futures_symbols:
            self.initialize_markets()

        momentum_data = self._get_cached_or_fetch(
            "coingecko_momentum", self._fetch_coingecko_gainers_losers
        )

        if not momentum_data:
            return {"gainers": [], "losers": []}

        gainers_futures = []
        losers_futures = []

        # Process gainers
        for coin in momentum_data.get("gainers", []):
            try:
                change_24h = coin.get("price_change_percentage_24h")
                if (
                    change_24h is not None
                    and isinstance(change_24h, (int, float))
                    and change_24h >= min_change_24h
                ):
                    symbol = coin.get("symbol", "").upper()
                    if symbol:
                        futures_pair = f"{symbol}/USDT:USDT"
                        if futures_pair in self.futures_symbols:
                            gainers_futures.append(futures_pair)
            except Exception as e:
                continue

        # Process losers (for contrarian strategies)
        for coin in momentum_data.get("losers", []):
            try:
                change_24h = coin.get("price_change_percentage_24h")
                if (
                    change_24h is not None
                    and isinstance(change_24h, (int, float))
                    and change_24h <= -min_change_24h
                ):
                    symbol = coin.get("symbol", "").upper()
                    if symbol:
                        futures_pair = f"{symbol}/USDT:USDT"
                        if futures_pair in self.futures_symbols:
                            losers_futures.append(futures_pair)
            except Exception as e:
                continue

        return {"gainers": gainers_futures, "losers": losers_futures}

    def get_defi_futures(self, min_tvl: float = 100_000_000) -> List[str]:
        """Get DeFi protocol tokens with high TVL"""
        if not self.futures_symbols:
            self.initialize_markets()

        protocols_data = self._get_cached_or_fetch(
            "defillama_protocols", self._fetch_defillama_protocols
        )

        if not protocols_data:
            return []

        defi_futures = []
        for protocol in protocols_data:
            try:
                # Safe TVL extraction with null checks
                tvl = protocol.get("tvl")
                if tvl is None or not isinstance(tvl, (int, float)):
                    continue

                if tvl >= min_tvl:
                    # Try to extract token symbol
                    name = protocol.get("name", "").upper()
                    symbol = protocol.get("symbol", "").upper()

                    # Check various possible symbols
                    possible_symbols = []
                    if symbol:
                        possible_symbols.append(symbol)
                    if name:
                        possible_symbols.append(name)

                    # Add common variations
                    if symbol:
                        possible_symbols.extend(
                            [
                                symbol.replace(" ", ""),
                                symbol.split()[0] if " " in symbol else symbol,
                            ]
                        )

                    for possible_symbol in possible_symbols:
                        if possible_symbol and len(possible_symbol) > 0:
                            futures_pair = f"{possible_symbol}/USDT:USDT"
                            if (
                                futures_pair in self.futures_symbols
                                and futures_pair not in defi_futures
                            ):
                                defi_futures.append(futures_pair)
                                break
            except Exception as e:
                # Skip problematic protocols
                continue

        return defi_futures

    def get_layer1_futures(self, min_tvl: float = 1_000_000_000) -> List[str]:
        """Get Layer 1 blockchain tokens based on DeFiLlama chain data"""
        if not self.futures_symbols:
            self.initialize_markets()

        chains_data = self._get_cached_or_fetch(
            "defillama_chains", self._fetch_defillama_chains
        )

        if not chains_data:
            return []

        layer1_futures = []
        for chain in chains_data:
            try:
                # Safe TVL extraction with null checks
                tvl = chain.get("tvl")
                if tvl is None or not isinstance(tvl, (int, float)):
                    continue

                if tvl >= min_tvl:
                    name = chain.get("name", "").upper()

                    # Map chain names to token symbols
                    chain_token_mapping = {
                        "ETHEREUM": "ETH",
                        "BSC": "BNB",
                        "BINANCE": "BNB",
                        "POLYGON": "MATIC",
                        "AVALANCHE": "AVAX",
                        "SOLANA": "SOL",
                        "FANTOM": "FTM",
                        "ARBITRUM": "ARB",
                        "OPTIMISM": "OP",
                        "POLKADOT": "DOT",
                        "COSMOS": "ATOM",
                        "NEAR": "NEAR",
                        "ALGORAND": "ALGO",
                        "CARDANO": "ADA",
                        "TRON": "TRX",
                    }

                    symbol = chain_token_mapping.get(name, name)
                    if symbol and len(symbol) > 0:
                        futures_pair = f"{symbol}/USDT:USDT"

                        if (
                            futures_pair in self.futures_symbols
                            and futures_pair not in layer1_futures
                        ):
                            layer1_futures.append(futures_pair)
            except Exception as e:
                # Skip problematic chains
                continue

        return layer1_futures

    def get_social_sentiment_futures(self, min_score: int = 0) -> List[str]:
        """Get coins with high social sentiment"""
        if not self.futures_symbols:
            self.initialize_markets()

        social_data = self._get_cached_or_fetch(
            "social_sentiment", self._fetch_social_sentiment
        )

        if not social_data:
            return []

        social_futures = []
        for coin in social_data:
            try:
                score = coin.get("score")
                if (
                    score is not None
                    and isinstance(score, (int, float))
                    and score >= min_score
                ):
                    symbol = coin.get("symbol", "").upper()
                    if symbol:
                        futures_pair = f"{symbol}/USDT:USDT"
                        if futures_pair in self.futures_symbols:
                            social_futures.append(futures_pair)
            except Exception as e:
                continue

        return social_futures

    def get_volume_leaders_futures(self, top_n: int = 50) -> List[str]:
        """Get futures pairs with highest trading volume from exchange data"""
        if not self.futures_symbols:
            self.initialize_markets()

        cache_key = f"volume_leaders_{top_n}"
        cached_data = self._get_cached_or_fetch(
            cache_key, self._fetch_volume_data, top_n
        )

        return cached_data or []

    def _fetch_volume_data(self, top_n: int) -> List[str]:
        """Fetch volume data from exchange"""
        volume_data = []

        print(
            f"Fetching volume data for {min(len(self.futures_symbols), 100)} pairs..."
        )

        for i, symbol in enumerate(
            self.futures_symbols[:100]
        ):  # Limit to avoid rate limits
            try:
                ticker = self.dm.fetch_symbol_ticker_info(symbol)
                if ticker.get("quoteVolume"):
                    volume_data.append(
                        {"symbol": symbol, "volume": ticker["quoteVolume"]}
                    )

                if i % 10 == 0:
                    time.sleep(1)  # Rate limiting
                    print(f"  Processed {i+1} pairs...")

            except Exception as e:
                continue

        # Sort by volume and return top N
        volume_data.sort(key=lambda x: x["volume"], reverse=True)
        return [item["symbol"] for item in volume_data[:top_n]]

    def get_comprehensive_filtered_pairs(
        self, criteria: Dict[str, Any] = None
    ) -> Dict[str, List[str]]:
        """
        Get filtered pairs using multiple dynamic criteria

        :param criteria: Dictionary with filtering criteria
        """
        if not criteria:
            criteria = {
                "trending": True,
                "high_market_cap": {"min_rank": 1, "max_rank": 50},
                "momentum": {"min_change_24h": 15.0},
                "defi": {"min_tvl": 50_000_000},
                "layer1": {"min_tvl": 500_000_000},
                "social_sentiment": {"min_score": 1},
                "volume_leaders": {"top_n": 20},
            }

        filtered_pairs = {}

        print("Fetching comprehensive crypto data from multiple sources...")

        # Trending coins
        if criteria.get("trending"):
            print("  Getting trending coins...")
            filtered_pairs["trending"] = self.get_trending_futures()

        # High market cap
        if criteria.get("high_market_cap"):
            print("  Getting high market cap coins...")
            mc_criteria = criteria["high_market_cap"]
            filtered_pairs["high_market_cap"] = self.get_high_market_cap_futures(
                mc_criteria.get("min_rank", 1), mc_criteria.get("max_rank", 50)
            )

        # Momentum (gainers/losers)
        if criteria.get("momentum"):
            print("  Getting momentum coins...")
            momentum_criteria = criteria["momentum"]
            momentum_pairs = self.get_momentum_futures(
                momentum_criteria.get("min_change_24h", 10.0)
            )
            filtered_pairs["momentum_gainers"] = momentum_pairs["gainers"]
            filtered_pairs["momentum_losers"] = momentum_pairs["losers"]

        # DeFi protocols
        if criteria.get("defi"):
            print("  Getting DeFi protocol tokens...")
            defi_criteria = criteria["defi"]
            filtered_pairs["defi"] = self.get_defi_futures(
                defi_criteria.get("min_tvl", 100_000_000)
            )

        # Layer 1 blockchains
        if criteria.get("layer1"):
            print("  Getting Layer 1 tokens...")
            layer1_criteria = criteria["layer1"]
            filtered_pairs["layer1"] = self.get_layer1_futures(
                layer1_criteria.get("min_tvl", 1_000_000_000)
            )

        # Social sentiment
        if criteria.get("social_sentiment"):
            print("  Getting social sentiment leaders...")
            social_criteria = criteria["social_sentiment"]
            filtered_pairs["social_sentiment"] = self.get_social_sentiment_futures(
                social_criteria.get("min_score", 0)
            )

        # Volume leaders
        if criteria.get("volume_leaders"):
            print("  Getting volume leaders...")
            volume_criteria = criteria["volume_leaders"]
            filtered_pairs["volume_leaders"] = self.get_volume_leaders_futures(
                volume_criteria.get("top_n", 20)
            )

        return filtered_pairs

    def get_smart_recommendations(
        self, max_pairs: int = 20, custom_criteria: Dict[str, Any] = None
    ) -> List[str]:
        """
        Get smart recommendations by combining multiple data sources
        """
        try:
            all_filtered = self.get_comprehensive_filtered_pairs(custom_criteria)
        except Exception as e:
            print(f"Error in comprehensive filtering: {e}")
            print("Falling back to basic filtering...")
            return self._get_fallback_recommendations(max_pairs)

        if not all_filtered or not any(all_filtered.values()):
            print("No pairs found with current criteria. Using fallback...")
            return self._get_fallback_recommendations(max_pairs)

        # Weight different categories
        category_weights = {
            "trending": 3,
            "high_market_cap": 2,
            "momentum_gainers": 2,
            "defi": 1.5,
            "layer1": 1.5,
            "social_sentiment": 1,
            "volume_leaders": 2,
            "momentum_losers": 0.5,  # Lower weight for losers
        }

        # Score pairs based on multiple appearances
        pair_scores = {}

        for category, pairs in all_filtered.items():
            weight = category_weights.get(category, 1)
            print(f"{category}: {len(pairs)} pairs (weight: {weight})")

            for pair in pairs:
                if pair not in pair_scores:
                    pair_scores[pair] = 0
                pair_scores[pair] += weight

        if not pair_scores:
            print("No scored pairs found. Using fallback...")
            return self._get_fallback_recommendations(max_pairs)

        # Sort by score and return top pairs
        recommended = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)

        print(f"\nTop {max_pairs} recommended pairs (with scores):")
        final_pairs = []
        for pair, score in recommended[:max_pairs]:
            print(f"  {pair} (score: {score})")
            final_pairs.append(pair)

        return final_pairs

    def _get_fallback_recommendations(self, max_pairs: int) -> List[str]:
        """
        Fallback method when API calls fail - returns basic popular pairs
        """
        if not self.futures_symbols:
            try:
                self.initialize_markets()
            except Exception as e:
                print(f"Error initializing markets: {e}")
                return []

        # Hardcoded popular pairs as absolute fallback
        popular_pairs = [
            "BTC/USDT:USDT",
            "ETH/USDT:USDT",
            "BNB/USDT:USDT",
            "SOL/USDT:USDT",
            "ADA/USDT:USDT",
            "DOGE/USDT:USDT",
            "AVAX/USDT:USDT",
            "MATIC/USDT:USDT",
            "DOT/USDT:USDT",
            "LINK/USDT:USDT",
            "UNI/USDT:USDT",
            "ATOM/USDT:USDT",
            "FTM/USDT:USDT",
            "NEAR/USDT:USDT",
            "ALGO/USDT:USDT",
        ]

        # Filter to only available pairs
        available_popular = [
            pair for pair in popular_pairs if pair in self.futures_symbols
        ]

        print(f"Using fallback recommendations: {len(available_popular)} popular pairs")
        return available_popular[:max_pairs]
