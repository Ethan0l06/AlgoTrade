"""
Updated Bitget Futures Data Pipeline with Dynamic Filtering

Uses multiple free data sources:
- CoinGecko (trending, market data, social sentiment)
- DeFiLlama (DeFi protocols, blockchain TVL)
- Exchange volume data
- Momentum analysis
"""

from AlgoTrade.Utils.DataManager import DataManager
from AlgoTrade.Utils.DynamicCryptoPairFilter import DynamicCryptoPairFilter
from AlgoTrade.Utils.SQLiteDataManager import SQLiteDataManager
import pandas as pd
from datetime import datetime
import time


def main():
    print("=== Dynamic Bitget Futures Data Pipeline ===\n")

    # 1. Initialize managers
    print("1. Initializing data managers...")
    dm = DataManager("bitget")
    dynamic_filter = DynamicCryptoPairFilter(dm, cache_duration_hours=6)
    sqlite_manager = SQLiteDataManager("bitget")

    # 2. Get smart recommendations using dynamic filtering
    print("\n2. Getting smart recommendations from multiple data sources...")

    # Define custom criteria (adjust as needed)
    custom_criteria = {
        "trending": True,
        "high_market_cap": {"min_rank": 1, "max_rank": 50},
        "momentum": {"min_change_24h": 10.0},
        "defi": {"min_tvl": 100_000_000},
        "layer1": {"min_tvl": 1_000_000_000},
        "social_sentiment": {"min_score": 1},
        "volume_leaders": {"top_n": 25},
    }

    recommended_pairs = dynamic_filter.get_smart_recommendations(
        max_pairs=15, custom_criteria=custom_criteria
    )

    if not recommended_pairs:
        print("No pairs found with current criteria. Using fallback...")
        # Fallback to basic filtering
        try:
            all_futures = dynamic_filter.get_available_futures_pairs()
            recommended_pairs = all_futures[:10]  # Just take first 10
        except:
            recommended_pairs = []

    if not recommended_pairs:
        print("ERROR: No futures pairs available. Check Bitget connection.")
        return

    print(f"\nSelected {len(recommended_pairs)} pairs for download:")
    for i, pair in enumerate(recommended_pairs, 1):
        print(f"  {i}. {pair}")

    # 3. Define timeframes to download
    timeframes_to_download = ["15m", "1h", "4h"]
    start_date = "2024-01-01 00:00:00"  # Adjust date range as needed

    print(f"\n3. Downloading data for timeframes: {', '.join(timeframes_to_download)}")
    print(f"   Date range: from {start_date}")

    # Track results
    successful_downloads = {}
    failed_downloads = {}

    for timeframe in timeframes_to_download:
        successful_downloads[timeframe] = []
        failed_downloads[timeframe] = []

    # 4. Download data
    total_combinations = len(recommended_pairs) * len(timeframes_to_download)
    current_combination = 0

    for i, pair in enumerate(recommended_pairs, 1):
        print(f"\n[{i}/{len(recommended_pairs)}] Processing {pair}...")

        # Download each timeframe for this pair
        for timeframe in timeframes_to_download:
            current_combination += 1
            try:
                print(
                    f"  [{current_combination}/{total_combinations}] Downloading {timeframe} data..."
                )
                data = dm.download(pair, timeframe, start_date)

                if not data.empty:
                    print(f"    Downloaded {len(data)} records for {timeframe}")

                    # Store in SQLite using safe batch method
                    print(f"    Storing {timeframe} data in SQLite...")
                    success = sqlite_manager.store_ohlcv_data_safe(
                        data, pair, timeframe, batch_size=100
                    )

                    if success:

                        successful_downloads[timeframe].append(pair)
                        print(f"    ✓ Successfully processed {pair} {timeframe}")
                    else:
                        print(f"    ✗ Failed to store {timeframe} data for {pair}")
                        failed_downloads[timeframe].append(pair)
                else:
                    print(f"    ✗ No {timeframe} data available for {pair}")
                    failed_downloads[timeframe].append(pair)

                # Rate limiting between timeframes
                time.sleep(1)

            except Exception as e:
                print(f"    ✗ Error processing {pair} {timeframe}: {e}")
                failed_downloads[timeframe].append(pair)

        # Longer delay between different pairs
        time.sleep(2)

    # 5. Comprehensive summary
    print(f"\n4. Download Summary:")
    print("=" * 50)

    total_successful = 0
    total_failed = 0

    for timeframe in timeframes_to_download:
        successful_count = len(successful_downloads[timeframe])
        failed_count = len(failed_downloads[timeframe])
        total_attempted = successful_count + failed_count

        total_successful += successful_count
        total_failed += failed_count

        success_rate = (
            (successful_count / total_attempted * 100) if total_attempted > 0 else 0
        )

        print(f"\n  📊 {timeframe} timeframe:")
        print(
            f"     Successful: {successful_count}/{total_attempted} pairs ({success_rate:.1f}%)"
        )

        if successful_downloads[timeframe]:
            print(
                f"     ✓ Success: {', '.join(successful_downloads[timeframe][:5])}"
                + (
                    f" + {len(successful_downloads[timeframe])-5} more"
                    if len(successful_downloads[timeframe]) > 5
                    else ""
                )
            )

        if failed_downloads[timeframe]:
            print(
                f"     ✗ Failed: {', '.join(failed_downloads[timeframe][:3])}"
                + (
                    f" + {len(failed_downloads[timeframe])-3} more"
                    if len(failed_downloads[timeframe]) > 3
                    else ""
                )
            )

    # Overall statistics
    total_combinations_attempted = total_successful + total_failed
    overall_success_rate = (
        (total_successful / total_combinations_attempted * 100)
        if total_combinations_attempted > 0
        else 0
    )

    print(f"\n  🎯 Overall Results:")
    print(
        f"     Total successful downloads: {total_successful}/{total_combinations_attempted} ({overall_success_rate:.1f}%)"
    )
    print(f"     Total pairs processed: {len(recommended_pairs)}")
    print(f"     Total timeframes: {len(timeframes_to_download)}")

    # 6. Database statistics
    print(f"\n5. Database Statistics:")
    print("=" * 30)
    try:
        db_stats = sqlite_manager.get_database_size()
        print(f"  📁 Database file: {db_stats['file_path']}")
        print(f"  💾 File size: {db_stats['file_size_mb']} MB")
        print(f"  📊 Total records: {db_stats['total_records']:,}")
        print(f"  🎲 Unique symbols: {db_stats['unique_symbols']}")
        print(f"  ⏰ Unique timeframes: {db_stats['unique_timeframes']}")
    except Exception as e:
        print(f"  ✗ Error getting database stats: {e}")

    # 7. Show detailed data info
    print(f"\n6. Stored Data Details:")
    print("=" * 25)
    try:
        data_info = sqlite_manager.get_data_info()
        if not data_info.empty:
            # Group by symbol for better readability
            for symbol in data_info["symbol"].unique():
                symbol_data = data_info[data_info["symbol"] == symbol]
                print(f"\n  📈 {symbol}:")
                for _, row in symbol_data.iterrows():
                    print(
                        f"     {row['timeframe']}: {row['total_records']:,} records "
                        f"({row['first_date']} to {row['last_date']})"
                    )
        else:
            print("  No data stored in database")
    except Exception as e:
        print(f"  ✗ Error getting data info: {e}")

    print(f"\n=== Pipeline Complete ===")

    # Return summary for potential further processing
    return {
        "successful_downloads": successful_downloads,
        "failed_downloads": failed_downloads,
        "recommended_pairs": recommended_pairs,
        "total_successful": total_successful,
        "total_failed": total_failed,
    }


def run_custom_filtering_examples():
    """
    Show examples of custom filtering with different criteria
    """
    print("\n=== Custom Filtering Examples ===")

    dm = DataManager("bitget")
    dynamic_filter = DynamicCryptoPairFilter(dm)

    # Example 1: Conservative strategy (established coins only)
    print("\n1. Conservative Strategy (Top 20 market cap + high volume):")
    conservative_criteria = {
        "high_market_cap": {"min_rank": 1, "max_rank": 20},
        "volume_leaders": {"top_n": 15},
    }
    conservative_pairs = dynamic_filter.get_smart_recommendations(
        max_pairs=10, custom_criteria=conservative_criteria
    )

    # Example 2: Growth strategy (trending + momentum)
    print("\n2. Growth Strategy (Trending + momentum gainers):")
    growth_criteria = {
        "trending": True,
        "momentum": {"min_change_24h": 15.0},
        "social_sentiment": {"min_score": 2},
    }
    growth_pairs = dynamic_filter.get_smart_recommendations(
        max_pairs=10, custom_criteria=growth_criteria
    )

    # Example 3: DeFi focused strategy
    print("\n3. DeFi Strategy (High TVL protocols + Layer 1):")
    defi_criteria = {
        "defi": {"min_tvl": 200_000_000},
        "layer1": {"min_tvl": 2_000_000_000},
        "high_market_cap": {"min_rank": 1, "max_rank": 100},
    }
    defi_pairs = dynamic_filter.get_smart_recommendations(
        max_pairs=10, custom_criteria=defi_criteria
    )

    # Example 4: Momentum strategy (both gainers and losers for swing trading)
    print("\n4. Momentum Strategy (Strong price movements):")
    momentum_criteria = {
        "momentum": {"min_change_24h": 20.0},
        "volume_leaders": {"top_n": 20},
        "trending": True,
    }
    momentum_pairs = dynamic_filter.get_smart_recommendations(
        max_pairs=10, custom_criteria=momentum_criteria
    )

    # Example 5: Comprehensive strategy (balanced approach)
    print("\n5. Comprehensive Strategy (Balanced criteria):")
    comprehensive_criteria = {
        "trending": True,
        "high_market_cap": {"min_rank": 1, "max_rank": 50},
        "momentum": {"min_change_24h": 5.0},
        "defi": {"min_tvl": 50_000_000},
        "layer1": {"min_tvl": 500_000_000},
        "social_sentiment": {"min_score": 0},
        "volume_leaders": {"top_n": 30},
    }
    comprehensive_pairs = dynamic_filter.get_smart_recommendations(
        max_pairs=15, custom_criteria=comprehensive_criteria
    )

    return {
        "conservative": conservative_pairs,
        "growth": growth_pairs,
        "defi": defi_pairs,
        "momentum": momentum_pairs,
        "comprehensive": comprehensive_pairs,
    }


def analyze_filtering_sources():
    """
    Analyze the effectiveness of different data sources
    """
    print("\n=== Data Source Analysis ===")

    dm = DataManager("bitget")
    dynamic_filter = DynamicCryptoPairFilter(dm)

    # Get data from each source individually
    print("\nAnalyzing individual data sources...")

    try:
        # CoinGecko sources
        trending = dynamic_filter.get_trending_futures()
        high_mc = dynamic_filter.get_high_market_cap_futures(1, 50)
        momentum = dynamic_filter.get_momentum_futures(10.0)
        social = dynamic_filter.get_social_sentiment_futures(1)

        # DeFiLlama sources
        defi = dynamic_filter.get_defi_futures(100_000_000)
        layer1 = dynamic_filter.get_layer1_futures(1_000_000_000)

        # Exchange source
        volume = dynamic_filter.get_volume_leaders_futures(25)

        print(f"\n📊 Data Source Results:")
        print(f"  🔥 Trending (CoinGecko): {len(trending)} pairs")
        print(f"  💰 High Market Cap: {len(high_mc)} pairs")
        print(f"  📈 Momentum Gainers: {len(momentum['gainers'])} pairs")
        print(f"  📉 Momentum Losers: {len(momentum['losers'])} pairs")
        print(f"  💬 Social Sentiment: {len(social)} pairs")
        print(f"  🏦 DeFi Protocols: {len(defi)} pairs")
        print(f"  🔗 Layer 1 Chains: {len(layer1)} pairs")
        print(f"  📊 Volume Leaders: {len(volume)} pairs")

        # Find overlaps
        all_sources = {
            "trending": set(trending),
            "high_market_cap": set(high_mc),
            "momentum_gainers": set(momentum["gainers"]),
            "social_sentiment": set(social),
            "defi": set(defi),
            "layer1": set(layer1),
            "volume_leaders": set(volume),
        }

        print(f"\n🔄 Source Overlaps:")
        for source1, pairs1 in all_sources.items():
            for source2, pairs2 in all_sources.items():
                if source1 < source2:  # Avoid duplicate comparisons
                    overlap = pairs1.intersection(pairs2)
                    if overlap:
                        print(
                            f"  {source1} ∩ {source2}: {len(overlap)} pairs - {list(overlap)[:3]}"
                        )

        # Most common pairs across sources
        all_pairs = {}
        for source, pairs in all_sources.items():
            for pair in pairs:
                if pair not in all_pairs:
                    all_pairs[pair] = []
                all_pairs[pair].append(source)

        # Sort by number of sources
        popular_pairs = sorted(all_pairs.items(), key=lambda x: len(x[1]), reverse=True)

        print(f"\n🌟 Most Popular Pairs (appearing in multiple sources):")
        for pair, sources in popular_pairs[:10]:
            print(f"  {pair}: appears in {len(sources)} sources - {sources}")

    except Exception as e:
        print(f"Error in analysis: {e}")


def batch_download_with_custom_strategies():
    """
    Download data using different filtering strategies
    """
    print("\n=== Batch Download with Custom Strategies ===")

    dm = DataManager("bitget")
    dynamic_filter = DynamicCryptoPairFilter(dm)
    sqlite_manager = SQLiteDataManager("bitget")

    # Define different strategies
    strategies = {
        "blue_chip": {
            "criteria": {
                "high_market_cap": {"min_rank": 1, "max_rank": 15},
                "volume_leaders": {"top_n": 10},
            },
            "max_pairs": 8,
            "timeframes": ["1h", "4h"],
        },
        "growth": {
            "criteria": {
                "trending": True,
                "momentum": {"min_change_24h": 12.0},
                "social_sentiment": {"min_score": 1},
            },
            "max_pairs": 6,
            "timeframes": ["15m", "1h"],
        },
        "defi_focus": {
            "criteria": {
                "defi": {"min_tvl": 150_000_000},
                "layer1": {"min_tvl": 1_500_000_000},
            },
            "max_pairs": 5,
            "timeframes": ["1h", "4h"],
        },
    }

    start_date = "2024-05-01 00:00:00"

    for strategy_name, strategy_config in strategies.items():
        print(f"\n🎯 Executing {strategy_name.upper()} strategy...")

        # Get pairs for this strategy
        pairs = dynamic_filter.get_smart_recommendations(
            max_pairs=strategy_config["max_pairs"],
            custom_criteria=strategy_config["criteria"],
        )

        if not pairs:
            print(f"  No pairs found for {strategy_name} strategy")
            continue

        print(f"  Selected pairs: {pairs}")

        # Download data for each pair and timeframe
        for pair in pairs:
            for timeframe in strategy_config["timeframes"]:
                try:
                    print(f"    Downloading {pair} {timeframe}...")
                    data = dm.download(pair, timeframe, start_date)

                    if not data.empty:
                        success = sqlite_manager.store_ohlcv_data_safe(
                            data, pair, timeframe, batch_size=100
                        )
                        if success:
                            print(f"      ✓ Stored {len(data)} records")
                        else:
                            print(f"      ✗ Failed to store data")
                    else:
                        print(f"      ✗ No data available")

                    time.sleep(1)

                except Exception as e:
                    print(f"      ✗ Error: {e}")

            time.sleep(2)


def update_data_with_smart_filtering():
    """
    Update existing data and add new pairs based on current market conditions
    """
    print("\n=== Smart Data Update ===")

    dm = DataManager("bitget")
    dynamic_filter = DynamicCryptoPairFilter(dm)
    sqlite_manager = SQLiteDataManager("bitget")

    # Get current pairs in database
    existing_symbols = sqlite_manager.get_available_symbols()
    print(f"Currently have data for {len(existing_symbols)} symbols")

    # Get current smart recommendations
    current_recommendations = dynamic_filter.get_smart_recommendations(max_pairs=20)

    # Find new pairs to add
    new_pairs = [
        pair for pair in current_recommendations if pair not in existing_symbols
    ]
    existing_pairs_to_update = [
        pair for pair in current_recommendations if pair in existing_symbols
    ]

    print(f"\n📊 Update Analysis:")
    print(f"  New pairs to add: {len(new_pairs)}")
    print(f"  Existing pairs to update: {len(existing_pairs_to_update)}")

    if new_pairs:
        print(
            f"  🆕 New pairs: {new_pairs[:5]}"
            + (f" + {len(new_pairs)-5} more" if len(new_pairs) > 5 else "")
        )

    if existing_pairs_to_update:
        print(
            f"  🔄 Update pairs: {existing_pairs_to_update[:5]}"
            + (
                f" + {len(existing_pairs_to_update)-5} more"
                if len(existing_pairs_to_update) > 5
                else ""
            )
        )

    # Download new pairs
    if new_pairs:
        print(f"\n📥 Downloading new pairs...")
        timeframes = ["1h", "4h"]
        start_date = "2024-01-01 00:00:00"

        for pair in new_pairs[:5]:  # Limit to 5 new pairs to avoid overload
            for timeframe in timeframes:
                try:
                    print(f"  Downloading {pair} {timeframe}...")
                    data = dm.download(pair, timeframe, start_date)

                    if not data.empty:
                        success = sqlite_manager.store_ohlcv_data_safe(
                            data, pair, timeframe, batch_size=100
                        )
                        if success:
                            print(f"    ✓ Added {len(data)} records")

                    time.sleep(1)
                except Exception as e:
                    print(f"    ✗ Error: {e}")

    # Update existing pairs (get latest data)
    if existing_pairs_to_update:
        print(f"\n🔄 Updating existing pairs...")

        for pair in existing_pairs_to_update[:3]:  # Update first 3
            for timeframe in ["1h"]:  # Just hourly updates
                try:
                    existing_data = sqlite_manager.load_ohlcv_data(pair, timeframe)

                    if not existing_data.empty:
                        last_date = existing_data.index.max()
                        start_date = (last_date + pd.Timedelta(hours=1)).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    else:
                        start_date = "2024-01-01 00:00:00"

                    print(f"  Updating {pair} {timeframe} from {start_date}...")
                    new_data = dm.download(pair, timeframe, start_date)

                    if not new_data.empty:
                        success = sqlite_manager.store_ohlcv_data_safe(
                            new_data, pair, timeframe, batch_size=100
                        )
                        if success:
                            print(f"    ✓ Added {len(new_data)} new records")
                    else:
                        print(f"    ℹ️ No new data available")

                    time.sleep(1)
                except Exception as e:
                    print(f"    ✗ Error: {e}")


if __name__ == "__main__":
    # Run the main dynamic pipeline
    results = main()

    # Optional: Run custom filtering examples
    print("\n" + "=" * 60)
    # custom_strategies = run_custom_filtering_examples()

    # Optional: Analyze data sources
    # analyze_filtering_sources()

    # Optional: Batch download with strategies
    # batch_download_with_custom_strategies()

    # Optional: Smart update
    # update_data_with_smart_filtering()
