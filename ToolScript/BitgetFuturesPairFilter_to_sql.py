"""
Complete Bitget Futures Data Download and Storage Pipeline with Multiple Timeframes

This script demonstrates how to:
1. Filter and select Bitget futures pairs
2. Download OHLCV data for multiple timeframes
3. Store data in SQLite database
4. Load and manage the stored data
"""

from AlgoTrade.Utils.DataManager import DataManager
from AlgoTrade.Utils.BitgetFuturesPairFilter import BitgetFuturesPairFilter
from AlgoTrade.Utils.SQLiteDataManager import SQLiteDataManager
import pandas as pd
from datetime import datetime
import time


def main():
    print("=== Bitget Futures Data Pipeline ===\n")

    # 1. Initialize managers
    print("1. Initializing data managers...")
    dm = DataManager("bitget")
    futures_filter = BitgetFuturesPairFilter(dm)
    sqlite_manager = SQLiteDataManager("bitget")

    # 2. Get recommended futures pairs
    print("\n2. Getting recommended futures pairs...")
    recommended_pairs = futures_filter.get_recommended_futures_pairs(max_pairs=10)

    if not recommended_pairs:
        print("No recommended pairs found. Getting some popular futures...")
        popular_pairs = futures_filter.filter_by_base_currency(
            ["BTC", "ETH", "SOL", "ADA", "DOGE", "AVAX", "MATIC", "DOT"]
        )
        recommended_pairs = popular_pairs[:5]

    print(f"Selected {len(recommended_pairs)} pairs for download:")
    for pair in recommended_pairs:
        print(f"  - {pair}")

    # 3. Define timeframes to download
    timeframes_to_download = ["15m", "1h", "4h"]  # Multiple timeframes
    start_date = "2024-04-01 00:00:00"

    print(f"\n3. Downloading data for timeframes: {', '.join(timeframes_to_download)}")

    # Track results
    successful_downloads = {}
    failed_downloads = {}

    for timeframe in timeframes_to_download:
        successful_downloads[timeframe] = []
        failed_downloads[timeframe] = []

    for i, pair in enumerate(recommended_pairs, 1):
        print(f"\n[{i}/{len(recommended_pairs)}] Processing {pair}...")

        # Download each timeframe for this pair
        for timeframe in timeframes_to_download:
            try:
                print(f"  Downloading {timeframe} data from {start_date}...")
                data = dm.download(pair, timeframe, start_date)

                if not data.empty:
                    print(f"    Downloaded {len(data)} records for {timeframe}")

                    # Store in SQLite using the safer method
                    print(f"    Storing {timeframe} data in SQLite...")
                    success = sqlite_manager.store_ohlcv_data_safe(
                        data, pair, timeframe, batch_size=100  # Safer batch size
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

                # Small delay between timeframes for the same pair
                time.sleep(1)

            except Exception as e:
                print(f"    ✗ Error processing {pair} {timeframe}: {e}")
                failed_downloads[timeframe].append(pair)

        # Longer delay between different pairs
        time.sleep(3)

    # 4. Summary and database info
    print(f"\n4. Download Summary:")

    for timeframe in timeframes_to_download:
        successful_count = len(successful_downloads[timeframe])
        failed_count = len(failed_downloads[timeframe])
        total_attempted = successful_count + failed_count

        print(f"\n  {timeframe} timeframe:")
        print(f"    Successful: {successful_count}/{total_attempted} pairs")
        print(f"    Failed: {failed_count}/{total_attempted} pairs")

        if successful_downloads[timeframe]:
            print(
                f"    Successfully downloaded: {', '.join(successful_downloads[timeframe])}"
            )

        if failed_downloads[timeframe]:
            print(f"    Failed downloads: {', '.join(failed_downloads[timeframe])}")

    # 5. Database statistics
    print(f"\n5. Database Statistics:")
    db_stats = sqlite_manager.get_database_size()
    print(f"  Database file: {db_stats['file_path']}")
    print(f"  File size: {db_stats['file_size_mb']} MB")
    print(f"  Total records: {db_stats['total_records']:,}")
    print(f"  Unique symbols: {db_stats['unique_symbols']}")
    print(f"  Unique timeframes: {db_stats['unique_timeframes']}")

    # 6. Show detailed data info
    print(f"\n6. Stored Data Information:")
    data_info = sqlite_manager.get_data_info()
    if not data_info.empty:
        print(data_info.to_string(index=False))
    else:
        print("  No data stored in database")

    print(f"\n=== Pipeline Complete ===")


def batch_download_by_category():
    """
    Download data for specific categories with multiple timeframes
    """
    print("\n=== Batch Download by Category ===")

    dm = DataManager("bitget")
    futures_filter = BitgetFuturesPairFilter(dm)
    sqlite_manager = SQLiteDataManager("bitget")

    start_date = "2024-01-01 00:00:00"
    timeframes = ["1h", "4h"]  # Multiple timeframes

    # Download top Layer 1 tokens
    print(f"\nDownloading Layer 1 tokens for timeframes: {', '.join(timeframes)}")
    layer1_pairs = futures_filter.get_layer1_futures()[
        :3
    ]  # Top 3 to avoid too much data

    for pair in layer1_pairs:
        print(f"  Processing {pair}...")

        for timeframe in timeframes:
            try:
                print(f"    Downloading {timeframe} data...")
                data = dm.download(pair, timeframe, start_date)

                if not data.empty:
                    success = sqlite_manager.store_ohlcv_data_safe(
                        data, pair, timeframe, batch_size=100
                    )
                    if success:
                        print(f"      ✓ Stored {len(data)} {timeframe} records")
                    else:
                        print(f"      ✗ Failed to store {timeframe} data")
                else:
                    print(f"      ✗ No {timeframe} data available")

                time.sleep(1)  # Rate limiting between timeframes

            except Exception as e:
                print(f"      ✗ Error with {timeframe}: {e}")

        time.sleep(2)  # Rate limiting between pairs

    # Download DeFi tokens
    print(f"\nDownloading DeFi tokens for timeframes: {', '.join(timeframes)}")
    defi_pairs = futures_filter.get_defi_related_futures()[:2]  # Top 2

    for pair in defi_pairs:
        print(f"  Processing {pair}...")

        for timeframe in timeframes:
            try:
                print(f"    Downloading {timeframe} data...")
                data = dm.download(pair, timeframe, start_date)

                if not data.empty:
                    success = sqlite_manager.store_ohlcv_data_safe(
                        data, pair, timeframe, batch_size=100
                    )
                    if success:
                        print(f"      ✓ Stored {len(data)} {timeframe} records")
                    else:
                        print(f"      ✗ Failed to store {timeframe} data")
                else:
                    print(f"      ✗ No {timeframe} data available")

                time.sleep(1)  # Rate limiting between timeframes

            except Exception as e:
                print(f"      ✗ Error with {timeframe}: {e}")

        time.sleep(2)  # Rate limiting between pairs


def update_existing_data_multi_timeframe(symbol: str, timeframes: list):
    """
    Update existing data with new records for multiple timeframes
    """
    print(f"\n=== Updating {symbol} for multiple timeframes ===")

    dm = DataManager("bitget")
    sqlite_manager = SQLiteDataManager("bitget")

    for timeframe in timeframes:
        print(f"\nUpdating {symbol} {timeframe}...")

        try:
            # Get the last date in database
            existing_data = sqlite_manager.load_ohlcv_data(symbol, timeframe)

            if not existing_data.empty:
                last_date = existing_data.index.max()
                start_date = (last_date + pd.Timedelta(hours=1)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                print(f"  Last data: {last_date}, updating from {start_date}")
            else:
                start_date = "2024-01-01 00:00:00"
                print(f"  No existing data, downloading from {start_date}")

            # Download new data
            new_data = dm.download(symbol, timeframe, start_date)

            if not new_data.empty:
                success = sqlite_manager.store_ohlcv_data_safe(
                    new_data, symbol, timeframe, batch_size=100
                )
                if success:
                    print(
                        f"  ✓ Updated {symbol} {timeframe} with {len(new_data)} new records"
                    )
                else:
                    print(f"  ✗ Failed to store updated {timeframe} data")
            else:
                print(f"  No new {timeframe} data available for {symbol}")

            time.sleep(1)  # Rate limiting between timeframes

        except Exception as e:
            print(f"  ✗ Error updating {symbol} {timeframe}: {e}")


def get_multi_timeframe_data(symbol: str, timeframes: list, days: int = 30):
    """
    Get data for a symbol across multiple timeframes
    """
    print(f"\n=== Loading {symbol} data for multiple timeframes ===")

    sqlite_manager = SQLiteDataManager("bitget")
    multi_timeframe_data = {}

    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=days)

    for timeframe in timeframes:
        try:
            data = sqlite_manager.load_ohlcv_data(
                symbol, timeframe, start_date.strftime("%Y-%m-%d %H:%M:%S")
            )
            if not data.empty:
                multi_timeframe_data[timeframe] = data
                print(f"  ✓ Loaded {len(data)} {timeframe} records")
                print(f"    Date range: {data.index.min()} to {data.index.max()}")
            else:
                print(f"  ✗ No {timeframe} data found")
        except Exception as e:
            print(f"  ✗ Error loading {timeframe} data: {e}")

    return multi_timeframe_data


def cleanup_and_maintenance():
    """
    Perform database maintenance tasks
    """
    print("\n=== Database Maintenance ===")

    sqlite_manager = SQLiteDataManager("bitget")

    # Get database info before cleanup
    print("\nBefore cleanup:")
    stats_before = sqlite_manager.get_database_size()
    print(f"  File size: {stats_before['file_size_mb']} MB")
    print(f"  Total records: {stats_before['total_records']:,}")

    # Perform cleanup
    print("\nPerforming database cleanup...")
    sqlite_manager.cleanup_database()

    # Get database info after cleanup
    print("\nAfter cleanup:")
    stats_after = sqlite_manager.get_database_size()
    print(f"  File size: {stats_after['file_size_mb']} MB")
    print(f"  Total records: {stats_after['total_records']:,}")

    space_saved = stats_before["file_size_mb"] - stats_after["file_size_mb"]
    if space_saved > 0:
        print(f"  Space saved: {space_saved:.2f} MB")


def advanced_filtering_examples():
    """
    Show advanced filtering options
    """
    print("\n=== Advanced Filtering Examples ===")

    dm = DataManager("bitget")
    futures_filter = BitgetFuturesPairFilter(dm)

    # Get all available futures pairs
    all_futures = futures_filter.get_available_futures_pairs()
    print(f"\nTotal available futures pairs: {len(all_futures)}")

    # Get filtered pairs by category
    filtered_pairs = futures_filter.get_filtered_futures_pairs(
        include_trending=True,
        include_top_market_cap=True,
        include_high_volume=True,
        include_defi=True,
        include_layer1=True,
        include_meme=True,  # Include meme coins
    )

    print("\nFiltered pairs by category:")
    for category, pairs in filtered_pairs.items():
        print(f"  {category}: {len(pairs)} pairs")
        if pairs:
            print(f"    Examples: {', '.join(pairs[:3])}")

    # Get specific categories
    print(f"\nDeFi futures pairs:")
    defi_pairs = futures_filter.get_defi_related_futures()
    for pair in defi_pairs[:10]:  # Show first 10
        print(f"  {pair}")

    print(f"\nLayer 1 futures pairs:")
    layer1_pairs = futures_filter.get_layer1_futures()
    for pair in layer1_pairs[:10]:  # Show first 10
        print(f"  {pair}")


if __name__ == "__main__":
    # Run the main pipeline
    main()

    # Optional: Run advanced filtering examples
    # advanced_filtering_examples()

    # Optional: Run batch download by category
    # batch_download_by_category()

    # Optional: Update specific symbol with multiple timeframes
    # update_existing_data_multi_timeframe('BTC/USDT:USDT', ['15m', '1h', '4h'])

    # Optional: Get multi-timeframe data for analysis
    # btc_data = get_multi_timeframe_data('BTC/USDT:USDT', ['15m', '1h', '4h'], days=7)

    # Optional: Perform database maintenance
    # cleanup_and_maintenance()
