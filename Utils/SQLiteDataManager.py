import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from AlgoTrade.Config.Paths import BITGET_DATA_DIR


class SQLiteDataManager:
    """
    Manages OHLCV data storage and retrieval using SQLite database.
    Stores data in the configured Bitget data directory.
    """

    def __init__(self, exchange_name: str = "bitget"):
        self.exchange_name = exchange_name
        self.db_path = BITGET_DATA_DIR / f"{exchange_name}_ohlcv.db"
        self._initialize_database()

    def _initialize_database(self):
        """Initialize the SQLite database with the required table structure"""
        with sqlite3.connect(self.db_path) as conn:
            # Create the main OHLCV table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    date TIMESTAMP NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, date)
                )
            """
            )

            # Create indexes for better query performance
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_symbol_timeframe 
                ON ohlcv_data(symbol, timeframe)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_date 
                ON ohlcv_data(date)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_date 
                ON ohlcv_data(symbol, timeframe, date)
            """
            )

            # Create metadata table to track data info
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    first_date TIMESTAMP,
                    last_date TIMESTAMP,
                    total_records INTEGER,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe)
                )
            """
            )

            conn.commit()

    def store_ohlcv_data(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        update_if_exists: bool = True,
        batch_size: int = 1000,
    ) -> None:
        """
        Store OHLCV data in the SQLite database using batch insertion

        :param data: DataFrame with OHLCV data (index should be datetime)
        :param symbol: Trading pair symbol (e.g., 'BTC/USDT:USDT')
        :param timeframe: Timeframe for the data
        :param update_if_exists: Whether to update existing records
        :param batch_size: Number of records to insert in each batch
        """
        if data.empty:
            print(f"No data to store for {symbol} {timeframe}")
            return

        # Prepare data for insertion
        data_copy = data.copy()
        data_copy.reset_index(inplace=True)
        data_copy["symbol"] = symbol
        data_copy["timeframe"] = timeframe

        # Ensure column names are lowercase
        data_copy.columns = [col.lower() for col in data_copy.columns]

        # Reorder columns for insertion
        columns_order = [
            "symbol",
            "timeframe",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        data_copy = data_copy[columns_order]

        # Convert to list of tuples for batch insertion
        records = [tuple(row) for row in data_copy.values]
        total_records = len(records)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
            conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes

            # Insert in batches to avoid SQL variable limit
            inserted_count = 0

            if update_if_exists:
                insert_sql = """
                    INSERT OR REPLACE INTO ohlcv_data 
                    (symbol, timeframe, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """
            else:
                insert_sql = """
                    INSERT OR IGNORE INTO ohlcv_data 
                    (symbol, timeframe, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """

            # Process in batches
            for i in range(0, total_records, batch_size):
                batch = records[i : i + batch_size]
                try:
                    conn.executemany(insert_sql, batch)
                    inserted_count += len(batch)

                    # Show progress for large datasets
                    if total_records > 1000:
                        progress = (i + len(batch)) / total_records * 100
                        print(
                            f"  Progress: {progress:.1f}% ({i + len(batch)}/{total_records})"
                        )

                except sqlite3.Error as e:
                    print(f"  Error inserting batch {i//batch_size + 1}: {e}")
                    # Try inserting records one by one in this batch
                    for record in batch:
                        try:
                            conn.execute(insert_sql, record)
                            inserted_count += 1
                        except sqlite3.Error as e2:
                            print(f"  Skipping record due to error: {e2}")

            # Update metadata
            self._update_metadata(conn, symbol, timeframe)
            conn.commit()

    def store_ohlcv_data_safe(
        self, data: pd.DataFrame, symbol: str, timeframe: str, batch_size: int = 500
    ) -> bool:
        """
        Safe version of store_ohlcv_data with better error handling and smaller batches

        :param data: DataFrame with OHLCV data (index should be datetime)
        :param symbol: Trading pair symbol (e.g., 'BTC/USDT:USDT')
        :param timeframe: Timeframe for the data
        :param batch_size: Number of records to insert in each batch (smaller = safer)
        :return: True if successful, False otherwise
        """
        if data.empty:
            print(f"No data to store for {symbol} {timeframe}")
            return True

        try:
            # Prepare data for insertion
            data_copy = data.copy()
            data_copy.reset_index(inplace=True)
            data_copy["symbol"] = symbol
            data_copy["timeframe"] = timeframe

            # Ensure column names are lowercase and handle any naming issues
            data_copy.columns = [col.lower().strip() for col in data_copy.columns]

            # Ensure we have the required columns
            required_columns = [
                "symbol",
                "timeframe",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
            for col in required_columns:
                if col not in data_copy.columns:
                    print(f"Missing required column: {col}")
                    return False

            # Reorder columns for insertion
            data_copy = data_copy[required_columns]

            # Remove any rows with null values
            initial_count = len(data_copy)
            data_copy = data_copy.dropna()
            if len(data_copy) < initial_count:
                print(f"Removed {initial_count - len(data_copy)} rows with null values")

            if data_copy.empty:
                print(f"No valid data remaining for {symbol} {timeframe}")
                return True

            # Convert to list of tuples for batch insertion
            records = []
            for _, row in data_copy.iterrows():
                try:
                    # Ensure proper data types
                    record = (
                        str(row["symbol"]),
                        str(row["timeframe"]),
                        str(row["date"]),
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        float(row["volume"]),
                    )
                    records.append(record)
                except (ValueError, TypeError) as e:
                    print(f"Skipping invalid row: {e}")
                    continue

            if not records:
                print(f"No valid records to insert for {symbol} {timeframe}")
                return True

            total_records = len(records)
            inserted_count = 0

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")  # Increase cache

                insert_sql = """
                    INSERT OR IGNORE INTO ohlcv_data 
                    (symbol, timeframe, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """

                # Process in smaller batches
                for i in range(0, total_records, batch_size):
                    batch = records[i : i + batch_size]

                    try:
                        # Use a transaction for each batch
                        conn.execute("BEGIN TRANSACTION")
                        cursor = conn.executemany(insert_sql, batch)
                        conn.execute("COMMIT")

                        inserted_count += len(batch)

                        # Show progress for large datasets
                        if total_records > 500:
                            progress = (i + len(batch)) / total_records * 100
                            print(
                                f"  Progress: {progress:.1f}% ({i + len(batch)}/{total_records})"
                            )

                    except sqlite3.Error as e:
                        conn.execute("ROLLBACK")
                        print(
                            f"  Batch insertion failed, trying individual records: {e}"
                        )

                        # Try inserting records one by one in this batch
                        for record in batch:
                            try:
                                conn.execute("BEGIN TRANSACTION")
                                conn.execute(insert_sql, record)
                                conn.execute("COMMIT")
                                inserted_count += 1
                            except sqlite3.Error as e2:
                                conn.execute("ROLLBACK")
                                print(f"  Skipping record: {e2}")

                # Update metadata
                try:
                    self._update_metadata(conn, symbol, timeframe)
                except Exception as e:
                    print(f"Warning: Could not update metadata: {e}")

            success_rate = (inserted_count / total_records) * 100
            print(
                f"✓ Stored {inserted_count}/{total_records} records ({success_rate:.1f}%) for {symbol} {timeframe}"
            )

            return inserted_count > 0

        except Exception as e:
            print(f"✗ Error storing data for {symbol} {timeframe}: {e}")
            return False

    def _update_metadata(self, conn: sqlite3.Connection, symbol: str, timeframe: str):
        """Update metadata table with information about stored data"""
        # Get data statistics
        result = conn.execute(
            """
            SELECT 
                MIN(date) as first_date,
                MAX(date) as last_date,
                COUNT(*) as total_records
            FROM ohlcv_data 
            WHERE symbol = ? AND timeframe = ?
        """,
            (symbol, timeframe),
        ).fetchone()

        if result:
            first_date, last_date, total_records = result

            # Insert or update metadata
            conn.execute(
                """
                INSERT OR REPLACE INTO data_metadata 
                (symbol, timeframe, first_date, last_date, total_records, last_updated)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (symbol, timeframe, first_date, last_date, total_records),
            )

    def load_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data from the SQLite database

        :param symbol: Trading pair symbol
        :param timeframe: Timeframe for the data
        :param start_date: Start date filter (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        :param end_date: End date filter (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        :return: DataFrame with OHLCV data
        """
        query = """
            SELECT date, open, high, low, close, volume
            FROM ohlcv_data 
            WHERE symbol = ? AND timeframe = ?
        """
        params = [symbol, timeframe]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        with sqlite3.connect(self.db_path) as conn:
            data = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])

        if not data.empty:
            data.set_index("date", inplace=True)
            print(f"✓ Loaded {len(data)} records for {symbol} {timeframe} from SQLite")
        else:
            print(f"No data found for {symbol} {timeframe} in SQLite")

        return data

    def get_available_symbols(self) -> List[str]:
        """Get list of all available symbols in the database"""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                """
                SELECT DISTINCT symbol FROM ohlcv_data ORDER BY symbol
            """
            ).fetchall()

        return [row[0] for row in result]

    def get_available_timeframes(self, symbol: Optional[str] = None) -> List[str]:
        """Get list of available timeframes for a symbol or all symbols"""
        query = "SELECT DISTINCT timeframe FROM ohlcv_data"
        params = []

        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)

        query += " ORDER BY timeframe"

        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(query, params).fetchall()

        return [row[0] for row in result]

    def get_data_info(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Get information about stored data"""
        query = """
            SELECT symbol, timeframe, first_date, last_date, total_records, last_updated
            FROM data_metadata
        """
        params = []

        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)

        query += " ORDER BY symbol, timeframe"

        with sqlite3.connect(self.db_path) as conn:
            info = pd.read_sql_query(query, conn, params=params)

        return info

    def delete_data(self, symbol: str, timeframe: Optional[str] = None) -> None:
        """Delete data for a specific symbol and optionally timeframe"""
        with sqlite3.connect(self.db_path) as conn:
            if timeframe:
                conn.execute(
                    """
                    DELETE FROM ohlcv_data WHERE symbol = ? AND timeframe = ?
                """,
                    (symbol, timeframe),
                )
                conn.execute(
                    """
                    DELETE FROM data_metadata WHERE symbol = ? AND timeframe = ?
                """,
                    (symbol, timeframe),
                )
                print(f"✓ Deleted data for {symbol} {timeframe}")
            else:
                conn.execute(
                    """
                    DELETE FROM ohlcv_data WHERE symbol = ?
                """,
                    (symbol,),
                )
                conn.execute(
                    """
                    DELETE FROM data_metadata WHERE symbol = ?
                """,
                    (symbol,),
                )
                print(f"✓ Deleted all data for {symbol}")

            conn.commit()

    def get_database_size(self) -> Dict[str, Any]:
        """Get database size and statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Get total records
            total_records = conn.execute("SELECT COUNT(*) FROM ohlcv_data").fetchone()[
                0
            ]

            # Get database file size
            file_size = self.db_path.stat().st_size / (1024 * 1024)  # MB

            # Get number of symbols and timeframes
            symbols_count = conn.execute(
                "SELECT COUNT(DISTINCT symbol) FROM ohlcv_data"
            ).fetchone()[0]
            timeframes_count = conn.execute(
                "SELECT COUNT(DISTINCT timeframe) FROM ohlcv_data"
            ).fetchone()[0]

        return {
            "file_path": str(self.db_path),
            "file_size_mb": round(file_size, 2),
            "total_records": total_records,
            "unique_symbols": symbols_count,
            "unique_timeframes": timeframes_count,
        }

    def cleanup_database(self):
        """Optimize database by running VACUUM and updating metadata"""
        with sqlite3.connect(self.db_path) as conn:
            # Update all metadata
            symbols_timeframes = conn.execute(
                """
                SELECT DISTINCT symbol, timeframe FROM ohlcv_data
            """
            ).fetchall()

            for symbol, timeframe in symbols_timeframes:
                self._update_metadata(conn, symbol, timeframe)

            # Vacuum database to reclaim space
            conn.execute("VACUUM")
            conn.commit()

        print("✓ Database cleanup completed")
        
