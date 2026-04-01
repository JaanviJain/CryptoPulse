import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class Database:
    """SQLite database handler"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._create_tables()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _create_tables(self):
        """Create all necessary tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Price data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    interval TEXT DEFAULT '15m',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, interval)
                )
            """)
            
            # Current prices table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS current_prices (
                    symbol TEXT PRIMARY KEY,
                    price REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Daily statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    price_change REAL,
                    price_change_percent REAL,
                    volume REAL,
                    high_24h REAL,
                    low_24h REAL
                )
            """)
            
            # Trading signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    signal_type TEXT NOT NULL,
                    confidence REAL,
                    sentiment_score REAL,
                    prediction_direction TEXT,
                    reasoning TEXT,
                    price_at_signal REAL
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_price_symbol_timestamp 
                ON price_data(symbol, timestamp DESC)
            """)
            
            conn.commit()
            logger.info("Database tables created successfully")
    
    def insert_price_data(self, symbol: str, df: pd.DataFrame, interval: str = '15m'):
        """Insert OHLCV data"""
        if df.empty:
            return
        
        with self.get_connection() as conn:
            data = []
            for _, row in df.iterrows():
                # Convert timestamp to string if it's datetime
                if isinstance(row['timestamp'], datetime):
                    timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                else:
                    timestamp_str = row['timestamp']
                    
                data.append((
                    symbol, 
                    timestamp_str, 
                    float(row['open']),
                    float(row['high']), 
                    float(row['low']), 
                    float(row['close']),
                    float(row['volume']), 
                    interval
                ))
            
            conn.executemany("""
                INSERT OR REPLACE INTO price_data 
                (symbol, timestamp, open, high, low, close, volume, interval)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, data)
            conn.commit()
            logger.debug(f"Inserted {len(data)} records for {symbol}")
    
    def insert_current_prices(self, prices: Dict[str, float]):
        """Insert current prices"""
        with self.get_connection() as conn:
            for symbol, price in prices.items():
                conn.execute("""
                    INSERT OR REPLACE INTO current_prices (symbol, price, timestamp)
                    VALUES (?, ?, ?)
                """, (symbol, price, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
    
    def insert_daily_stats(self, symbol: str, stats: Dict):
        """Insert 24-hour statistics"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO daily_stats 
                (symbol, price_change, price_change_percent, volume, high_24h, low_24h)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                symbol, 
                stats.get('price_change', 0),
                stats.get('price_change_percent', 0),
                stats.get('volume', 0),
                stats.get('high_24h', 0),
                stats.get('low_24h', 0)
            ))
            conn.commit()
            logger.debug(f"Inserted daily stats for {symbol}")
    
    def get_current_prices(self) -> Dict[str, float]:
        """Get all current prices"""
        query = "SELECT symbol, price FROM current_prices"
        with self.get_connection() as conn:
            cursor = conn.execute(query)
            return {row['symbol']: row['price'] for row in cursor.fetchall()}
    
    def get_market_summary(self) -> pd.DataFrame:
        """Get market summary"""
        query = """
            SELECT 
                cp.symbol,
                cp.price,
                COALESCE(ds.price_change_percent, 0) as change_24h,
                COALESCE(ds.volume, 0) as volume_24h
            FROM current_prices cp
            LEFT JOIN daily_stats ds ON cp.symbol = ds.symbol
            WHERE ds.timestamp = (
                SELECT MAX(timestamp) 
                FROM daily_stats 
                WHERE symbol = cp.symbol
            ) OR ds.timestamp IS NULL
            ORDER BY cp.price DESC
        """
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn)
        return df
    
    def get_recent_prices(self, symbol: str, hours: int = 24, interval: str = '15m') -> pd.DataFrame:
        """Get recent price data"""
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM price_data
            WHERE symbol = ? AND interval = ?
            AND timestamp >= datetime('now', '-' || ? || ' hours')
            ORDER BY timestamp ASC
        """
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(symbol, interval, hours))
        
        return df
    
    def get_historical_data(self, symbol: str, start_date: datetime, 
                           end_date: datetime, interval: str = '15m') -> pd.DataFrame:
        """Get historical data between dates for training"""
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM price_data
            WHERE symbol = ? AND interval = ?
            AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
        """
        
        with self.get_connection() as conn:
            # Convert datetime to strings for SQLite
            start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
            end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
            
            df = pd.read_sql_query(query, conn, params=(
                symbol, interval, start_str, end_str
            ))
        
        return df
    
    def get_latest_timestamp(self, symbol: str, interval: str = '15m') -> Optional[datetime]:
        """Get latest timestamp for a symbol"""
        query = """
            SELECT MAX(timestamp) as latest
            FROM price_data
            WHERE symbol = ? AND interval = ?
        """
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, (symbol, interval))
            row = cursor.fetchone()
            if row and row['latest']:
                return datetime.fromisoformat(row['latest'])
        return None