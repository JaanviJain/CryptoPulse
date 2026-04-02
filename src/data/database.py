import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._create_tables()
    
    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _create_tables(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL, high REAL, low REAL, close REAL, volume REAL,
                    interval TEXT DEFAULT '15m',
                    UNIQUE(symbol, timestamp, interval)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS current_prices (
                    symbol TEXT PRIMARY KEY,
                    price REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    price_change REAL, price_change_percent REAL,
                    volume REAL, high_24h REAL, low_24h REAL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS news_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT, description TEXT, content TEXT,
                    url TEXT UNIQUE, source TEXT, published_at DATETIME,
                    crypto_mentioned TEXT, sentiment_score REAL,
                    sentiment_label TEXT, analyzed_at DATETIME
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS whale_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, transaction_hash TEXT,
                    from_address TEXT, to_address TEXT,
                    amount REAL, value_usd REAL,
                    network TEXT, timestamp DATETIME
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS exchange_flows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    inflow_24h REAL, outflow_24h REAL,
                    net_flow REAL, sentiment TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    signal_type TEXT, confidence REAL,
                    sentiment_score REAL, prediction_direction TEXT,
                    whale_alert BOOLEAN, reasoning TEXT,
                    price_at_signal REAL
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_price_symbol_timestamp 
                ON price_data(symbol, timestamp DESC)
            """)
            
            conn.commit()
            logger.info("Database tables created successfully")
    
    def insert_price_data(self, symbol: str, df: pd.DataFrame, interval: str = '15m'):
        if df.empty:
            return
        with self.get_connection() as conn:
            data = []
            for _, row in df.iterrows():
                ts = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(row['timestamp'], 'strftime') else row['timestamp']
                data.append((symbol, ts, float(row['open']), float(row['high']), 
                            float(row['low']), float(row['close']), float(row['volume']), interval))
            conn.executemany("""
                INSERT OR REPLACE INTO price_data 
                (symbol, timestamp, open, high, low, close, volume, interval)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, data)
            conn.commit()
    
    def insert_current_prices(self, prices: Dict[str, float]):
        with self.get_connection() as conn:
            for symbol, price in prices.items():
                conn.execute("""
                    INSERT OR REPLACE INTO current_prices (symbol, price, timestamp)
                    VALUES (?, ?, ?)
                """, (symbol, price, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
    
    def insert_daily_stats(self, symbol: str, stats: Dict):
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO daily_stats 
                (symbol, price_change, price_change_percent, volume, high_24h, low_24h)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol, stats.get('price_change', 0), stats.get('price_change_percent', 0),
                  stats.get('volume', 0), stats.get('high_24h', 0), stats.get('low_24h', 0)))
            conn.commit()
    
    def insert_news_article(self, article: Dict):
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO news_articles 
                (title, description, content, url, source, published_at, crypto_mentioned, 
                 sentiment_score, sentiment_label, analyzed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                article.get('title'), article.get('description'),
                article.get('content'), article.get('url'),
                article.get('source'), article.get('published_at'),
                article.get('crypto_mentioned'),
                article.get('sentiment_score'), article.get('sentiment_label'),
                article.get('analyzed_at')
            ))
            conn.commit()
    
    def insert_whale_alert(self, alert: Dict):
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO whale_alerts 
                (symbol, transaction_hash, from_address, to_address, amount, value_usd, network, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.get('symbol'), alert.get('hash'),
                alert.get('from'), alert.get('to'),
                alert.get('amount'), alert.get('value_usd'),
                alert.get('network'), alert.get('timestamp')
            ))
            conn.commit()
    
    def insert_exchange_flows(self, symbol: str, flows: Dict):
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO exchange_flows 
                (symbol, inflow_24h, outflow_24h, net_flow, sentiment)
                VALUES (?, ?, ?, ?, ?)
            """, (
                symbol, flows.get('inflow_24h', 0),
                flows.get('outflow_24h', 0), flows.get('net_flow', 0),
                flows.get('sentiment', 'neutral')
            ))
            conn.commit()
    
    def get_current_prices(self) -> Dict[str, float]:
        query = "SELECT symbol, price FROM current_prices"
        with self.get_connection() as conn:
            cursor = conn.execute(query)
            return {row['symbol']: row['price'] for row in cursor.fetchall()}
    
    def get_market_summary(self) -> pd.DataFrame:
        query = """
            SELECT cp.symbol, cp.price, COALESCE(ds.price_change_percent, 0) as change_24h
            FROM current_prices cp
            LEFT JOIN daily_stats ds ON cp.symbol = ds.symbol
            WHERE ds.timestamp = (SELECT MAX(timestamp) FROM daily_stats WHERE symbol = cp.symbol)
            ORDER BY cp.price DESC
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)
    
    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str = '15m') -> pd.DataFrame:
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM price_data
            WHERE symbol = ? AND interval = ?
            AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
        """
        with self.get_connection() as conn:
            start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
            end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
            return pd.read_sql_query(query, conn, params=(symbol, interval, start_str, end_str))
    
    def get_latest_timestamp(self, symbol: str, interval: str = '15m') -> Optional[datetime]:
        query = "SELECT MAX(timestamp) as latest FROM price_data WHERE symbol = ? AND interval = ?"
        with self.get_connection() as conn:
            cursor = conn.execute(query, (symbol, interval))
            row = cursor.fetchone()
            if row and row['latest']:
                return datetime.fromisoformat(row['latest'])
        return None