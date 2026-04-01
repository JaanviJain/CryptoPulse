import logging
from datetime import datetime, timedelta
from typing import List
from tqdm import tqdm

from src.config import Config
from src.data.binance_collector import BinanceCollector
from src.data.database import Database

logger = logging.getLogger(__name__)

class DataPipeline:
    """Main data pipeline for collecting crypto data"""
    
    def __init__(self):
        self.config = Config()
        self.collector = BinanceCollector()
        self.db = Database(self.config.DB_PATH)
        self.symbols = self.config.DEFAULT_SYMBOLS
        logger.info("Data pipeline initialized")
    
    def fetch_historical_data(self, days_back: int = 7, interval: str = '15m'):
        """Fetch historical data for all symbols with batch processing"""
        logger.info(f"Fetching {days_back} days of historical data")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        for symbol in tqdm(self.symbols, desc="Fetching data"):
            try:
                # Use batch fetching to get all data
                df = self.collector.get_historical_data_range(symbol, interval, start_date, end_date)
                
                if not df.empty:
                    # Convert timestamp to string for SQLite
                    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    # Store in database
                    self.db.insert_price_data(symbol, df, interval)
                    
                    # Calculate expected vs actual candles
                    expected_candles = days_back * 96  # 96 candles per day for 15min
                    actual_candles = len(df)
                    
                    logger.info(f"Stored {actual_candles} candles for {symbol} "
                               f"(Expected: {expected_candles}, "
                               f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
                    
                    if actual_candles < expected_candles * 0.8:
                        logger.warning(f"Only got {actual_candles}/{expected_candles} candles for {symbol}")
                else:
                    logger.warning(f"No data received for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
    
    def fetch_current_prices(self):
        """Fetch and store current prices"""
        logger.info("Fetching current prices")
        prices = self.collector.get_multiple_prices(self.symbols)
        
        if prices:
            self.db.insert_current_prices(prices)
            for symbol, price in prices.items():
                logger.info(f"{symbol}: ${price:,.2f}")
        
        return prices
    
    def fetch_daily_stats(self):
        """Fetch 24-hour statistics"""
        logger.info("Fetching 24-hour statistics")
        
        for symbol in self.symbols:
            try:
                stats = self.collector.get_24hr_ticker(symbol)
                if stats:
                    self.db.insert_daily_stats(symbol, stats)
                    logger.info(f"Updated stats for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching stats for {symbol}: {e}")
    
    def run_initial_setup(self):
        """Run initial data collection"""
        logger.info("Starting initial setup...")
        self.fetch_historical_data(days_back=60)  # Fetch 60 days by default
        self.fetch_current_prices()
        self.fetch_daily_stats()
        logger.info("Initial setup complete")