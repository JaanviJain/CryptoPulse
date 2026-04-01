import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    """Configuration management for the crypto terminal"""
    
    # Project paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / 'data'
    
    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)
    
    # Binance Configuration
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
    BINANCE_BASE_URL = 'https://api.binance.com'
    
    # Trading pairs to track
    TRADING_PAIRS = {
        'BTCUSDT': 'Bitcoin',
        'ETHUSDT': 'Ethereum', 
        'SOLUSDT': 'Solana',
        'BNBUSDT': 'Binance Coin',
        'ADAUSDT': 'Cardano'
    }
    
    # Default symbols (first 5 for MVP)
    DEFAULT_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT']
    
    # Timeframes (in minutes)
    INTERVALS = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
        '1h': '1h',
        '4h': '4h',
        '1d': '1d'
    }
    
    # Default interval for analysis
    DEFAULT_INTERVAL = '15m'
    
    # Database
    DB_PATH = os.getenv('DB_PATH', str(DATA_DIR / 'crypto_data.db'))
    
    # Data collection settings
    UPDATE_INTERVAL = int(os.getenv('UPDATE_INTERVAL', 900))
    MAX_HISTORICAL_DAYS = int(os.getenv('MAX_HISTORICAL_DAYS', 90))
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def get_symbol_name(cls, symbol):
        """Get human-readable name for symbol"""
        return cls.TRADING_PAIRS.get(symbol, symbol.replace('USDT', ''))