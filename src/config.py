import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / 'data'
    DATA_DIR.mkdir(exist_ok=True)
    
    # Binance Configuration
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
    BINANCE_BASE_URL = 'https://api.binance.com'
    
    # News APIs
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
    CRYPTOPANIC_API_KEY = os.getenv('CRYPTOPANIC_API_KEY', '')
    
    # On-Chain APIs
    ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY', '')
    SOLSCAN_API_KEY = os.getenv('SOLSCAN_API_KEY', '')
    
    # Trading pairs
    TRADING_PAIRS = {
        'BTCUSDT': 'Bitcoin',
        'ETHUSDT': 'Ethereum',
        'SOLUSDT': 'Solana',
        'BNBUSDT': 'Binance Coin',
        'ADAUSDT': 'Cardano'
    }
    
    DEFAULT_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT']
    
    INTERVALS = {
        '1m': '1m', '5m': '5m', '15m': '15m',
        '30m': '30m', '1h': '1h', '4h': '4h', '1d': '1d'
    }
    
    DEFAULT_INTERVAL = '15m'
    DB_PATH = os.getenv('DB_PATH', str(DATA_DIR / 'crypto_data.db'))
    UPDATE_INTERVAL = int(os.getenv('UPDATE_INTERVAL', 900))
    MAX_HISTORICAL_DAYS = int(os.getenv('MAX_HISTORICAL_DAYS', 90))
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def get_symbol_name(cls, symbol):
        return cls.TRADING_PAIRS.get(symbol, symbol.replace('USDT', ''))    