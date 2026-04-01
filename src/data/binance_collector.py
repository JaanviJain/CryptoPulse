import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class BinanceCollector:
    """Binance API collector with batch fetching support"""
    
    def __init__(self, base_url: str = "https://api.binance.com"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Crypto-Terminal/1.0'})
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with error handling"""
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        endpoint = '/api/v3/ticker/price'
        params = {'symbol': symbol}
        data = self._make_request(endpoint, params)
        return float(data['price']) if data and 'price' in data else None
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols"""
        endpoint = '/api/v3/ticker/price'
        data = self._make_request(endpoint)
        
        if data:
            prices = {}
            for item in data:
                if item['symbol'] in symbols:
                    prices[item['symbol']] = float(item['price'])
            return prices
        return {}
    
    def get_klines(self, symbol: str, interval: str = '15m', limit: int = 100) -> pd.DataFrame:
        """Fetch kline/candlestick data (single batch)"""
        endpoint = '/api/v3/klines'
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': min(limit, 1000)
        }
        
        data = self._make_request(endpoint, params)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    def get_historical_data_range(self, symbol: str, interval: str, 
                                  start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch historical data within a date range with pagination
        This handles the 1000 candle limit by fetching in batches
        """
        all_data = []
        current_start = start_date
        batch_count = 0
        
        logger.info(f"Fetching {symbol} data from {start_date} to {end_date}")
        
        while current_start < end_date:
            batch_count += 1
            # Convert to milliseconds timestamp
            start_ts = int(current_start.timestamp() * 1000)
            end_ts = int(end_date.timestamp() * 1000)
            
            endpoint = '/api/v3/klines'
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_ts,
                'endTime': end_ts,
                'limit': 1000  # Max limit
            }
            
            data = self._make_request(endpoint, params)
            
            if not data:
                logger.warning(f"No data returned for batch {batch_count}")
                break
                
            all_data.extend(data)
            logger.debug(f"Batch {batch_count}: fetched {len(data)} candles")
            
            # If we got less than 1000, we've reached the end
            if len(data) < 1000:
                break
                
            # Move to next batch (last timestamp + 1ms)
            last_timestamp = data[-1][0]
            current_start = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(milliseconds=1)
            
            # Rate limit protection
            time.sleep(0.1)
        
        if all_data:
            logger.info(f"Total fetched for {symbol}: {len(all_data)} candles in {batch_count} batches")
            return self._parse_klines_data(all_data)
        
        return pd.DataFrame()
    
    def _parse_klines_data(self, data: List) -> pd.DataFrame:
        """Parse klines data into DataFrame"""
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    def get_24hr_ticker(self, symbol: str) -> Dict:
        """Get 24-hour statistics"""
        endpoint = '/api/v3/ticker/24hr'
        params = {'symbol': symbol}
        data = self._make_request(endpoint, params)
        
        if data:
            return {
                'symbol': symbol,
                'price_change': float(data.get('priceChange', 0)),
                'price_change_percent': float(data.get('priceChangePercent', 0)),
                'volume': float(data.get('volume', 0)),
                'high_24h': float(data.get('highPrice', 0)),
                'low_24h': float(data.get('lowPrice', 0))
            }
        return {}
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic technical indicators"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Price change
        df['price_change'] = df['close'].pct_change()
        
        return df