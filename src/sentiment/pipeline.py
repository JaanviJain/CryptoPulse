import logging
from typing import Dict, List
from datetime import datetime

from src.sentiment.analyzer import SentimentAnalyzer
from src.data.database import Database

logger = logging.getLogger(__name__)

class SentimentPipeline:
    """Simplified sentiment pipeline using only price data and mock sentiment for now"""
    
    def __init__(self, db: Database, model_name: str = "llama3.2:3b"):
        self.db = db
        self.analyzer = SentimentAnalyzer(model_name)
        logger.info("Sentiment pipeline initialized")
    
    def analyze_crypto_sentiment(self, symbol: str) -> Dict:
        """Analyze sentiment using mock data (replace with real data later)"""
        logger.info(f"Analyzing sentiment for {symbol}")
        
        # Mock sentiment data for testing
        # In production, you'd get this from social media, news, etc.
        mock_sentiments = {
            'BTCUSDT': {'score': 0.75, 'sentiment': 'BULLISH'},
            'ETHUSDT': {'score': 0.62, 'sentiment': 'BULLISH'},
            'SOLUSDT': {'score': 0.45, 'sentiment': 'BULLISH'},
            'BNBUSDT': {'score': 0.12, 'sentiment': 'NEUTRAL'},
            'ADAUSDT': {'score': -0.08, 'sentiment': 'NEUTRAL'}
        }
        
        mock_data = mock_sentiments.get(symbol, {'score': 0.0, 'sentiment': 'NEUTRAL'})
        
        result = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'sentiment': mock_data['sentiment'],
            'sentiment_score': mock_data['score'],
            'source': 'mock_data'
        }
        
        # Store in database
        self._store_sentiment_result(result)
        
        return result
    
    def _store_sentiment_result(self, result: Dict):
        """Store sentiment analysis result in database"""
        try:
            with self.db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO trading_signals 
                    (symbol, timestamp, sentiment_score, signal_type, reasoning)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    result['symbol'],
                    result['timestamp'],
                    result['sentiment_score'],
                    result['sentiment'],
                    f"Sentiment score: {result['sentiment_score']:.2f}"
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing sentiment: {e}")
    
    def analyze_all_cryptos(self) -> Dict[str, Dict]:
        """Analyze sentiment for all tracked cryptocurrencies"""
        results = {}
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT']
        
        for symbol in symbols:
            try:
                result = self.analyze_crypto_sentiment(symbol)
                results[symbol] = result
                logger.info(f"{symbol}: {result['sentiment']} (score: {result['sentiment_score']:.2f})")
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                results[symbol] = {'sentiment': 'ERROR', 'sentiment_score': 0}
        
        return results