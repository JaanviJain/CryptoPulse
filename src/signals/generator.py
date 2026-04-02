import logging
from typing import Dict
from datetime import datetime
from tqdm import tqdm

logger = logging.getLogger(__name__)

class SignalGenerator:
    def __init__(self, db):
        self.db = db
        logger.info("Signal generator initialized")
    
    def generate_signal(self, symbol: str, sentiment: Dict, prediction: Dict, onchain: Dict, current_price: float) -> Dict:
        sentiment_score = sentiment.get('sentiment_score', 0)
        sentiment_label = sentiment.get('sentiment', 'NEUTRAL')
        prediction_direction = prediction.get('direction', 'SIDEWAYS')
        prediction_confidence = prediction.get('confidence', 0.5)
        whale_alerts = onchain.get('whale_alerts', [])
        exchange_flows = onchain.get('exchange_flows', {})
        
        has_whale_activity = len(whale_alerts) > 0
        is_accumulation = exchange_flows.get('sentiment') == 'accumulation'
        
        reasoning = []
        
        if sentiment_score > 0.7:
            reasoning.append(f"Strong bullish sentiment: {sentiment_score:.2f}")
        elif sentiment_score < -0.3:
            reasoning.append(f"Bearish sentiment: {sentiment_score:.2f}")
        
        if prediction_direction == 'UP' and prediction_confidence > 0.55:
            reasoning.append(f"Price prediction: UP with {prediction_confidence:.1%} confidence")
        elif prediction_direction == 'DOWN' and prediction_confidence > 0.55:
            reasoning.append(f"Price prediction: DOWN with {prediction_confidence:.1%} confidence")
        
        if has_whale_activity:
            reasoning.append(f"Whale activity detected: {len(whale_alerts)} large transactions")
        
        if is_accumulation:
            reasoning.append("Exchange outflows suggest accumulation")
        
        signal_type = "HOLD"
        confidence = 0.5
        
        if sentiment_score > 0.7 and prediction_direction == 'UP' and has_whale_activity:
            signal_type = "STRONG BUY"
            confidence = min(0.95, (sentiment_score + prediction_confidence + 0.7) / 3)
            reasoning.append("Strong buy: Bullish sentiment + Up prediction + Whale accumulation")
        elif sentiment_score > 0.5 and prediction_direction == 'UP':
            signal_type = "BUY"
            confidence = (sentiment_score + prediction_confidence) / 2
            reasoning.append("Buy: Bullish sentiment with upward price prediction")
        elif sentiment_score < -0.3 and prediction_direction == 'DOWN' and has_whale_activity:
            signal_type = "STRONG SELL"
            confidence = min(0.95, (abs(sentiment_score) + prediction_confidence + 0.7) / 3)
            reasoning.append("Strong sell: Bearish sentiment + Down prediction + Whale distribution")
        elif sentiment_score < -0.2 and prediction_direction == 'DOWN':
            signal_type = "SELL"
            confidence = (abs(sentiment_score) + prediction_confidence) / 2
            reasoning.append("Sell: Bearish sentiment with downward price prediction")
        
        signal = {
            'symbol': symbol,
            'signal_type': signal_type,
            'confidence': confidence,
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'prediction_direction': prediction_direction,
            'prediction_confidence': prediction_confidence,
            'whale_alerts_count': len(whale_alerts),
            'exchange_flow_sentiment': exchange_flows.get('sentiment', 'neutral'),
            'reasoning': ' | '.join(reasoning),
            'price_at_signal': current_price,
            'timestamp': datetime.now()
        }
        
        self._store_signal(signal)
        return signal
    
    def _store_signal(self, signal: Dict):
        try:
            with self.db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO trading_signals 
                    (symbol, signal_type, confidence, sentiment_score, prediction_direction, 
                     whale_alert, reasoning, price_at_signal, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal['symbol'], signal['signal_type'], signal['confidence'],
                    signal['sentiment_score'], signal['prediction_direction'],
                    signal['whale_alerts_count'] > 0, signal['reasoning'],
                    signal['price_at_signal'], signal['timestamp']
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing signal: {e}")