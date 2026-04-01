import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from src.prediction.price_model import PricePredictor
from src.data.database import Database

logger = logging.getLogger(__name__)

class PredictionPipeline:
    """Enhanced pipeline for price prediction"""
    
    def __init__(self, db: Database, model_type: str = 'xgboost'):
        self.db = db
        self.model_type = model_type
        self.predictors = {}
        logger.info(f"Enhanced prediction pipeline initialized with {model_type}")
    
    def train_model(self, symbol: str, days_back: int = 60) -> Dict:
        """Train model with comprehensive features"""
        logger.info(f"Training enhanced model for {symbol} with {days_back} days of data")
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        df = self.db.get_historical_data(symbol, start_date, end_date, interval='15m')
        
        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return {'error': 'No data available'}
        
        logger.info(f"Retrieved {len(df)} candles for {symbol}")
        
        # Create and train predictor
        predictor = PricePredictor(model_type=self.model_type)
        result = predictor.train(df)
        
        if result.get('status') == 'success':
            self.predictors[symbol] = predictor
            logger.info(f"Enhanced model trained for {symbol}: {result}")
        
        return result
    
    def train_all_models(self, days_back: int = 60) -> Dict[str, Dict]:
        """Train models for all cryptocurrencies"""
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT']
        results = {}
        
        for symbol in symbols:
            try:
                result = self.train_model(symbol, days_back)
                results[symbol] = result
                if result.get('status') == 'success':
                    logger.info(f"✅ {symbol}: {result.get('accuracy', 0):.1%} accuracy")
            except Exception as e:
                logger.error(f"Error training {symbol}: {e}")
                results[symbol] = {'error': str(e)}
        
        return results
    
    def predict(self, symbol: str) -> Dict:
        """Make prediction using trained model"""
        if symbol not in self.predictors:
            logger.warning(f"Model not trained for {symbol}. Training now...")
            self.train_model(symbol)
        
        predictor = self.predictors.get(symbol)
        if not predictor or not predictor.is_trained:
            return {'error': 'Model not trained'}
        
        # Get recent data for prediction
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        df = self.db.get_historical_data(symbol, start_date, end_date, interval='15m')
        
        if df.empty:
            return {'error': 'No recent data'}
        
        # Make prediction
        prediction = predictor.predict(df, lookback=24)
        
        # Add metadata
        prediction['symbol'] = symbol
        prediction['timestamp'] = datetime.now()
        prediction['model_type'] = 'xgboost_enhanced'
        
        # Store prediction
        self._store_prediction(prediction)
        
        return prediction
    
    def predict_all(self) -> Dict[str, Dict]:
        """Make predictions for all cryptocurrencies"""
        predictions = {}
        
        for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT']:
            try:
                pred = self.predict(symbol)
                predictions[symbol] = pred
                logger.info(f"{symbol}: {pred.get('direction', 'ERROR')} with {pred.get('confidence', 0):.1%} confidence")
            except Exception as e:
                logger.error(f"Error predicting {symbol}: {e}")
                predictions[symbol] = {'error': str(e)}
        
        return predictions
    
    def _store_prediction(self, prediction: Dict):
        """Store prediction in database"""
        try:
            with self.db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO trading_signals 
                    (symbol, timestamp, prediction_direction, confidence, signal_type, reasoning)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    prediction['symbol'],
                    prediction['timestamp'],
                    prediction.get('direction', 'UNKNOWN'),
                    prediction.get('confidence', 0),
                    'PREDICTION_ENHANCED',
                    f"Features: {prediction.get('features_used', 0)}, Prob Up: {prediction.get('probability_up', 0):.2%}"
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
    
    def backtest_model(self, symbol: str, days_back: int = 60, test_size: float = 0.2) -> Dict:
        """Backtest enhanced model"""
        logger.info(f"Backtesting {symbol} with {days_back} days data")
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        df = self.db.get_historical_data(symbol, start_date, end_date, interval='15m')
        
        if df.empty:
            return {'error': 'No data available'}
        
        # Create predictor and backtest
        predictor = PricePredictor(model_type=self.model_type)
        results = predictor.backtest(df, test_size)
        
        return results