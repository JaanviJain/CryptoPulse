import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class SimplePredictor:
    def __init__(self):
        self.is_trained = True
        logger.info("Simple predictor initialized")
    
    def predict(self, df: pd.DataFrame) -> Dict:
        if df.empty or len(df) < 50:
            return {'direction': 'SIDEWAYS', 'confidence': 0.5, 'probability_up': 0.5}
        
        recent_prices = df['close'].tail(50).values
        current_price = recent_prices[-1]
        
        ma5 = df['close'].tail(20).mean()
        ma20 = df['close'].tail(50).mean()
        ma_signal = 1 if ma5 > ma20 else -1
        
        roc_5 = (current_price - df['close'].iloc[-6]) / df['close'].iloc[-6] if len(df) > 6 else 0
        roc_10 = (current_price - df['close'].iloc[-11]) / df['close'].iloc[-11] if len(df) > 11 else 0
        momentum_signal = 1 if roc_5 > 0 and roc_10 > 0 else -1 if roc_5 < 0 and roc_10 < 0 else 0
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).tail(14).mean()
        loss = -delta.where(delta < 0, 0).tail(14).mean()
        rs = gain / loss if loss != 0 else 1
        rsi = 100 - (100 / (1 + rs))
        rsi_signal = 1 if rsi < 30 else -1 if rsi > 70 else 0
        
        volume_avg = df['volume'].tail(20).mean()
        current_volume = df['volume'].iloc[-1]
        volume_signal = 1 if current_volume > volume_avg * 1.2 else -1 if current_volume < volume_avg * 0.8 else 0
        
        if len(recent_prices) >= 12:
            trend = np.polyfit(range(12), recent_prices[-12:], 1)[0]
            trend_signal = 1 if trend > 0 else -1 if trend < 0 else 0
        else:
            trend_signal = 0
        
        total_score = (
            ma_signal * 0.30 +
            momentum_signal * 0.25 +
            rsi_signal * 0.15 +
            volume_signal * 0.10 +
            trend_signal * 0.20
        )
        
        probability = 0.5 + (total_score * 0.25)
        probability = np.clip(probability, 0.3, 0.7)
        
        if probability > 0.55:
            direction = "UP"
            confidence = probability
        elif probability < 0.45:
            direction = "DOWN"
            confidence = 1 - probability
        else:
            direction = "SIDEWAYS"
            confidence = 0.5
        
        return {
            'direction': direction,
            'confidence': float(confidence),
            'probability_up': float(probability),
            'probability_down': float(1 - probability),
            'signals': {
                'ma_crossover': ma_signal,
                'momentum': momentum_signal,
                'rsi': rsi_signal,
                'volume': volume_signal,
                'trend': trend_signal,
                'total_score': total_score
            },
            'model': 'simple_predictor'
        }
    
    def backtest(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        if df.empty or len(df) < 200:
            return {'accuracy': 0.5, 'sharpe_ratio': 0, 'test_samples': 0}
        
        predictions = []
        actuals = []
        
        start_idx = 100
        for i in range(start_idx, len(df) - 4):
            try:
                train_df = df.iloc[:i+1]
                pred = self.predict(train_df)
                
                future_price = df['close'].iloc[i+4]
                current_price = df['close'].iloc[i]
                actual = 1 if future_price > current_price else 0
                
                predictions.append(1 if pred['direction'] == 'UP' else 0)
                actuals.append(actual)
            except Exception as e:
                continue
        
        if len(predictions) == 0:
            return {'accuracy': 0.5, 'sharpe_ratio': 0, 'test_samples': 0}
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        accuracy = np.mean(predictions == actuals)
        
        test_df = df.iloc[start_idx:].copy()
        test_df = test_df[:len(predictions)]
        test_df['prediction'] = predictions
        test_df['returns'] = test_df['close'].pct_change()
        test_df['signal_return'] = test_df['prediction'].shift(1) * test_df['returns']
        
        signal_returns = test_df['signal_return'].dropna()
        if len(signal_returns) > 0 and np.std(signal_returns) > 0:
            sharpe = np.sqrt(365) * np.mean(signal_returns) / np.std(signal_returns)
        else:
            sharpe = 0
        
        tp = np.sum((predictions == 1) & (actuals == 1))
        fp = np.sum((predictions == 1) & (actuals == 0))
        fn = np.sum((predictions == 0) & (actuals == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'sharpe_ratio': sharpe,
            'test_samples': len(predictions),
            'train_samples': start_idx,
            'model': 'simple_predictor'
        }