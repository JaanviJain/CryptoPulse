import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("LightGBM not installed. Install with: pip install lightgbm")

logger = logging.getLogger(__name__)

class LightGBMPredictor:
    """LightGBM model for fast and accurate crypto prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = None
        logger.info("LightGBM predictor initialized")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create efficient features for LightGBM"""
        df = df.copy()
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'].fillna(0))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'ma_{period}'] = df['close'].rolling(period).mean()
            df[f'price_ma_{period}_ratio'] = df['close'] / df[f'ma_{period}'] - 1
        
        # Price momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Target: next 4 periods (1 hour)
        df['target'] = (df['close'].shift(-4) > df['close']).astype(int)
        
        # Drop NaN
        df = df.dropna()
        
        return df
    
    def get_features(self) -> List[str]:
        """Return feature list"""
        return [
            'returns', 'log_returns',
            'price_ma_5_ratio', 'price_ma_10_ratio', 'price_ma_20_ratio', 'price_ma_50_ratio',
            'momentum_5', 'momentum_10', 'momentum_20',
            'rsi',
            'volume_ratio',
            'volatility',
            'macd', 'macd_signal'
        ]
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train LightGBM model"""
        if not LGB_AVAILABLE:
            return {'error': 'LightGBM not installed'}
        
        logger.info("Creating features...")
        df = self.create_features(df)
        
        if df.empty:
            return {'error': 'No data after feature creation'}
        
        # Get features
        feature_cols = self.get_features()
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features].values
        y = df['target'].values
        
        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split data (80/20 with time order)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # LightGBM parameters optimized for accuracy
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'num_threads': 4,
            'seed': 42
        }
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        
        # Train model (simplified, no callbacks to avoid version issues)
        logger.info("Training LightGBM model...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data]
        )
        
        # Predict on test data
        y_pred_proba = self.model.predict(X_test_scaled)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Feature importance
        self.feature_importance = dict(zip(available_features, self.model.feature_importance()))
        self.is_trained = True
        
        logger.info(f"✅ LightGBM model trained!")
        logger.info(f"  Accuracy: {accuracy:.2%}")
        logger.info(f"  Precision: {precision:.2%}")
        logger.info(f"  Recall: {recall:.2%}")
        logger.info(f"  F1 Score: {f1:.2%}")
        
        # Log top features
        top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"Top features: {[f[0] for f in top_features]}")
        
        return {
            'status': 'success',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': len(available_features),
            'top_features': [f[0] for f in top_features],
            'model_type': 'lightgbm'
        }
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """Make prediction"""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        df = self.create_features(df)
        
        if df.empty:
            return {'error': 'No data for prediction'}
        
        feature_cols = self.get_features()
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Get last 30 periods
        X_latest = df[available_features].tail(30).values
        X_latest = np.nan_to_num(X_latest, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X_latest)
        
        # Get probabilities
        probs = self.model.predict(X_scaled)
        
        # Weighted average of last 12 predictions
        weights = np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05])
        weighted_prob = np.average(probs[-12:], weights=weights[:len(probs[-12:])])
        
        # Get recent trend
        recent_trend = df['close'].pct_change().tail(12).mean()
        
        # Combine signals
        final_prob = weighted_prob * 0.7 + (0.5 + recent_trend * 5) * 0.3
        final_prob = np.clip(final_prob, 0, 1)
        
        if final_prob > 0.55:
            direction = "UP"
            confidence = final_prob
        elif final_prob < 0.45:
            direction = "DOWN"
            confidence = 1 - final_prob
        else:
            direction = "SIDEWAYS"
            confidence = 0.5
        
        return {
            'direction': direction,
            'confidence': float(confidence),
            'probability_up': float(final_prob),
            'probability_down': float(1 - final_prob),
            'weighted_signal': float(weighted_prob),
            'trend_signal': float(recent_trend),
            'model': 'lightgbm'
        }
    
    def backtest(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """Backtest LightGBM model"""
        df = self.create_features(df)
        
        if df.empty:
            return {'error': 'No data for backtesting'}
        
        feature_cols = self.get_features()
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features].values
        y = df['target'].values
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model (simplified)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'verbose': -1,
            'seed': 42
        }
        
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        model = lgb.train(params, train_data, num_boost_round=500)
        
        # Predict
        y_pred_proba = model.predict(X_test_scaled)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Calculate Sharpe ratio
        test_df = df.iloc[split_idx:].copy()
        test_df = test_df[:len(y_pred)]
        test_df['prediction'] = y_pred
        test_df['returns'] = test_df['close'].pct_change()
        test_df['signal_return'] = test_df['prediction'].shift(1) * test_df['returns']
        
        signal_returns = test_df['signal_return'].dropna()
        sharpe = np.sqrt(365) * np.mean(signal_returns) / np.std(signal_returns) if np.std(signal_returns) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'sharpe_ratio': sharpe,
            'test_samples': len(y_test),
            'train_samples': len(X_train),
            'model': 'lightgbm'
        }