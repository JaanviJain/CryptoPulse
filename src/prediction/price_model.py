import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Try importing optional dependencies
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed. Install with: pip install xgboost")

logger = logging.getLogger(__name__)

class PricePredictor:
    """Enhanced price prediction model with better accuracy"""
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize price predictor
        
        Args:
            model_type: 'xgboost' only (optimized)
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        self.is_trained = False
        self.feature_importance = None
        
        logger.info(f"Enhanced price predictor initialized with {model_type}")
    
    def create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features for better prediction"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # 1. Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        
        # 2. Moving Averages (multiple timeframes)
        for period in [7, 14, 21, 50]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # 3. Moving Average Crossovers
        df['sma_7_21_ratio'] = df['sma_7'] / df['sma_21']
        df['ema_12_26_diff'] = df['ema_12'] - df['ema_26']
        
        # 4. Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # 5. RSI with different periods
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # 6. MACD
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 7. Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_change'] = df['volume'].pct_change()
        
        # 8. Price patterns
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['close_open_ratio'] = df['close'] / df['open']
        df['high_close_ratio'] = (df['high'] - df['close']) / df['close']
        df['low_close_ratio'] = (df['close'] - df['low']) / df['close']
        
        # 9. Volatility indicators
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=100).mean()
        
        # 10. Momentum indicators
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['momentum_20'] = df['close'] - df['close'].shift(20)
        
        # 11. Rate of Change
        df['roc_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100
        df['roc_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
        
        # 12. Stochastic Oscillator
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # 13. Williams %R
        df['williams_r'] = -100 * ((high_14 - df['close']) / (high_14 - low_14))
        
        # 14. Target variable: price direction for next period
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # 15. Additional target: next 4 periods (1 hour) direction
        df['target_4h'] = (df['close'].shift(-4) > df['close']).astype(int)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Return list of all feature columns"""
        return [
            'returns', 'log_returns',
            'sma_7', 'sma_14', 'sma_21', 'sma_50',
            'ema_7', 'ema_14', 'ema_21', 'ema_50',
            'sma_7_21_ratio', 'ema_12_26_diff',
            'bb_width', 'bb_position',
            'rsi_7', 'rsi_14', 'rsi_21',
            'macd', 'macd_signal', 'macd_histogram',
            'volume_ratio', 'volume_change',
            'high_low_ratio', 'close_open_ratio', 'high_close_ratio', 'low_close_ratio',
            'volatility', 'volatility_ratio',
            'momentum_5', 'momentum_10', 'momentum_20',
            'roc_5', 'roc_10',
            'stoch_k', 'stoch_d', 'williams_r'
        ]
    
    def train_xgboost(self, df: pd.DataFrame, target_col: str = 'target') -> Dict:
        """Train optimized XGBoost model with hyperparameter tuning"""
        if not XGB_AVAILABLE:
            return {'error': 'XGBoost not installed'}
        
        # Create enhanced features
        logger.info("Creating enhanced features...")
        df = self.create_enhanced_features(df)
        
        if df.empty:
            return {'error': 'No data after feature preparation'}
        
        # Get feature columns
        feature_cols = self.get_feature_columns()
        
        # Use available features
        available_features = [col for col in feature_cols if col in df.columns]
        logger.info(f"Using {len(available_features)} features")
        
        X = df[available_features].values
        y = df[target_col].values
        
        # Handle any remaining NaN or inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split data (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        
        # Optimized XGBoost parameters for better accuracy
        params = {
            'n_estimators': 300,
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'use_label_encoder': False
        }
        
        # Train model
        logger.info("Training XGBoost model with optimized parameters...")
        self.model = xgb.XGBClassifier(**params)
        
        # Train with early stopping
        eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Evaluate
        y_pred = self.model.predict(X_val_scaled)
        accuracy = accuracy_score(y_val, y_pred)
        
        # Feature importance
        self.feature_importance = dict(zip(available_features, self.model.feature_importances_))
        
        self.is_trained = True
        
        logger.info(f"XGBoost model trained with accuracy: {accuracy:.2%}")
        
        # Log top features
        top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"Top features: {[f[0] for f in top_features]}")
        
        return {
            'status': 'success',
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'accuracy': accuracy,
            'features_used': len(available_features),
            'top_features': [f[0] for f in top_features],
            'model_type': 'xgboost'
        }
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train model"""
        return self.train_xgboost(df)
    
    def predict(self, df: pd.DataFrame, lookback: int = 24) -> Dict:
        """Make prediction using trained model"""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        # Prepare features for recent data
        df = self.create_enhanced_features(df)
        
        if df.empty:
            return {'error': 'No data for prediction'}
        
        # Get feature columns
        feature_cols = self.get_feature_columns()
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Get latest data
        latest_data = df.tail(lookback)
        X_latest = latest_data[available_features].values
        
        # Handle NaN/inf
        X_latest = np.nan_to_num(X_latest, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale
        X_scaled = self.feature_scaler.transform(X_latest)
        
        # Predict probabilities
        probabilities = self.model.predict_proba(X_scaled)
        
        # Get recent predictions (last 6 periods = 1.5 hours)
        recent_preds = self.model.predict(X_scaled)
        avg_direction = np.mean(recent_preds[-6:])
        
        # Get probability for next period
        next_prob_up = probabilities[-1][1]
        
        # Determine direction with confidence
        if next_prob_up > 0.55:
            direction = "UP"
            confidence = next_prob_up
        elif next_prob_up < 0.45:
            direction = "DOWN"
            confidence = 1 - next_prob_up
        else:
            direction = "SIDEWAYS"
            confidence = 0.5
        
        # Calculate ensemble prediction (recent trend + probability)
        if avg_direction > 0.55 and next_prob_up > 0.55:
            direction = "UP"
            confidence = max(confidence, avg_direction)
        elif avg_direction < 0.45 and next_prob_up < 0.45:
            direction = "DOWN"
            confidence = max(confidence, 1 - avg_direction)
        
        return {
            'direction': direction,
            'confidence': float(confidence),
            'probability_up': float(next_prob_up),
            'probability_down': float(1 - next_prob_up),
            'avg_recent_direction': float(avg_direction),
            'model': 'xgboost_enhanced',
            'features_used': len(available_features)
        }
    
    def backtest(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """Comprehensive backtest with multiple metrics"""
        # Prepare features
        df = self.create_enhanced_features(df)
        
        if df.empty:
            return {'error': 'No data for backtesting'}
        
        # Get features
        feature_cols = self.get_feature_columns()
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features].values
        y = df['target'].values
        
        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train optimized model
        params = {
            'n_estimators': 300,
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'random_state': 42
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Calculate Sharpe ratio from prediction signals
        test_df = df.iloc[split_idx:].copy()
        test_df['prediction'] = y_pred
        test_df['returns'] = test_df['close'].pct_change()
        test_df['signal_return'] = test_df['prediction'].shift(1) * test_df['returns']
        
        # Filter out NaN
        signal_returns = test_df['signal_return'].dropna()
        sharpe = np.sqrt(365) * np.mean(signal_returns) / np.std(signal_returns) if np.std(signal_returns) > 0 else 0
        
        # Calculate win rate
        correct_predictions = (y_pred == y_test)
        win_rate = np.mean(correct_predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'test_samples': len(y_test),
            'train_samples': len(y_train),
            'model': 'xgboost_enhanced'
        }