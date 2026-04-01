import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not installed. Install with: pip install tensorflow")

logger = logging.getLogger(__name__)

class LSTMPredictor:
    """LSTM-based price prediction model optimized for CPU"""
    
    def __init__(self, sequence_length: int = 60):
        """
        Initialize LSTM predictor
        
        Args:
            sequence_length: Number of past time steps to use for prediction
        """
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.feature_columns = []
        
        logger.info(f"LSTM predictor initialized with sequence length: {sequence_length}")
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)
    
    def create_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Create features for LSTM model"""
        df = df.copy()
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'].fillna(0))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'price_to_ma_{period}'] = (df['close'] - df[f'ma_{period}']) / df[f'ma_{period}']
        
        # Exponential moving averages
        for period in [12, 26]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Price momentum
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        
        # Target: next 4 periods (1 hour) direction
        df['target'] = (df['close'].shift(-4) > df['close']).astype(int)
        
        # Select features
        feature_cols = [
            'close', 'volume',
            'returns', 'log_returns',
            'price_to_ma_5', 'price_to_ma_10', 'price_to_ma_20', 'price_to_ma_50',
            'macd', 'macd_signal', 'macd_hist',
            'rsi',
            'bb_width', 'bb_position',
            'volume_ratio',
            'volatility',
            'momentum_5', 'momentum_10'
        ]
        
        # Keep only existing columns
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Fill NaN values
        df[available_features] = df[available_features].fillna(method='ffill').fillna(0)
        
        # Prepare data
        data = df[available_features].values
        target = df['target'].values
        
        return data, target, available_features
    
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build LSTM model architecture optimized for CPU"""
        model = Sequential([
            # First LSTM layer with dropout
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            
            # Second LSTM layer
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            
            # Third LSTM layer
            LSTM(32),
            Dropout(0.2),
            
            # Dense layers
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 32) -> Dict:
        """Train LSTM model"""
        if not TF_AVAILABLE:
            return {'error': 'TensorFlow not installed'}
        
        logger.info("Creating features for LSTM...")
        
        # Create features
        data, target, features = self.create_features(df)
        
        if len(data) < self.sequence_length + 100:
            return {'error': f'Insufficient data: {len(data)} rows, need at least {self.sequence_length + 100}'}
        
        # Scale features
        data_scaled = self.scaler.fit_transform(data)
        self.feature_columns = features
        
        # Create sequences
        X, y = self.create_sequences(data_scaled, target)
        
        # Split data (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)
        
        # Callbacks for better training
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.00001, verbose=1)
        ]
        
        # Train model
        logger.info("Training LSTM model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test data
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        self.is_trained = True
        
        logger.info(f"✅ LSTM model trained!")
        logger.info(f"  Accuracy: {accuracy:.2%}")
        logger.info(f"  Precision: {precision:.2%}")
        logger.info(f"  Recall: {recall:.2%}")
        logger.info(f"  F1 Score: {f1:.2%}")
        
        return {
            'status': 'success',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': len(features),
            'model_type': 'lstm'
        }
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """Make prediction using trained LSTM model"""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        # Create features
        data, target, features = self.create_features(df)
        
        if len(data) < self.sequence_length:
            return {'error': f'Insufficient data for prediction. Need at least {self.sequence_length} rows'}
        
        # Scale features
        data_scaled = self.scaler.transform(data)
        
        # Get last sequence
        last_sequence = data_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        
        # Predict
        proba = self.model.predict(last_sequence, verbose=0)[0][0]
        
        # Get recent trend (last 12 periods)
        recent_trend = df['close'].pct_change().tail(12).mean()
        
        # Combine with trend
        final_prob = proba * 0.7 + (0.5 + recent_trend * 5) * 0.3
        final_prob = np.clip(final_prob, 0, 1)
        
        # Determine direction
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
            'lstm_probability': float(proba),
            'trend_signal': float(recent_trend),
            'model': 'lstm'
        }
    
    def backtest(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """Backtest LSTM model"""
        if not TF_AVAILABLE:
            return {'error': 'TensorFlow not available'}
        
        # Create features
        data, target, features = self.create_features(df)
        
        if len(data) < self.sequence_length + 100:
            return {'error': 'Insufficient data for backtesting'}
        
        # Scale features
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Create sequences
        X, y = self.create_sequences(data_scaled, target)
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build and train model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = self.build_model(input_shape)
        
        callbacks = [EarlyStopping(patience=10, restore_best_weights=True)]
        
        model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        # Predict
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Calculate Sharpe ratio
        test_df = df.iloc[split_idx + self.sequence_length:].copy()
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
            'model': 'lstm'
        }