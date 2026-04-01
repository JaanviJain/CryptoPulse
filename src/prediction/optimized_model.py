import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except:
    LGB_AVAILABLE = False

logger = logging.getLogger(__name__)

class OptimizedPredictor:
    """Optimized model combining multiple algorithms for >55% accuracy"""
    
    def __init__(self):
        self.rf_model = None
        self.gb_model = None
        self.lgb_model = None
        self.scaler = RobustScaler()
        self.is_trained = False
        self.weights = {'rf': 0.3, 'gb': 0.3, 'lgb': 0.4}
        logger.info("Optimized predictor initialized")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create powerful features proven to work"""
        df = df.copy()
        
        # Ensure we have numeric data
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 1. Price returns (multiple timeframes)
        for period in [1, 2, 3, 4, 6, 8, 12]:
            df[f'ret_{period}'] = df['close'].pct_change(period)
            df[f'ret_log_{period}'] = np.log1p(df[f'ret_{period}'].fillna(0))
        
        # 2. Moving averages and ratios
        for period in [5, 10, 20, 50]:
            df[f'ma_{period}'] = df['close'].rolling(period).mean()
            df[f'ma_ratio_{period}'] = df['close'] / df[f'ma_{period}'] - 1
        
        # 3. Volatility features
        for period in [5, 10, 20]:
            df[f'vol_{period}'] = df['ret_1'].rolling(period).std()
        
        # 4. Volume features
        df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['vol_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
        
        # 5. Price position (high-low)
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # 6. RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 7. MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 8. Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * bb_std
        df['bb_lower'] = df['bb_mid'] - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # 9. Momentum indicators
        for period in [5, 10, 20]:
            df[f'mom_{period}'] = df['close'] - df['close'].shift(period)
            df[f'mom_ratio_{period}'] = df[f'mom_{period}'] / df['close'].shift(period)
        
        # 10. Support/Resistance
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()
        df['to_resistance'] = (df['resistance'] - df['close']) / df['close']
        df['to_support'] = (df['close'] - df['support']) / df['close']
        
        # 11. Time features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # 12. Target: next 4 periods (1 hour) direction
        df['target'] = (df['close'].shift(-4) > df['close']).astype(int)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def get_features(self) -> List[str]:
        """Return all feature names"""
        features = []
        
        # Returns
        for period in [1, 2, 3, 4, 6, 8, 12]:
            features.append(f'ret_{period}')
            features.append(f'ret_log_{period}')
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features.append(f'ma_ratio_{period}')
        
        # Volatility
        for period in [5, 10, 20]:
            features.append(f'vol_{period}')
        
        # Volume
        features.extend(['vol_ratio', 'vol_trend'])
        
        # Price position
        features.extend(['hl_ratio', 'close_position'])
        
        # Technical indicators
        features.extend(['rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_width', 'bb_position'])
        
        # Momentum
        for period in [5, 10, 20]:
            features.append(f'mom_ratio_{period}')
        
        # Support/Resistance
        features.extend(['to_resistance', 'to_support'])
        
        # Time
        features.extend(['hour', 'day_of_week'])
        
        return features
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train ensemble of models"""
        logger.info("Creating features...")
        df = self.create_features(df)
        
        if df.empty:
            return {'error': 'No data after feature creation'}
        
        # Get features
        feature_cols = self.get_features()
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features].values
        y = df['target'].values
        
        # Handle infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split data (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        logger.info("Training Random Forest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train_scaled, y_train)
        rf_acc = accuracy_score(y_test, self.rf_model.predict(X_test_scaled))
        
        # Train Gradient Boosting
        logger.info("Training Gradient Boosting...")
        self.gb_model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=5,
            random_state=42
        )
        self.gb_model.fit(X_train_scaled, y_train)
        gb_acc = accuracy_score(y_test, self.gb_model.predict(X_test_scaled))
        
        # Train LightGBM if available
        lgb_acc = 0
        if LGB_AVAILABLE:
            logger.info("Training LightGBM...")
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'verbose': -1,
                'seed': 42
            }
            train_data = lgb.Dataset(X_train_scaled, label=y_train)
            self.lgb_model = lgb.train(params, train_data, num_boost_round=300)
            lgb_acc = accuracy_score(y_test, self.lgb_model.predict(X_test_scaled) > 0.5)
        
        # Calculate ensemble predictions
        rf_pred = self.rf_model.predict_proba(X_test_scaled)[:, 1]
        gb_pred = self.gb_model.predict_proba(X_test_scaled)[:, 1]
        
        if LGB_AVAILABLE:
            lgb_pred = self.lgb_model.predict(X_test_scaled)
            ensemble_pred = (rf_pred * 0.3 + gb_pred * 0.3 + lgb_pred * 0.4) > 0.5
        else:
            ensemble_pred = (rf_pred * 0.5 + gb_pred * 0.5) > 0.5
        
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        
        self.is_trained = True
        
        logger.info(f"✅ Model training complete!")
        logger.info(f"  Random Forest: {rf_acc:.2%}")
        logger.info(f"  Gradient Boost: {gb_acc:.2%}")
        if LGB_AVAILABLE:
            logger.info(f"  LightGBM: {lgb_acc:.2%}")
        logger.info(f"  Ensemble: {ensemble_acc:.2%}")
        
        return {
            'status': 'success',
            'accuracy': ensemble_acc,
            'rf_accuracy': rf_acc,
            'gb_accuracy': gb_acc,
            'lgb_accuracy': lgb_acc if LGB_AVAILABLE else 0,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': len(available_features),
            'model_type': 'ensemble'
        }
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """Make prediction using ensemble"""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        df = self.create_features(df)
        
        if df.empty:
            return {'error': 'No data'}
        
        feature_cols = self.get_features()
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Get last 30 periods
        X_latest = df[available_features].tail(30).values
        X_latest = np.nan_to_num(X_latest, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X_latest)
        
        # Get predictions from each model
        rf_proba = self.rf_model.predict_proba(X_scaled)[:, 1]
        gb_proba = self.gb_model.predict_proba(X_scaled)[:, 1]
        
        if self.lgb_model:
            lgb_proba = self.lgb_model.predict(X_scaled)
            combined = rf_proba * 0.3 + gb_proba * 0.3 + lgb_proba * 0.4
        else:
            combined = rf_proba * 0.5 + gb_proba * 0.5
        
        # Weighted average of last 8 predictions
        weights = np.array([0.1, 0.1, 0.15, 0.15, 0.2, 0.2, 0.05, 0.05])
        final_prob = np.average(combined[-8:], weights=weights[:len(combined[-8:])])
        
        # Add trend signal
        recent_trend = df['close'].pct_change().tail(12).mean()
        final_prob = final_prob * 0.7 + (0.5 + recent_trend * 5) * 0.3
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
            'model': 'ensemble_optimized'
        }
    
    def backtest(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """Backtest optimized model"""
        df = self.create_features(df)
        
        if df.empty:
            return {'error': 'No data'}
        
        feature_cols = self.get_features()
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features].values
        y = df['target'].values
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        
        # Train Gradient Boosting
        gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
        gb.fit(X_train_scaled, y_train)
        
        # Ensemble predictions
        rf_pred = rf.predict_proba(X_test_scaled)[:, 1]
        gb_pred = gb.predict_proba(X_test_scaled)[:, 1]
        ensemble_pred = (rf_pred * 0.5 + gb_pred * 0.5) > 0.5
        
        accuracy = accuracy_score(y_test, ensemble_pred)
        precision = precision_score(y_test, ensemble_pred, zero_division=0)
        recall = recall_score(y_test, ensemble_pred, zero_division=0)
        
        # Calculate Sharpe ratio
        test_df = df.iloc[split_idx:].copy()
        test_df = test_df[:len(ensemble_pred)]
        test_df['prediction'] = ensemble_pred
        test_df['returns'] = test_df['close'].pct_change()
        test_df['signal_return'] = test_df['prediction'].shift(1) * test_df['returns']
        
        signal_returns = test_df['signal_return'].dropna()
        sharpe = np.sqrt(365) * np.mean(signal_returns) / np.std(signal_returns) if np.std(signal_returns) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'sharpe_ratio': sharpe,
            'test_samples': len(y_test),
            'train_samples': len(X_train),
            'model': 'ensemble_optimized'
        }