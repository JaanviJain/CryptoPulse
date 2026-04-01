import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnsemblePredictor:
    """Ensemble model combining multiple algorithms for better accuracy"""
    
    def __init__(self):
        self.models = []
        self.scaler = StandardScaler()
        self.is_trained = False
        self.ensemble_model = None
        logger.info("Ensemble predictor initialized")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create simple but effective features"""
        df = df.copy()
        
        # Simple price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'].fillna(0))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'ma_{period}'] = df['close'].rolling(period).mean()
            df[f'price_to_ma_{period}'] = df['close'] / df[f'ma_{period}'] - 1
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume features
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Price momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        
        # Target: next 4 periods (1 hour) direction
        df['target'] = (df['close'].shift(-4) > df['close']).astype(int)
        
        # Drop NaN
        df = df.dropna()
        
        return df
    
    def get_features(self) -> List[str]:
        return [
            'returns', 'log_returns',
            'price_to_ma_5', 'price_to_ma_10', 'price_to_ma_20', 'price_to_ma_50',
            'volatility', 'rsi', 'volume_ratio', 'momentum'
        ]
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train ensemble model"""
        logger.info("Creating features...")
        df = self.create_features(df)
        
        if df.empty:
            return {'error': 'No data'}
        
        # Get features
        feature_cols = self.get_features()
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features].values
        y = df['target'].values
        
        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split data (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create multiple models
        models = [
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)),
            ('lr', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
        ]
        
        if XGB_AVAILABLE:
            models.append(('xgb', xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)))
        
        # Train each model
        for name, model in models:
            model.fit(X_train_scaled, y_train)
        
        # Create voting classifier
        self.ensemble_model = VotingClassifier(estimators=models, voting='soft')
        self.ensemble_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.ensemble_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        
        logger.info(f"Ensemble model trained with accuracy: {accuracy:.2%}")
        
        return {
            'status': 'success',
            'accuracy': accuracy,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': len(available_features),
            'model_type': 'ensemble'
        }
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """Make prediction"""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        df = self.create_features(df)
        
        if df.empty:
            return {'error': 'No data'}
        
        feature_cols = self.get_features()
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features].tail(20).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities
        proba = self.ensemble_model.predict_proba(X_scaled)
        
        # Use weighted average of last 5 predictions
        weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
        weighted_prob = np.average(proba[-5:, 1], weights=weights)
        
        if weighted_prob > 0.55:
            direction = "UP"
            confidence = weighted_prob
        elif weighted_prob < 0.45:
            direction = "DOWN"
            confidence = 1 - weighted_prob
        else:
            direction = "SIDEWAYS"
            confidence = 0.5
        
        return {
            'direction': direction,
            'confidence': float(confidence),
            'probability_up': float(weighted_prob),
            'model': 'ensemble'
        }
    
    def backtest(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """Backtest ensemble model"""
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
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train ensemble
        models = [
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42))
        ]
        
        ensemble = VotingClassifier(estimators=models, voting='soft')
        ensemble.fit(X_train_scaled, y_train)
        
        y_pred = ensemble.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'test_samples': len(y_test),
            'train_samples': len(X_train),
            'model': 'ensemble'
        }