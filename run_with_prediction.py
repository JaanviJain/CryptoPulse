"""Main entry point with price prediction"""
import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.data.data_pipeline import DataPipeline
from src.sentiment.pipeline import SentimentPipeline
from src.prediction.pipeline import PredictionPipeline
from src.config import Config
from src.data.database import Database

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def display_dashboard(data_pipeline: DataPipeline, 
                      sentiment_pipeline: SentimentPipeline,
                      prediction_pipeline: PredictionPipeline):
    """Display complete dashboard with predictions"""
    print("\n" + "="*80)
    print("🎯 CRYPTO SENTIMENT TERMINAL - WITH AI PREDICTIONS")
    print("="*80)
    
    # Get current prices
    prices = data_pipeline.db.get_current_prices()
    
    # Get sentiment
    sentiments = sentiment_pipeline.analyze_all_cryptos()
    
    # Get predictions
    predictions = prediction_pipeline.predict_all()
    
    print(f"\n{'Symbol':<10} {'Price':>12} {'24h%':>8} {'Sentiment':>12} {'Prediction':>12} {'Confidence':>10} {'Signal':>12}")
    print("-" * 90)
    
    for symbol in Config.DEFAULT_SYMBOLS:
        price = prices.get(symbol, 0)
        
        # Get 24h change
        summary = data_pipeline.db.get_market_summary()
        change_row = summary[summary['symbol'] == symbol]
        change = change_row['change_24h'].values[0] if not change_row.empty else 0
        
        # Get sentiment
        sentiment_data = sentiments.get(symbol, {})
        sentiment = sentiment_data.get('sentiment', 'NEUTRAL')
        sentiment_score = sentiment_data.get('sentiment_score', 0)
        
        # Get prediction
        pred_data = predictions.get(symbol, {})
        pred_direction = pred_data.get('direction', 'UNKNOWN')
        pred_confidence = pred_data.get('confidence', 0)
        
        # Generate combined signal
        if pred_direction == "UP" and sentiment_score > 0.3:
            signal = "🟢 STRONG BUY"
        elif pred_direction == "UP":
            signal = "🟢 BUY"
        elif pred_direction == "DOWN" and sentiment_score < -0.3:
            signal = "🔴 STRONG SELL"
        elif pred_direction == "DOWN":
            signal = "🔴 SELL"
        else:
            signal = "⚪ HOLD"
        
        # Format output
        sentiment_icon = "🟢" if sentiment == "BULLISH" else "🔴" if sentiment == "BEARISH" else "🟡"
        pred_icon = "📈" if pred_direction == "UP" else "📉" if pred_direction == "DOWN" else "➡️"
        change_icon = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        
        print(f"{symbol:<10} ${price:>11,.2f} {change_icon} {change:>+6.2f}%  "
              f"{sentiment_icon} {sentiment:>8}  {pred_icon} {pred_direction:>8}  "
              f"{pred_confidence:>8.1%}  {signal}")
    
    print("\n" + "="*80)
    print("📊 PREDICTION DETAILS:")
    
    for symbol, pred in predictions.items():
        if 'error' not in pred:
            print(f"\n{symbol}:")
            print(f"  Direction: {pred.get('direction', 'N/A')}")
            print(f"  Confidence: {pred.get('confidence', 0):.1%}")
            if pred.get('probability_up'):
                print(f"  Probability Up: {pred.get('probability_up', 0):.1%}")
                print(f"  Probability Down: {pred.get('probability_down', 0):.1%}")
            print(f"  Model: {pred.get('model', 'N/A')}")
    
    print("\n" + "="*80)
    print("✅ Terminal ready! Predictions update every 15 minutes")
    print("💡 Tip: Run backtest to see model accuracy")
    print("="*80)

def main():
    """Main function"""
    print("="*80)
    print("🚀 CRYPTO SENTIMENT TERMINAL - WITH AI PRICE PREDICTION")
    print("="*80)
    
    # Initialize data pipeline
    print("\n📊 Initializing data pipeline...")
    data_pipeline = DataPipeline()
    
    # Check if we have data
    db = Database(Config.DB_PATH)
    current_prices = db.get_current_prices()
    
    if not current_prices:
        print("No data found. Fetching initial data...")
        data_pipeline.run_initial_setup()
    else:
        print("Using existing data. Updating prices...")
        data_pipeline.fetch_current_prices()
        data_pipeline.fetch_daily_stats()
    
    # Initialize sentiment pipeline
    print("\n🤖 Initializing sentiment analyzer...")
    sentiment_pipeline = SentimentPipeline(db, model_name="llama3.2:3b")
    
    # Initialize prediction pipeline
    print("\n📈 Initializing price prediction models...")
    prediction_pipeline = PredictionPipeline(db)
    
    # Train models
    print("\n🎓 Training prediction models (this may take a minute)...")
    train_results = prediction_pipeline.train_all_models(days_back=60)
    
    # Show training results
    for symbol, result in train_results.items():
        if result.get('status') == 'success':
            accuracy = result.get('accuracy', 0)
            print(f"  ✅ {symbol}: Trained - {result.get('model_type')} (Accuracy: {accuracy:.1%})")
        else:
            print(f"  ❌ {symbol}: Training failed - {result.get('error', 'Unknown error')}")
    
    # Display dashboard
    display_dashboard(data_pipeline, sentiment_pipeline, prediction_pipeline)

if __name__ == "__main__":
    main()