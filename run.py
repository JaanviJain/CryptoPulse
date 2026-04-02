import sys
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.data.data_pipeline import DataPipeline
from src.sentiment.pipeline import SentimentPipeline
from src.prediction.pipeline import PredictionPipeline
from src.signals.generator import SignalGenerator
from src.data.onchain_collector import OnChainCollector
from src.config import Config
from src.data.database import Database

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def display_dashboard(data_pipeline, sentiment_pipeline, prediction_pipeline, signal_generator, onchain_collector):
    print("\n" + "="*90)
    print("CRYPTO SENTIMENT TERMINAL - WITH AI PREDICTIONS & SIGNALS")
    print("="*90)
    
    print("\nFetching latest data...")
    
    # Get prices with progress
    with tqdm(total=3, desc="Loading data", unit="step") as pbar:
        prices = data_pipeline.db.get_current_prices()
        pbar.update(1)
        pbar.set_description("Loading prices")
        
        sentiments = sentiment_pipeline.analyze_all_cryptos()
        pbar.update(1)
        pbar.set_description("Loading sentiment")
        
        predictions = prediction_pipeline.predict_all()
        pbar.update(1)
        pbar.set_description("Loading predictions")
    
    print(f"\n{'Symbol':<10} {'Price':>12} {'24h%':>8} {'Sentiment':>12} {'Prediction':>12} {'Confidence':>10} {'Signal':>12}")
    print("-"*90)
    
    for symbol in Config.DEFAULT_SYMBOLS:
        price = prices.get(symbol, 0)
        
        summary = data_pipeline.db.get_market_summary()
        change_row = summary[summary['symbol'] == symbol]
        change = change_row['change_24h'].values[0] if not change_row.empty else 0
        
        sentiment_data = sentiments.get(symbol, {})
        sentiment = sentiment_data.get('sentiment', 'NEUTRAL')
        sentiment_score = sentiment_data.get('sentiment_score', 0)
        
        pred_data = predictions.get(symbol, {})
        pred_direction = pred_data.get('direction', 'UNKNOWN')
        pred_confidence = pred_data.get('confidence', 0)
        
        # Get on-chain data using the onchain_collector
        onchain_data = onchain_collector.check_whale_activity(symbol)
        
        signal = signal_generator.generate_signal(symbol, sentiment_data, pred_data, onchain_data, price)
        
        # Format output
        sentiment_display = f"{sentiment}"
        if sentiment_score > 0.7:
            sentiment_display = f"🔺{sentiment}"
        elif sentiment_score < -0.3:
            sentiment_display = f"🔻{sentiment}"
        
        pred_display = f"{pred_direction}"
        if pred_direction == 'UP':
            pred_display = f"📈{pred_direction}"
        elif pred_direction == 'DOWN':
            pred_display = f"📉{pred_direction}"
        
        change_icon = "+" if change > 0 else "" if change == 0 else ""
        
        print(f"{symbol:<10} ${price:>11,.2f} {change_icon}{change:>+7.2f}%  "
              f"{sentiment_display:>12}  {pred_display:>12}  "
              f"{pred_confidence:>8.1%}  {signal['signal_type']:>12}")
        
        if signal['reasoning']:
            print(f"  📝 Reason: {signal['reasoning'][:80]}")
    
    print("\n" + "="*90)
    
    # Display sentiment summary
    print("\n📊 SENTIMENT ANALYSIS SUMMARY:")
    print("-"*50)
    for symbol, data in sentiments.items():
        if data.get('articles_analyzed', 0) > 0:
            print(f"  {symbol}: {data['sentiment']} (score: {data['sentiment_score']:.3f}) - "
                  f"Analyzed {data['articles_analyzed']} articles")
            print(f"    Bullish: {data.get('bullish_count', 0)} | "
                  f"Bearish: {data.get('bearish_count', 0)} | "
                  f"Neutral: {data.get('neutral_count', 0)}")
    
    print("\n" + "="*90)

def main():
    print("="*90)
    print("CRYPTO SENTIMENT TERMINAL - COMPLETE SYSTEM")
    print("="*90)
    
    # Step 1: Initialize data pipeline
    print("\n[1/5] Initializing data pipeline...")
    data_pipeline = DataPipeline()
    
    db = Database(Config.DB_PATH)
    current_prices = db.get_current_prices()
    
    if not current_prices:
        print("  No data found. Fetching initial data...")
        data_pipeline.run_initial_setup()
    else:
        print("  Using existing data. Updating prices...")
        data_pipeline.fetch_current_prices()
        data_pipeline.fetch_daily_stats()
    
    # Step 2: Initialize on-chain collector
    print("\n[2/5] Initializing on-chain collector...")
    onchain_collector = OnChainCollector()
    
    # Step 3: Initialize sentiment analyzer
    print("\n[3/5] Initializing sentiment analyzer with Ollama...")
    sentiment_pipeline = SentimentPipeline(db, model_name="llama3.2:latest")
    
    # Step 4: Initialize prediction models
    print("\n[4/5] Initializing prediction models...")
    prediction_pipeline = PredictionPipeline(db)
    prediction_pipeline.train_all_models()
    
    # Step 5: Initialize signal generator
    print("\n[5/5] Initializing signal generator...")
    signal_generator = SignalGenerator(db)
    
    # Display dashboard with progress
    print("\n" + "="*90)
    print("LOADING COMPLETE DASHBOARD")
    print("="*90)
    
    display_dashboard(data_pipeline, sentiment_pipeline, prediction_pipeline, signal_generator, onchain_collector)
    
    print("\n✅ Terminal ready! Run backtest_model.py to see accuracy metrics.")
    print("💡 Tip: Sentiment data is stored in the database for future analysis.")

if __name__ == "__main__":
    main()