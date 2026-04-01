"""Main entry point with sentiment analysis"""
import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.data.data_pipeline import DataPipeline
from src.sentiment.pipeline import SentimentPipeline
from src.config import Config
from src.data.database import Database

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def display_sentiment_dashboard(pipeline: DataPipeline, sentiment_pipeline: SentimentPipeline):
    """Display sentiment analysis dashboard"""
    print("\n" + "="*70)
    print("🎯 CRYPTO SENTIMENT TERMINAL - LIVE DASHBOARD")
    print("="*70)
    
    # Get current prices
    prices = pipeline.db.get_current_prices()
    
    # Get sentiment for all cryptos
    sentiments = sentiment_pipeline.analyze_all_cryptos()
    
    print(f"\n{'Symbol':<10} {'Price':>12} {'24h Change':>12} {'Sentiment':>10} {'Score':>8} {'Signal':>10}")
    print("-" * 70)
    
    for symbol in Config.DEFAULT_SYMBOLS:
        price = prices.get(symbol, 0)
        
        # Get market summary for 24h change
        summary = pipeline.db.get_market_summary()
        change_row = summary[summary['symbol'] == symbol]
        change = change_row['change_24h'].values[0] if not change_row.empty else 0
        
        # Get sentiment
        sentiment_data = sentiments.get(symbol, {})
        sentiment = sentiment_data.get('sentiment', 'NEUTRAL')
        score = sentiment_data.get('sentiment_score', 0)
        
        # Determine signal
        if sentiment == 'BULLISH' and change > 0:
            signal = "🟢 STRONG BUY"
        elif sentiment == 'BULLISH':
            signal = "🟢 BUY"
        elif sentiment == 'BEARISH' and change < 0:
            signal = "🔴 STRONG SELL"
        elif sentiment == 'BEARISH':
            signal = "🔴 SELL"
        else:
            signal = "⚪ HOLD"
        
        # Color coding for sentiment
        sentiment_icon = "🟢" if sentiment == "BULLISH" else "🔴" if sentiment == "BEARISH" else "🟡"
        change_icon = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        
        print(f"{symbol:<10} ${price:>11,.2f} {change_icon} {change:>+8.2f}%  {sentiment_icon} {sentiment:>8} {score:>+7.2f}  {signal}")
    
    print("\n" + "="*70)
    print("📊 Sentiment Analysis Details:")
    
    for symbol, data in sentiments.items():
        if data.get('total_analyzed', 0) > 0:
            print(f"\n{symbol}:")
            print(f"  Analyzed {data['total_analyzed']} posts")
            print(f"  Bullish: {data.get('bullish_posts', 0)} ({data.get('bullish_percentage', 0):.1f}%)")
            print(f"  Bearish: {data.get('bearish_posts', 0)}")
            print(f"  Neutral: {data.get('neutral_posts', 0)}")

def main():
    """Main function"""
    print("="*70)
    print("🚀 CRYPTO SENTIMENT TERMINAL - WITH LLM ANALYSIS")
    print("="*70)
    
    # Initialize data pipeline
    print("\n📊 Initializing data pipeline...")
    data_pipeline = DataPipeline()
    
    # Check if we need to fetch fresh data
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
    print("\n🤖 Initializing sentiment analyzer with Ollama...")
    sentiment_pipeline = SentimentPipeline(db, model_name="llama3.2:3b")
    
    # Display dashboard
    display_sentiment_dashboard(data_pipeline, sentiment_pipeline)
    
    print("\n" + "="*70)
    print("✅ Terminal ready!")
    print("💡 Tip: Run this script periodically to update sentiment analysis")
    print("📝 Sentiment data is stored in the database for backtesting")
    print("="*70)

if __name__ == "__main__":
    main()