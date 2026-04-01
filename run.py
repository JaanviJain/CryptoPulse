"""Main entry point for Crypto Sentiment Terminal"""
import sys
import logging
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.data_pipeline import DataPipeline
from src.config import Config

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    """Main function"""
    print("="*60)
    print("🚀 CRYPTO SENTIMENT TERMINAL")
    print("="*60)
    print("Initializing...\n")
    
    # Initialize pipeline
    pipeline = DataPipeline()
    
    # Run initial setup
    pipeline.run_initial_setup()
    
    # Show summary
    print("\n📊 MARKET SUMMARY")
    print("-" * 60)
    summary = pipeline.db.get_market_summary()
    if not summary.empty:
        for _, row in summary.iterrows():
            symbol = row['symbol']
            price = row['price']
            change = row.get('change_24h', 0)
            change_icon = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"{symbol:10} ${price:>12,.2f}  {change_icon} {change:>+6.2f}%")
    
    print("\n✅ Setup complete!")
    print("💾 Database created at: data/crypto_data.db")
    print("\nNext steps:")
    print("  1. Add sentiment analysis with Ollama")
    print("  2. Add price prediction models")
    print("  3. Build dashboard interface")
    
if __name__ == "__main__":
    main()