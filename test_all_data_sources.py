"""Test all data collection sources"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.reddit_collector import RedditCollector
from src.data.news_collector import NewsCollector
from src.data.onchain_collector import OnChainCollector
from src.sentiment.analyzer import SentimentAnalyzer
from src.sentiment.pipeline import SentimentPipeline
from src.data.database import Database
from src.config import Config

def test_reddit():
    print("\n" + "="*60)
    print("Testing Reddit Data Collection")
    print("="*60)
    
    reddit = RedditCollector()
    
    # Test hot posts
    posts = reddit.get_hot_posts('cryptocurrency', limit=5)
    print(f"✅ Got {len(posts)} posts")
    if posts:
        print(f"Sample: {posts[0]['title'][:100]}")
    
    # Test search
    btc_posts = reddit.search_crypto('bitcoin', limit=5)
    print(f"✅ Found {len(btc_posts)} posts about Bitcoin")

def test_news():
    print("\n" + "="*60)
    print("Testing News API Collection")
    print("="*60)
    
    news = NewsCollector()
    
    # Test crypto news
    articles = news.get_crypto_news('bitcoin', days_back=1, page_size=5)
    print(f"✅ Got {len(articles)} news articles")
    if articles:
        print(f"Sample: {articles[0]['title'][:100]}")

def test_onchain():
    print("\n" + "="*60)
    print("Testing On-Chain Data Collection")
    print("="*60)
    
    onchain = OnChainCollector()
    
    # Test whale activity
    whale_data = onchain.check_whale_activity('BTCUSDT')
    print(f"✅ Whale alert: {whale_data.get('whale_alert', False)}")
    print(f"✅ Whale amount: ${whale_data.get('amount_usd', 0):,.0f}")
    
    # Test exchange flows
    flows = onchain.get_exchange_flows()
    print(f"✅ Exchange net flow: ${flows.get('net_flow', 0):,.0f}")

def test_complete_pipeline():
    print("\n" + "="*60)
    print("Testing Complete Sentiment Pipeline")
    print("="*60)
    
    db = Database(Config.DB_PATH)
    pipeline = SentimentPipeline(db)
    
    # Analyze all cryptos
    results = pipeline.analyze_all_cryptos()
    
    print("\nSentiment Summary:")
    print("-" * 50)
    for symbol, result in results.items():
        print(f"{symbol}: {result['sentiment']} (score: {result['sentiment_score']:.2f})")
        if result.get('whale_alert'):
            print(f"  🐋 WHALE ALERT DETECTED!")

if __name__ == "__main__":
    print("🚀 Testing All Data Sources")
    print("Make sure Ollama is running!\n")
    
    test_reddit()
    test_news()
    test_onchain()
    test_complete_pipeline()
    
    print("\n✅ All tests completed!")