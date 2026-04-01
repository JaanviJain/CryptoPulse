"""Test Reddit collector specifically"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.reddit_collector import RedditCollector

def test_reddit():
    print("="*60)
    print("Testing Reddit Data Collection")
    print("="*60)
    
    reddit = RedditCollector()
    
    # Test get_hot_posts
    print("\n1. Testing get_hot_posts...")
    posts = reddit.get_hot_posts('cryptocurrency', limit=5)
    print(f"✅ Got {len(posts)} posts")
    if posts:
        print(f"Sample post: {posts[0]['title'][:100]}")
    
    # Test search_crypto
    print("\n2. Testing search_crypto...")
    btc_posts = reddit.search_crypto('bitcoin', limit=3)
    print(f"✅ Found {len(btc_posts)} posts about Bitcoin")
    if btc_posts:
        print(f"Sample: {btc_posts[0]['title'][:100]}")
    
    # Test get_sentiment_ready_posts
    print("\n3. Testing get_sentiment_ready_posts...")
    sentiment_posts = reddit.get_sentiment_ready_posts('BTCUSDT')
    print(f"✅ Got {len(sentiment_posts)} posts ready for sentiment analysis")
    
    print("\n✅ Reddit collector test completed!")

if __name__ == "__main__":
    test_reddit()