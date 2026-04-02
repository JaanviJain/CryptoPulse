import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.news_collector import NewsCollector
from src.data.onchain_collector import OnChainCollector

def test_apis():
    print("="*60)
    print("Testing API Connections")
    print("="*60)
    
    print("\n1. Testing News API...")
    news = NewsCollector()
    articles = news.get_combined_news('bitcoin', limit=5)
    print(f"Got {len(articles)} news articles")
    if articles:
        print(f"   Sample: {articles[0]['title'][:80]}...")
    
    print("\n2. Testing On-Chain Data...")
    onchain = OnChainCollector()
    
    eth_whales = onchain.get_eth_whale_transactions(min_value_eth=1000, limit=3)
    print(f"Found {len(eth_whales)} ETH whale transactions")
    
    flows = onchain.get_exchange_flows()
    print(f"Exchange Net Flow: ${flows.get('net_flow', 0):,.0f}")
    
    print("\n All API tests completed!")

if __name__ == "__main__":
    test_apis()