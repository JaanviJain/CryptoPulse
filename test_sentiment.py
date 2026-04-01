"""Test sentiment analysis with Ollama"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.sentiment.analyzer import SentimentAnalyzer
from src.sentiment.pipeline import SentimentPipeline
from src.data.database import Database
from src.config import Config

def test_single_analysis():
    """Test single text sentiment analysis"""
    print("="*60)
    print("Testing Single Text Sentiment Analysis")
    print("="*60)
    
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "Bitcoin is exploding! New all-time high incoming! 🚀",
        "Crypto market is crashing, sell everything!",
        "Ethereum trading sideways, no clear direction",
        "Solana has strong fundamentals and growing adoption"
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        result = analyzer.analyze_sentiment(text)
        print(f"Sentiment: {result['sentiment']}")
        print(f"Score: {result['score']:.2f}")
        print(f"Confidence: {result['confidence']}")
        print(f"Explanation: {result['explanation']}")

def test_crypto_sentiment():
    """Test sentiment for specific cryptocurrencies"""
    print("\n" + "="*60)
    print("Testing Cryptocurrency Sentiment Analysis")
    print("="*60)
    
    # Initialize database and pipeline
    db = Database(Config.DB_PATH)
    pipeline = SentimentPipeline(db)
    
    # Analyze sentiment for BTC
    result = pipeline.analyze_crypto_sentiment('BTCUSDT')
    
    print(f"\nBitcoin Sentiment Analysis:")
    print(f"Overall Sentiment: {result['sentiment']}")
    print(f"Sentiment Score: {result['sentiment_score']:.3f}")
    print(f"Bullish Posts: {result['bullish_posts']}")
    print(f"Bearish Posts: {result['bearish_posts']}")
    print(f"Neutral Posts: {result['neutral_posts']}")
    print(f"Total Analyzed: {result['total_analyzed']}")
    
    return result

def test_all_cryptos():
    """Test sentiment for all cryptocurrencies"""
    print("\n" + "="*60)
    print("Testing All Cryptocurrencies Sentiment")
    print("="*60)
    
    db = Database(Config.DB_PATH)
    pipeline = SentimentPipeline(db)
    
    results = pipeline.analyze_all_cryptos()
    
    print("\nSentiment Summary:")
    print("-" * 40)
    for symbol, result in results.items():
        sentiment = result['sentiment']
        score = result['sentiment_score']
        emoji = "🟢" if sentiment == "BULLISH" else "🔴" if sentiment == "BEARISH" else "🟡"
        print(f"{symbol:10} {emoji} {sentiment:8} (score: {score:.2f})")
    
    return results

if __name__ == "__main__":
    print("\n🚀 Testing Sentiment Analysis with Ollama")
    print("Make sure Ollama is running with llama3.2:3b\n")
    
    # Test 1: Single text analysis
    test_single_analysis()
    
    # Test 2: Cryptocurrency sentiment
    test_crypto_sentiment()
    
    # Test 3: All cryptocurrencies
    test_all_cryptos()
    
    print("\n✅ Sentiment analysis tests completed!")