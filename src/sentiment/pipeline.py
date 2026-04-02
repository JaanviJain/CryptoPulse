import logging
from typing import Dict
from datetime import datetime
from tqdm import tqdm

from src.sentiment.analyzer import SentimentAnalyzer
from src.data.news_collector import NewsCollector
from src.data.onchain_collector import OnChainCollector
from src.data.database import Database

logger = logging.getLogger(__name__)

class SentimentPipeline:
    def __init__(self, db: Database, model_name: str = "llama3.2:latest"):
        self.db = db
        self.analyzer = SentimentAnalyzer(model_name)
        self.news = NewsCollector()
        self.onchain = OnChainCollector()
        
        self.crypto_map = {
            'BTCUSDT': 'bitcoin',
            'ETHUSDT': 'ethereum',
            'SOLUSDT': 'solana',
            'BNBUSDT': 'binance',
            'ADAUSDT': 'cardano'
        }
        
        self.crypto_names = {
            'BTCUSDT': 'Bitcoin',
            'ETHUSDT': 'Ethereum',
            'SOLUSDT': 'Solana',
            'BNBUSDT': 'Binance Coin',
            'ADAUSDT': 'Cardano'
        }
    
    def analyze_crypto_sentiment(self, symbol: str, show_progress: bool = True) -> Dict:
        crypto_name = self.crypto_map.get(symbol, symbol.replace('USDT', '').lower())
        display_name = self.crypto_names.get(symbol, symbol)
        
        print(f"\n[{display_name}] Fetching news articles...")
        news_articles = self.news.get_combined_news(crypto_name, limit=15)
        
        if not news_articles:
            print(f"[{display_name}] No news articles found")
            return {'sentiment': 'NEUTRAL', 'sentiment_score': 0.0, 'articles_analyzed': 0}
        
        print(f"[{display_name}] Analyzing {len(news_articles)} articles with Llama 3.2...")
        
        sentiments = []
        
        # Progress bar for article analysis
        with tqdm(total=len(news_articles), desc=f"  Analyzing {display_name}", unit="article", 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            
            for article in news_articles:
                text = f"{article['title']} {article.get('description', '')}"
                sentiment = self.analyzer.analyze_sentiment(text)
                sentiments.append(sentiment)
                
                article['sentiment_score'] = sentiment['score']
                article['sentiment_label'] = sentiment['sentiment']
                article['analyzed_at'] = datetime.now()
                self.db.insert_news_article(article)
                
                pbar.update(1)
                pbar.set_postfix({
                    'Current': f"{sentiment['sentiment'][:4]} ({sentiment['score']:.2f})"
                })
        
        if not sentiments:
            return {'sentiment': 'NEUTRAL', 'sentiment_score': 0.0, 'articles_analyzed': 0}
        
        avg_score = sum(s['score'] for s in sentiments) / len(sentiments)
        bullish_count = sum(1 for s in sentiments if s['sentiment'] == 'BULLISH')
        bearish_count = sum(1 for s in sentiments if s['sentiment'] == 'BEARISH')
        neutral_count = sum(1 for s in sentiments if s['sentiment'] == 'NEUTRAL')
        
        if avg_score > 0.2:
            overall = 'BULLISH'
        elif avg_score < -0.2:
            overall = 'BEARISH'
        else:
            overall = 'NEUTRAL'
        
        # Display summary
        print(f"\n  [{display_name}] Sentiment Summary:")
        print(f"    Overall: {overall} (score: {avg_score:.3f})")
        print(f"    Bullish: {bullish_count} | Bearish: {bearish_count} | Neutral: {neutral_count}")
        
        return {
            'symbol': symbol,
            'sentiment': overall,
            'sentiment_score': avg_score,
            'articles_analyzed': len(sentiments),
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'timestamp': datetime.now()
        }
    
    def analyze_all_cryptos(self) -> Dict[str, Dict]:
        results = {}
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT']
        
        print("\n" + "="*70)
        print("STARTING SENTIMENT ANALYSIS FOR ALL CRYPTOCURRENCIES")
        print("="*70)
        
        # Overall progress bar for cryptocurrencies
        with tqdm(total=len(symbols), desc="Overall Progress", unit="crypto",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} cryptocurrencies completed [{elapsed}<{remaining}]",
                  position=0) as crypto_pbar:
            
            for symbol in symbols:
                try:
                    result = self.analyze_crypto_sentiment(symbol, show_progress=True)
                    results[symbol] = result
                    logger.info(f"{symbol}: {result['sentiment']} (score: {result['sentiment_score']:.2f})")
                    crypto_pbar.update(1)
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    results[symbol] = {'sentiment': 'ERROR', 'sentiment_score': 0}
                    crypto_pbar.update(1)
        
        print("\n" + "="*70)
        print("SENTIMENT ANALYSIS COMPLETE")
        print("="*70)
        
        return results