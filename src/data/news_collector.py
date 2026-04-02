import requests
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class NewsCollector:
    def __init__(self):
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.cryptopanic_key = os.getenv('CRYPTOPANIC_API_KEY')
        logger.info("News collector initialized")
    
    def get_news_from_newsapi(self, query: str = 'cryptocurrency', days_back: int = 1) -> List[Dict]:
        if not self.news_api_key:
            return self._get_mock_news(query)
        
        articles = []
        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'relevancy',
                'apiKey': self.news_api_key,
                'pageSize': 30,
                'language': 'en'
            }
            response = requests.get('https://newsapi.org/v2/everything', params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for article in data.get('articles', []):
                    articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', '')[:500],
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('name', 'Unknown'),
                        'published_at': article.get('publishedAt'),
                        'crypto_mentioned': query,
                        'source_api': 'newsapi'
                    })
                logger.info(f"NewsAPI: Fetched {len(articles)} articles")
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
        return articles if articles else self._get_mock_news(query)
    
    def get_news_from_cryptopanic(self, query: str = 'BTC', limit: int = 30) -> List[Dict]:
        if not self.cryptopanic_key:
            return []
        
        articles = []
        try:
            params = {'auth_token': self.cryptopanic_key, 'public': 'true', 'currencies': query}
            response = requests.get('https://cryptopanic.com/api/v1/posts/', params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for post in data.get('results', [])[:limit]:
                    articles.append({
                        'title': post.get('title', ''),
                        'description': post.get('metadata', {}).get('description', ''),
                        'content': post.get('metadata', {}).get('description', '')[:500],
                        'url': post.get('url', ''),
                        'source': post.get('source', {}).get('title', 'Unknown'),
                        'published_at': post.get('published_at'),
                        'crypto_mentioned': query,
                        'source_api': 'cryptopanic'
                    })
                logger.info(f"CryptoPanic: Fetched {len(articles)} articles")
        except Exception as e:
            logger.error(f"CryptoPanic error: {e}")
        return articles
    
    def get_combined_news(self, crypto_name: str = 'bitcoin', limit: int = 20) -> List[Dict]:
        all_articles = []
        newsapi_articles = self.get_news_from_newsapi(crypto_name)
        all_articles.extend(newsapi_articles)
        
        cryptopanic_articles = self.get_news_from_cryptopanic(crypto_name.upper())
        all_articles.extend(cryptopanic_articles)
        
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article['url'] not in seen_urls:
                seen_urls.add(article['url'])
                unique_articles.append(article)
        
        return unique_articles[:limit]
    
    def _get_mock_news(self, query: str) -> List[Dict]:
        mock_articles = {
            'bitcoin': [
                {'title': 'Bitcoin Surges Past $70,000 as Institutional Interest Grows', 'description': 'Major financial institutions are increasing Bitcoin holdings.', 'content': 'BlackRock and Fidelity have increased Bitcoin ETF holdings.', 'source': 'CryptoNews', 'crypto_mentioned': 'bitcoin', 'url': 'https://example.com/1'},
                {'title': 'Bitcoin Mining Difficulty Reaches All-Time High', 'description': 'Network security strengthens.', 'content': 'Bitcoin mining difficulty has reached a new ATH.', 'source': 'CoinDesk', 'crypto_mentioned': 'bitcoin', 'url': 'https://example.com/2'}
            ],
            'ethereum': [
                {'title': 'Ethereum Gas Fees Drop to 6-Month Low', 'description': 'Layer 2 solutions reducing costs.', 'content': 'Ethereum gas fees have dropped significantly.', 'source': 'The Block', 'crypto_mentioned': 'ethereum', 'url': 'https://example.com/3'}
            ],
            'solana': [
                {'title': 'Solana Network Activity Hits All-Time High', 'description': 'Daily transactions surpass records.', 'content': 'Solana sees strong adoption.', 'source': 'CryptoBriefing', 'crypto_mentioned': 'solana', 'url': 'https://example.com/4'}
            ]
        }
        return mock_articles.get(query, [{'title': f'{query.upper()} Shows Strong Momentum', 'description': 'Positive sentiment', 'content': 'Market showing strength', 'source': 'CryptoDaily', 'crypto_mentioned': query, 'url': f'https://example.com/{query}'}])