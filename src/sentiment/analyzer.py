import ollama
import pandas as pd
import logging
from typing import List, Dict, Optional
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """LLM-based sentiment analyzer for crypto news and social media"""
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        """
        Initialize sentiment analyzer with Ollama model
        
        Args:
            model_name: Ollama model name (llama3.2:3b, mistral, etc.)
        """
        self.model_name = model_name
        self.available_models = self._check_available_models()
        
        if model_name not in self.available_models:
            logger.warning(f"Model {model_name} not found. Available: {self.available_models}")
            if self.available_models:
                self.model_name = self.available_models[0]
                logger.info(f"Using {self.model_name} instead")
        
        logger.info(f"Sentiment analyzer initialized with model: {self.model_name}")
    
    def _check_available_models(self) -> List[str]:
        """Check which Ollama models are available"""
        try:
            models = ollama.list()
            return [model['model'] for model in models['models']]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text
        
        Returns:
            Dict with sentiment, score, and confidence
        """
        # Few-shot prompt for better accuracy
        prompt = f"""Analyze the sentiment of this cryptocurrency-related text. 
Classify as BULLISH, BEARISH, or NEUTRAL. Also provide a sentiment score from -1 (very bearish) to +1 (very bullish).

Examples:
Text: "Bitcoin breaking out! To the moon! 🚀"
Response: BULLISH | Score: 0.9 | Confidence: High

Text: "Massive dump incoming, sell everything!"
Response: BEARISH | Score: -0.8 | Confidence: High

Text: "Crypto market trading sideways today"
Response: NEUTRAL | Score: 0.0 | Confidence: Medium

Now analyze this text:
Text: "{text}"

Respond in exactly this format:
SENTIMENT: [BULLISH/BEARISH/NEUTRAL]
SCORE: [number between -1 and 1]
CONFIDENCE: [High/Medium/Low]
EXPLANATION: [brief explanation]
"""
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.1,  # Lower temperature for more consistent responses
                    'num_predict': 150   # Limit response length
                }
            )
            
            return self._parse_response(response['response'])
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                'sentiment': 'NEUTRAL',
                'score': 0.0,
                'confidence': 'Low',
                'explanation': f"Error: {str(e)}"
            }
    
    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response into structured format"""
        result = {
            'sentiment': 'NEUTRAL',
            'score': 0.0,
            'confidence': 'Medium',
            'explanation': ''
        }
        
        # Extract sentiment
        sentiment_match = re.search(r'SENTIMENT:\s*(\w+)', response, re.IGNORECASE)
        if sentiment_match:
            result['sentiment'] = sentiment_match.group(1).upper()
        
        # Extract score
        score_match = re.search(r'SCORE:\s*([+-]?\d*\.?\d+)', response)
        if score_match:
            try:
                result['score'] = float(score_match.group(1))
                # Clamp score between -1 and 1
                result['score'] = max(-1.0, min(1.0, result['score']))
            except:
                pass
        
        # Extract confidence
        confidence_match = re.search(r'CONFIDENCE:\s*(\w+)', response, re.IGNORECASE)
        if confidence_match:
            result['confidence'] = confidence_match.group(1).capitalize()
        
        # Extract explanation
        explanation_match = re.search(r'EXPLANATION:\s*(.+?)(?:\n|$)', response)
        if explanation_match:
            result['explanation'] = explanation_match.group(1).strip()
        
        return result
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment for multiple texts"""
        results = []
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
        return results
    
    def calculate_aggregate_sentiment(self, sentiments: List[Dict]) -> Dict:
        """Calculate aggregate sentiment from multiple analyses"""
        if not sentiments:
            return {'overall_score': 0, 'bullish_count': 0, 'bearish_count': 0}
        
        scores = [s['score'] for s in sentiments]
        bullish = sum(1 for s in sentiments if s['sentiment'] == 'BULLISH')
        bearish = sum(1 for s in sentiments if s['sentiment'] == 'BEARISH')
        neutral = sum(1 for s in sentiments if s['sentiment'] == 'NEUTRAL')
        
        return {
            'overall_score': sum(scores) / len(scores),
            'bullish_count': bullish,
            'bearish_count': bearish,
            'neutral_count': neutral,
            'total_analyzed': len(sentiments),
            'bullish_percentage': (bullish / len(sentiments)) * 100,
            'bearish_percentage': (bearish / len(sentiments)) * 100
        }
    
    def analyze_crypto_news(self, news_items: List[Dict]) -> Dict:
        """
        Analyze multiple news items and return aggregated sentiment
        
        Args:
            news_items: List of dicts with 'title' and optionally 'content'
        """
        sentiments = []
        
        for item in news_items:
            # Combine title and content for better analysis
            text = item.get('title', '')
            if item.get('content'):
                text += " " + item['content'][:500]  # Limit content length
            
            sentiment = self.analyze_sentiment(text)
            sentiments.append(sentiment)
        
        return self.calculate_aggregate_sentiment(sentiments)