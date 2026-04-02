import ollama
import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, model_name: str = "llama3.2:latest"):
        self.model_name = model_name
        self._check_model()
        logger.info(f"Sentiment analyzer initialized with {model_name}")
    
    def _check_model(self):
        try:
            models = ollama.list()
            available = [m['model'] for m in models['models']]
            if self.model_name not in available:
                logger.warning(f"Model {self.model_name} not found. Available: {available}")
                if available:
                    self.model_name = available[0]
        except Exception as e:
            logger.error(f"Ollama connection error: {e}")
    
    def analyze_sentiment(self, text: str) -> Dict:
        prompt = f"""Analyze the sentiment of this cryptocurrency text. 
Classify as BULLISH, BEARISH, or NEUTRAL. Provide a score from -1 (very bearish) to +1 (very bullish).

Examples:
Text: "Bitcoin breaking out to the moon"
Response: BULLISH | Score: 0.9

Text: "Massive dump incoming, sell everything"
Response: BEARISH | Score: -0.8

Text: "Crypto market trading sideways"
Response: NEUTRAL | Score: 0.0

Text: "{text[:500]}"

Respond in exactly this format:
SENTIMENT: [BULLISH/BEARISH/NEUTRAL]
SCORE: [number between -1 and 1]"""
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={'temperature': 0.1, 'num_predict': 100}
            )
            return self._parse_response(response['response'])
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {'sentiment': 'NEUTRAL', 'score': 0.0}
    
    def _parse_response(self, response: str) -> Dict:
        result = {'sentiment': 'NEUTRAL', 'score': 0.0}
        
        sentiment_match = re.search(r'SENTIMENT:\s*(\w+)', response, re.IGNORECASE)
        if sentiment_match:
            result['sentiment'] = sentiment_match.group(1).upper()
        
        score_match = re.search(r'SCORE:\s*([+-]?\d*\.?\d+)', response)
        if score_match:
            result['score'] = float(score_match.group(1))
            result['score'] = max(-1.0, min(1.0, result['score']))
        
        return result
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        return [self.analyze_sentiment(text) for text in texts]