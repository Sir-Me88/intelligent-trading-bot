"""News sentiment analysis using VADER and FinBERT."""

import asyncio
import aiohttp
from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta
import re

# Sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    logging.warning("FinBERT not available - install transformers and torch")

# FinGPT Integration
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    FINGPT_AVAILABLE = True
except ImportError:
    FINGPT_AVAILABLE = False
    logging.warning("FinGPT not available - install transformers and torch")

# Social media
import tweepy

# News APIs
from eventregistry import EventRegistry

from ..config.settings import settings

logger = logging.getLogger(__name__)


class FinGPTAnalyzer:
    """FinGPT-based sentiment analysis for financial text."""

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if FINGPT_AVAILABLE:
            try:
                # Load FinGPT model (using a smaller, more accessible model for demo)
                model_name = "microsoft/DialoGPT-small"  # Placeholder - replace with actual FinGPT model
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
                logger.info("FinGPT model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load FinGPT: {e}")
                FINGPT_AVAILABLE = False

    def analyze_sentiment(self, text: str, context: str = None) -> Dict[str, float]:
        """Analyze sentiment using FinGPT with financial context."""
        if not FINGPT_AVAILABLE or not self.model:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'confidence': 0.0}

        try:
            # Create financial context prompt
            if context:
                prompt = f"Financial context: {context}\nAnalyze the sentiment of this text: {text}\nSentiment:"
            else:
                prompt = f"Analyze the sentiment of this financial text: {text}\nSentiment:"

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_text = response.replace(prompt, "").strip()

            # Parse sentiment from response
            sentiment_score = self._parse_sentiment_response(response_text)

            return {
                'compound': sentiment_score,
                'positive': max(0, sentiment_score),
                'negative': max(0, -sentiment_score),
                'neutral': 1.0 - abs(sentiment_score),
                'confidence': 0.8,  # FinGPT confidence
                'raw_response': response_text
            }

        except Exception as e:
            logger.error(f"FinGPT analysis failed: {e}")
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'confidence': 0.0}

    def _parse_sentiment_response(self, response: str) -> float:
        """Parse sentiment score from FinGPT response."""
        response_lower = response.lower()

        # Look for sentiment keywords
        if any(word in response_lower for word in ['positive', 'bullish', 'optimistic', 'good', 'strong']):
            if any(word in response_lower for word in ['very', 'extremely', 'highly']):
                return 0.8
            return 0.6
        elif any(word in response_lower for word in ['negative', 'bearish', 'pessimistic', 'bad', 'weak']):
            if any(word in response_lower for word in ['very', 'extremely', 'highly']):
                return -0.8
            return -0.6
        elif any(word in response_lower for word in ['neutral', 'mixed', 'uncertain']):
            return 0.0
        else:
            # Try to extract numerical sentiment
            import re
            numbers = re.findall(r'-?\d+\.?\d*', response)
            if numbers:
                score = float(numbers[0])
                return max(-1.0, min(1.0, score / 10.0))  # Normalize to -1 to 1
            return 0.0

    def analyze_market_impact(self, text: str, currency_pair: str) -> Dict:
        """Analyze potential market impact of news on currency pair."""
        if not FINGPT_AVAILABLE or not self.model:
            return {'impact': 'unknown', 'direction': 'neutral', 'confidence': 0.0}

        try:
            prompt = f"""Analyze the market impact of this financial news on {currency_pair}:

News: {text}

Market Impact Analysis:
- Direction: (bullish/bearish/neutral)
- Strength: (weak/moderate/strong)
- Timeframe: (short/medium/long term)
- Confidence: (low/medium/high)

Analysis:"""

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            analysis_text = response.replace(prompt, "").strip()

            # Parse analysis
            direction = 'neutral'
            strength = 'moderate'
            timeframe = 'medium'
            confidence = 0.5

            if 'bullish' in analysis_text.lower():
                direction = 'bullish'
            elif 'bearish' in analysis_text.lower():
                direction = 'bearish'

            if 'strong' in analysis_text.lower():
                strength = 'strong'
                confidence = 0.8
            elif 'weak' in analysis_text.lower():
                strength = 'weak'
                confidence = 0.3

            if 'long' in analysis_text.lower():
                timeframe = 'long'
            elif 'short' in analysis_text.lower():
                timeframe = 'short'

            return {
                'impact': strength,
                'direction': direction,
                'timeframe': timeframe,
                'confidence': confidence,
                'analysis': analysis_text
            }

        except Exception as e:
            logger.error(f"FinGPT market impact analysis failed: {e}")
            return {'impact': 'unknown', 'direction': 'neutral', 'confidence': 0.0}


class SentimentAnalyzer:
    """Multi-source sentiment analysis with FinGPT integration."""

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.finbert_tokenizer = None
        self.finbert_model = None
        self.fingpt_analyzer = FinGPTAnalyzer()

        if FINBERT_AVAILABLE:
            try:
                self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                logger.info("FinBERT model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load FinBERT: {e}")
    
    def analyze_vader_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER."""
        scores = self.vader.polarity_scores(text)
        return {
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }
    
    def analyze_finbert_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using FinBERT."""
        if not FINBERT_AVAILABLE or not self.finbert_model:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        try:
            inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT outputs: [positive, negative, neutral]
            pos_score = predictions[0][0].item()
            neg_score = predictions[0][1].item()
            neu_score = predictions[0][2].item()
            
            # Calculate compound score
            compound = pos_score - neg_score
            
            return {
                'compound': compound,
                'positive': pos_score,
                'negative': neg_score,
                'neutral': neu_score
            }
            
        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    
    def combine_sentiments(self, vader_scores: Dict, finbert_scores: Dict) -> Dict[str, float]:
        """Combine VADER and FinBERT sentiment scores."""
        # Weighted average (FinBERT gets higher weight for financial text)
        vader_weight = 0.3
        finbert_weight = 0.7
        
        combined_compound = (vader_weight * vader_scores['compound'] + 
                           finbert_weight * finbert_scores['compound'])
        
        return {
            'compound': combined_compound,
            'positive': (vader_weight * vader_scores['positive'] + 
                        finbert_weight * finbert_scores['positive']),
            'negative': (vader_weight * vader_scores['negative'] + 
                        finbert_weight * finbert_scores['negative']),
            'neutral': (vader_weight * vader_scores['neutral'] + 
                       finbert_weight * finbert_scores['neutral']),
            'vader_compound': vader_scores['compound'],
            'finbert_compound': finbert_scores['compound']
        }


class TwitterSentimentMonitor:
    """Monitor Twitter for forex-related sentiment."""
    
    def __init__(self):
        self.client = None
        self.sentiment_analyzer = SentimentAnalyzer()
        self.keywords = [
            "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD",
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
            "Federal Reserve", "ECB", "BOE", "BOJ", "RBA", "BOC", "RBNZ",
            "interest rates", "inflation", "CPI", "NFP", "GDP", "FOMC"
        ]
        
        try:
            self.client = tweepy.Client(bearer_token=settings.news.twitter_bearer_token)
            logger.info("Twitter client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Twitter client: {e}")
    
    async def get_recent_sentiment(self, currency_pair: str, hours_back: int = 4) -> Dict:
        """Get recent Twitter sentiment for a currency pair."""
        if not self.client:
            return {'sentiment': 0.0, 'tweet_count': 0, 'confidence': 0.0}
        
        try:
            # Extract currencies from pair
            base_currency, quote_currency = currency_pair.split('_')
            
            # Build search query
            query_terms = [base_currency, quote_currency, currency_pair.replace('_', '')]
            query = ' OR '.join(query_terms) + ' -is:retweet lang:en'
            
            # Search recent tweets
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                start_time=start_time,
                end_time=end_time,
                max_results=100
            ).flatten(limit=100)
            
            sentiments = []
            tweet_count = 0
            
            for tweet in tweets:
                if tweet.text:
                    # Clean tweet text
                    clean_text = self._clean_tweet_text(tweet.text)
                    
                    # Analyze sentiment
                    vader_scores = self.sentiment_analyzer.analyze_vader_sentiment(clean_text)
                    finbert_scores = self.sentiment_analyzer.analyze_finbert_sentiment(clean_text)
                    combined = self.sentiment_analyzer.combine_sentiments(vader_scores, finbert_scores)
                    
                    sentiments.append(combined['compound'])
                    tweet_count += 1
            
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)
                confidence = min(tweet_count / 50.0, 1.0)  # Higher confidence with more tweets
            else:
                avg_sentiment = 0.0
                confidence = 0.0
            
            return {
                'sentiment': avg_sentiment,
                'tweet_count': tweet_count,
                'confidence': confidence,
                'timeframe_hours': hours_back
            }
            
        except Exception as e:
            logger.error(f"Error getting Twitter sentiment: {e}")
            return {'sentiment': 0.0, 'tweet_count': 0, 'confidence': 0.0}
    
    def _clean_tweet_text(self, text: str) -> str:
        """Clean tweet text for sentiment analysis."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()


class NewsEventMonitor:
    """Monitor breaking news events using EventRegistry."""
    
    def __init__(self):
        self.er = None
        self.sentiment_analyzer = SentimentAnalyzer()
        
        try:
            self.er = EventRegistry(apiKey=settings.news.eventregistry_api_key)
            logger.info("EventRegistry client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize EventRegistry: {e}")
    
    async def get_breaking_news_sentiment(self, currency: str, hours_back: int = 2) -> Dict:
        """Get sentiment from breaking news about a currency."""
        if not self.er:
            return {'sentiment': 0.0, 'article_count': 0, 'confidence': 0.0}
        
        try:
            # Search for recent articles
            query = {
                "keyword": currency,
                "lang": "eng",
                "dateStart": (datetime.now() - timedelta(hours=hours_back)).strftime("%Y-%m-%d"),
                "dateEnd": datetime.now().strftime("%Y-%m-%d")
            }
            
            articles = self.er.getRecentArticles(query, maxArticles=50)
            
            sentiments = []
            article_count = 0
            
            for article in articles.get('articles', {}).get('results', []):
                title = article.get('title', '')
                body = article.get('body', '')
                
                # Combine title and body for analysis
                text = f"{title}. {body}"[:1000]  # Limit text length
                
                if text.strip():
                    # Analyze sentiment
                    vader_scores = self.sentiment_analyzer.analyze_vader_sentiment(text)
                    finbert_scores = self.sentiment_analyzer.analyze_finbert_sentiment(text)
                    combined = self.sentiment_analyzer.combine_sentiments(vader_scores, finbert_scores)
                    
                    sentiments.append(combined['compound'])
                    article_count += 1
            
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)
                confidence = min(article_count / 20.0, 1.0)  # Higher confidence with more articles
            else:
                avg_sentiment = 0.0
                confidence = 0.0
            
            return {
                'sentiment': avg_sentiment,
                'article_count': article_count,
                'confidence': confidence,
                'timeframe_hours': hours_back
            }
            
        except Exception as e:
            logger.error(f"Error getting news sentiment: {e}")
            return {'sentiment': 0.0, 'article_count': 0, 'confidence': 0.0}


class SentimentAggregator:
    """Aggregates sentiment from multiple sources."""
    
    def __init__(self):
        self.twitter_monitor = TwitterSentimentMonitor()
        self.news_monitor = NewsEventMonitor()
    
    async def get_overall_sentiment(self, currency_pair: str) -> Dict:
        """Get overall sentiment for a currency pair from all sources."""
        # Handle currency pairs like "EURUSD" -> "EUR", "USD"
        if len(currency_pair) == 6:
            base_currency = currency_pair[:3]
            quote_currency = currency_pair[3:]
        else:
            # Fallback for other formats
            base_currency = currency_pair[:3]
            quote_currency = currency_pair[3:]
        
        # Get sentiment from different sources
        twitter_sentiment = await self.twitter_monitor.get_recent_sentiment(currency_pair)
        base_news_sentiment = await self.news_monitor.get_breaking_news_sentiment(base_currency)
        quote_news_sentiment = await self.news_monitor.get_breaking_news_sentiment(quote_currency)
        
        # Combine sentiments with weights
        sentiments = []
        weights = []
        
        # Twitter sentiment
        if twitter_sentiment['confidence'] > 0:
            sentiments.append(twitter_sentiment['sentiment'])
            weights.append(twitter_sentiment['confidence'] * 0.4)  # 40% weight for Twitter
        
        # Base currency news sentiment
        if base_news_sentiment['confidence'] > 0:
            sentiments.append(base_news_sentiment['sentiment'])
            weights.append(base_news_sentiment['confidence'] * 0.3)  # 30% weight for base currency news
        
        # Quote currency news sentiment (inverted for pair sentiment)
        if quote_news_sentiment['confidence'] > 0:
            sentiments.append(-quote_news_sentiment['sentiment'])  # Inverted
            weights.append(quote_news_sentiment['confidence'] * 0.3)  # 30% weight for quote currency news
        
        # Calculate weighted average
        if sentiments and weights:
            overall_sentiment = sum(s * w for s, w in zip(sentiments, weights)) / sum(weights)
            overall_confidence = sum(weights) / len(weights)
        else:
            overall_sentiment = 0.0
            overall_confidence = 0.0
        
        return {
            'overall_sentiment': overall_sentiment,
            'overall_confidence': overall_confidence,
            'twitter_sentiment': twitter_sentiment,
            'base_currency_news': base_news_sentiment,
            'quote_currency_news': quote_news_sentiment,
            'recommendation': self._get_sentiment_recommendation(overall_sentiment, overall_confidence)
        }
    
    def _get_sentiment_recommendation(self, sentiment: float, confidence: float) -> Dict:
        """Get trading recommendation based on sentiment."""
        if confidence < 0.3:
            return {'action': 'ignore', 'reason': 'Low confidence sentiment data'}
        
        if sentiment < -0.3:
            return {
                'action': 'reduce_position_size',
                'factor': 0.5,
                'reason': f'Negative sentiment ({sentiment:.2f}) detected'
            }
        elif sentiment > 0.3:
            return {
                'action': 'normal',
                'factor': 1.0,
                'reason': f'Positive sentiment ({sentiment:.2f}) detected'
            }
        else:
            return {
                'action': 'normal',
                'factor': 1.0,
                'reason': 'Neutral sentiment'
            }
