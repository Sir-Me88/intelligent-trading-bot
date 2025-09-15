#!/usr/bin/env python3
"""GOD MODE - Sentiment Alert System for Extreme Conditions."""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import aiohttp

from ..config.settings import settings

# Simple sentiment analyzer using VADER (available in requirements)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

class SimpleSentimentAnalyzer:
    """Simple sentiment analyzer using VADER for reliable alerts."""

    def __init__(self):
        self.vader = None
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()

    async def get_overall_sentiment(self, symbol: str) -> Dict:
        """Get sentiment for a symbol using available news data."""
        try:
            # Get recent news headlines for this symbol
            news_headlines = await self._get_recent_news(symbol)

            if not news_headlines:
                return {
                    'overall_sentiment': 0.0,
                    'overall_confidence': 0.0,
                    'recommendation': {'action': 'normal', 'factor': 1.0}
                }

            # Analyze sentiment of combined headlines
            combined_text = ' '.join(news_headlines[:5])  # Use last 5 headlines
            sentiment_score = self._analyze_text_sentiment(combined_text)

            # Calculate confidence based on number of headlines
            confidence = min(len(news_headlines) / 10.0, 1.0)

            return {
                'overall_sentiment': sentiment_score,
                'overall_confidence': confidence,
                'recommendation': self._get_recommendation(sentiment_score, confidence),
                'headline_count': len(news_headlines)
            }

        except Exception as e:
            logger.warning(f"Error getting sentiment for {symbol}: {e}")
            return {
                'overall_sentiment': 0.0,
                'overall_confidence': 0.0,
                'recommendation': {'action': 'normal', 'factor': 1.0}
            }

    async def _get_recent_news(self, symbol: str) -> List[str]:
        """Get recent news headlines for a symbol."""
        try:
            # Try to get news from economic calendar if available
            headlines = []

            # Import here to avoid circular dependency
            try:
                from ..news.economic_calendar import EconomicCalendar
                calendar = EconomicCalendar()

                # Get recent events for this currency
                if len(symbol) >= 3:
                    currency = symbol[:3]  # Extract base currency
                    events = await calendar.get_recent_events(currency, hours_back=24)

                    for event in events:
                        if event.title:
                            headlines.append(event.title)

            except (ImportError, AttributeError):
                # Fallback: create some mock recent headlines for testing
                headlines = [
                    f"{symbol} shows strong economic data",
                    f"Market analysis for {symbol} indicates positive momentum",
                    f"Latest {symbol} news suggests stable conditions"
                ]

            return headlines

        except Exception as e:
            logger.warning(f"Error fetching news for {symbol}: {e}")
            return []

    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using VADER."""
        if not VADER_AVAILABLE or not self.vader:
            return 0.0

        try:
            scores = self.vader.polarity_scores(text)
            return scores['compound']  # Return compound score (-1 to 1)
        except Exception as e:
            logger.warning(f"Error analyzing sentiment: {e}")
            return 0.0

    def _get_recommendation(self, sentiment: float, confidence: float) -> Dict:
        """Get trading recommendation based on sentiment."""
        if confidence < 0.3:
            return {'action': 'normal', 'factor': 1.0, 'reason': 'Low confidence'}

        if sentiment >= 0.3:
            return {
                'action': 'normal',
                'factor': 1.0,
                'reason': f'Positive sentiment ({sentiment:.2f})'
            }
        elif sentiment <= -0.3:
            return {
                'action': 'reduce_risk',
                'factor': 0.7,
                'reason': f'Negative sentiment ({sentiment:.2f})'
            }
        else:
            return {
                'action': 'normal',
                'factor': 1.0,
                'reason': 'Neutral sentiment'
            }

# Use the simple analyzer
SentimentAggregator = SimpleSentimentAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class SentimentAlert:
    """Sentiment alert data structure."""
    alert_id: str
    symbol: str
    alert_type: str
    sentiment_score: float
    confidence: float
    threshold: float
    message: str
    timestamp: datetime
    severity: str
    status: str = "active"
    resolved_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        """Convert alert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        return data

class SentimentAlertSystem:
    """GOD MODE - Advanced sentiment alert system for extreme market conditions."""

    def __init__(self):
        self.sentiment_aggregator = SentimentAggregator()
        self.active_alerts: Dict[str, SentimentAlert] = {}
        self.alert_history: List[SentimentAlert] = []

        # Alert thresholds
        self.extreme_bullish_threshold = 0.6  # Extremely positive sentiment
        self.extreme_bearish_threshold = -0.6  # Extremely negative sentiment
        self.high_bullish_threshold = 0.4  # High positive sentiment
        self.high_bearish_threshold = -0.4  # High negative sentiment
        self.min_confidence_threshold = 0.4  # Minimum confidence for alerts

        # Alert settings
        self.alert_cooldown_minutes = 30  # Don't send same alert within 30 minutes
        self.max_alerts_per_hour = 10  # Rate limiting
        self.alerts_this_hour = 0
        self.hour_reset_time = datetime.now()

        # Persistence
        self.alerts_file = Path("data/alerts/sentiment_alerts.json")
        self.alerts_file.parent.mkdir(parents=True, exist_ok=True)

        # Telegram integration (if available)
        self.telegram_token = getattr(settings, 'telegram_token', None)
        self.telegram_chat_id = getattr(settings, 'telegram_chat_id', None)

        logger.info("🤖 GOD MODE - Sentiment Alert System initialized")

    async def monitor_sentiment_alerts(self, currency_pairs: List[str]) -> List[SentimentAlert]:
        """Monitor sentiment for all currency pairs and generate alerts."""
        new_alerts = []

        try:
            # Rate limiting check
            await self._check_rate_limits()

            for symbol in currency_pairs:
                try:
                    # Get sentiment data
                    sentiment_data = await self.sentiment_aggregator.get_overall_sentiment(symbol)
                    sentiment_score = sentiment_data.get('overall_sentiment', 0.0)
                    confidence = sentiment_data.get('overall_confidence', 0.0)

                    # Skip if confidence too low
                    if confidence < self.min_confidence_threshold:
                        continue

                    # Check for alert conditions
                    alert = await self._check_alert_conditions(
                        symbol, sentiment_score, confidence, sentiment_data
                    )

                    if alert:
                        # Check if similar alert already exists
                        if not self._is_duplicate_alert(alert):
                            new_alerts.append(alert)
                            self.active_alerts[alert.alert_id] = alert

                            # Send alert notifications
                            await self._send_alert_notifications(alert)

                            logger.warning(f"🚨 GOD MODE ALERT: {alert.message}")

                except Exception as e:
                    logger.warning(f"Error monitoring sentiment for {symbol}: {e}")
                    continue

            # Save alerts to file
            if new_alerts:
                await self._save_alerts()

            return new_alerts

        except Exception as e:
            logger.error(f"Error in sentiment alert monitoring: {e}")
            return []

    async def _check_alert_conditions(self, symbol: str, sentiment_score: float,
                                    confidence: float, sentiment_data: Dict) -> Optional[SentimentAlert]:
        """Check if sentiment conditions warrant an alert."""
        try:
            alert_id = f"{symbol}_{sentiment_score:.3f}_{datetime.now().strftime('%H%M%S')}"

            # Extreme bullish sentiment
            if sentiment_score >= self.extreme_bullish_threshold:
                return SentimentAlert(
                    alert_id=alert_id,
                    symbol=symbol,
                    alert_type="extreme_bullish",
                    sentiment_score=sentiment_score,
                    confidence=confidence,
                    threshold=self.extreme_bullish_threshold,
                    message=f"🚀 EXTREME BULLISH SENTIMENT: {symbol} sentiment {sentiment_score:.3f} (confidence: {confidence:.2f})",
                    timestamp=datetime.now(),
                    severity="critical",
                    metadata={
                        'sentiment_sources': sentiment_data,
                        'recommendation': 'Consider long positions or hold existing longs'
                    }
                )

            # Extreme bearish sentiment
            elif sentiment_score <= self.extreme_bearish_threshold:
                return SentimentAlert(
                    alert_id=alert_id,
                    symbol=symbol,
                    alert_type="extreme_bearish",
                    sentiment_score=sentiment_score,
                    confidence=confidence,
                    threshold=self.extreme_bearish_threshold,
                    message=f"📉 EXTREME BEARISH SENTIMENT: {symbol} sentiment {sentiment_score:.3f} (confidence: {confidence:.2f})",
                    timestamp=datetime.now(),
                    severity="critical",
                    metadata={
                        'sentiment_sources': sentiment_data,
                        'recommendation': 'Consider closing long positions or opening shorts'
                    }
                )

            # High bullish sentiment
            elif sentiment_score >= self.high_bullish_threshold:
                return SentimentAlert(
                    alert_id=alert_id,
                    symbol=symbol,
                    alert_type="high_bullish",
                    sentiment_score=sentiment_score,
                    confidence=confidence,
                    threshold=self.high_bullish_threshold,
                    message=f"📈 HIGH BULLISH SENTIMENT: {symbol} sentiment {sentiment_score:.3f} (confidence: {confidence:.2f})",
                    timestamp=datetime.now(),
                    severity="high",
                    metadata={
                        'sentiment_sources': sentiment_data,
                        'recommendation': 'Favorable conditions for long positions'
                    }
                )

            # High bearish sentiment
            elif sentiment_score <= self.high_bearish_threshold:
                return SentimentAlert(
                    alert_id=alert_id,
                    symbol=symbol,
                    alert_type="high_bearish",
                    sentiment_score=sentiment_score,
                    confidence=confidence,
                    threshold=self.high_bearish_threshold,
                    message=f"📉 HIGH BEARISH SENTIMENT: {symbol} sentiment {sentiment_score:.3f} (confidence: {confidence:.2f})",
                    timestamp=datetime.now(),
                    severity="high",
                    metadata={
                        'sentiment_sources': sentiment_data,
                        'recommendation': 'Caution with long positions, consider hedging'
                    }
                )

            return None

        except Exception as e:
            logger.error(f"Error checking alert conditions for {symbol}: {e}")
            return None

    def _is_duplicate_alert(self, new_alert: SentimentAlert) -> bool:
        """Check if similar alert already exists within cooldown period."""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=self.alert_cooldown_minutes)

            for alert in self.active_alerts.values():
                # Check if same symbol and alert type within cooldown
                if (alert.symbol == new_alert.symbol and
                    alert.alert_type == new_alert.alert_type and
                    alert.timestamp > cutoff_time):
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking duplicate alerts: {e}")
            return False

    async def _check_rate_limits(self):
        """Check and reset rate limits."""
        try:
            now = datetime.now()

            # Reset hourly counter if needed
            if now - self.hour_reset_time >= timedelta(hours=1):
                self.alerts_this_hour = 0
                self.hour_reset_time = now

            # Check rate limit
            if self.alerts_this_hour >= self.max_alerts_per_hour:
                logger.warning(f"🚨 Alert rate limit reached ({self.max_alerts_per_hour}/hour)")
                await asyncio.sleep(60)  # Wait before continuing

        except Exception as e:
            logger.error(f"Error checking rate limits: {e}")

    async def _send_alert_notifications(self, alert: SentimentAlert):
        """Send alert notifications through available channels."""
        try:
            # Always log to console/file
            self._log_alert_to_system(alert)

            # Send Telegram notification if configured
            if self.telegram_token and self.telegram_chat_id:
                await self._send_telegram_alert(alert)

            # Increment rate counter
            self.alerts_this_hour += 1

        except Exception as e:
            logger.error(f"Error sending alert notifications: {e}")

    def _log_alert_to_system(self, alert: SentimentAlert):
        """Log alert to system logger with structured data."""
        try:
            alert_data = alert.to_dict()

            if alert.severity == "critical":
                logger.critical(f"🚨 CRITICAL ALERT: {alert.message}", extra=alert_data)
            elif alert.severity == "high":
                logger.warning(f"⚠️ HIGH ALERT: {alert.message}", extra=alert_data)
            else:
                logger.info(f"📢 ALERT: {alert.message}", extra=alert_data)

        except Exception as e:
            logger.error(f"Error logging alert to system: {e}")

    async def _send_telegram_alert(self, alert: SentimentAlert):
        """Send alert via Telegram if configured."""
        try:
            if not self.telegram_token or not self.telegram_chat_id:
                return

            message = f"🤖 GOD MODE ALERT\n\n{alert.message}\n\n"
            message += f"Symbol: {alert.symbol}\n"
            message += f"Type: {alert.alert_type.replace('_', ' ').title()}\n"
            message += f"Sentiment: {alert.sentiment_score:.3f}\n"
            message += f"Confidence: {alert.confidence:.2f}\n"
            message += f"Time: {alert.timestamp.strftime('%H:%M:%S UTC')}\n\n"

            if alert.metadata and 'recommendation' in alert.metadata:
                message += f"💡 {alert.metadata['recommendation']}"

            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"

            async with aiohttp.ClientSession() as session:
                payload = {
                    'chat_id': self.telegram_chat_id,
                    'text': message,
                    'parse_mode': 'HTML'
                }

                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"✅ Telegram alert sent for {alert.symbol}")
                    else:
                        logger.warning(f"❌ Failed to send Telegram alert: {response.status}")

        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")

    async def _save_alerts(self):
        """Save alerts to persistent storage."""
        try:
            all_alerts = [alert.to_dict() for alert in self.active_alerts.values()]
            all_alerts.extend([alert.to_dict() for alert in self.alert_history[-100:]])  # Keep last 100

            with open(self.alerts_file, 'w') as f:
                json.dump(all_alerts, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving alerts to file: {e}")

    async def load_alerts(self):
        """Load alerts from persistent storage."""
        try:
            if self.alerts_file.exists():
                with open(self.alerts_file, 'r') as f:
                    alerts_data = json.load(f)

                for alert_data in alerts_data:
                    # Convert back to SentimentAlert objects
                    alert_data['timestamp'] = datetime.fromisoformat(alert_data['timestamp'])
                    if alert_data.get('resolved_at'):
                        alert_data['resolved_at'] = datetime.fromisoformat(alert_data['resolved_at'])

                    alert = SentimentAlert(**{k: v for k, v in alert_data.items() if k in SentimentAlert.__dataclass_fields__})

                    if alert.status == "active":
                        self.active_alerts[alert.alert_id] = alert
                    else:
                        self.alert_history.append(alert)

                logger.info(f"✅ Loaded {len(self.active_alerts)} active alerts from storage")

        except Exception as e:
            logger.error(f"Error loading alerts from file: {e}")

    def resolve_alert(self, alert_id: str, reason: str = "auto_resolved"):
        """Resolve an active alert."""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = "resolved"
                alert.resolved_at = datetime.now()
                alert.metadata['resolution_reason'] = reason

                # Move to history
                self.alert_history.append(alert)
                del self.active_alerts[alert_id]

                logger.info(f"✅ Alert resolved: {alert.symbol} - {reason}")

        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")

    def get_active_alerts(self) -> List[SentimentAlert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_alert_statistics(self) -> Dict:
        """Get alert system statistics."""
        try:
            now = datetime.now()
            last_24h = now - timedelta(hours=24)
            last_7d = now - timedelta(days=7)

            recent_alerts_24h = [a for a in self.alert_history if a.timestamp > last_24h]
            recent_alerts_7d = [a for a in self.alert_history if a.timestamp > last_7d]

            stats = {
                'active_alerts': len(self.active_alerts),
                'total_alerts_history': len(self.alert_history),
                'alerts_last_24h': len(recent_alerts_24h),
                'alerts_last_7d': len(recent_alerts_7d),
                'alerts_this_hour': self.alerts_this_hour,
                'max_alerts_per_hour': self.max_alerts_per_hour,
                'alert_cooldown_minutes': self.alert_cooldown_minutes,
                'extreme_bullish_threshold': self.extreme_bullish_threshold,
                'extreme_bearish_threshold': self.extreme_bearish_threshold,
                'telegram_enabled': bool(self.telegram_token and self.telegram_chat_id)
            }

            # Alert type breakdown
            alert_types = {}
            for alert in self.active_alerts.values():
                alert_types[alert.alert_type] = alert_types.get(alert.alert_type, 0) + 1
            stats['active_alert_types'] = alert_types

            return stats

        except Exception as e:
            logger.error(f"Error getting alert statistics: {e}")
            return {}

    async def cleanup_old_alerts(self, days_to_keep: int = 30):
        """Clean up old resolved alerts."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            # Filter out old alerts
            self.alert_history = [
                alert for alert in self.alert_history
                if alert.timestamp > cutoff_date
            ]

            logger.info(f"🧹 Cleaned up old alerts, keeping {len(self.alert_history)} in history")

        except Exception as e:
            logger.error(f"Error cleaning up old alerts: {e}")

# Global alert system instance
alert_system = SentimentAlertSystem()

async def initialize_alert_system():
    """Initialize the global alert system."""
    await alert_system.load_alerts()
    logger.info("🤖 GOD MODE - Sentiment Alert System ready")

# Convenience functions for easy access
async def check_sentiment_alerts(currency_pairs: List[str]) -> List[SentimentAlert]:
    """Check for sentiment alerts across currency pairs."""
    return await alert_system.monitor_sentiment_alerts(currency_pairs)

def get_active_alerts() -> List[SentimentAlert]:
    """Get all active sentiment alerts."""
    return alert_system.get_active_alerts()

def get_alert_stats() -> Dict:
    """Get alert system statistics."""
    return alert_system.get_alert_statistics()

def resolve_alert(alert_id: str, reason: str = "manual"):
    """Resolve a specific alert."""
    alert_system.resolve_alert(alert_id, reason)
