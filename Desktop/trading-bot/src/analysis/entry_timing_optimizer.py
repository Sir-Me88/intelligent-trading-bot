#!/usr/bin/env python3
"""GOD MODE - Sentiment-Based Entry Timing Optimization."""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

try:
    from ..news.sentiment import SentimentAggregator, SentimentTrendAnalyzer
    SENTIMENT_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    SENTIMENT_AVAILABLE = False
    # Create mock classes for basic functionality
    class SentimentAggregator:
        async def get_overall_sentiment(self, symbol):
            return {'overall_sentiment': 0.0, 'overall_confidence': 0.0}

    class SentimentTrendAnalyzer:
        def __init__(self):
            self.sentiment_history = {}

        async def analyze_sentiment_trend(self, symbol, hours_back=24):
            return {'trend': 'neutral', 'strength': 0.0, 'regime': 'unknown'}

    SentimentAggregator = SentimentAggregator
    SentimentTrendAnalyzer = SentimentTrendAnalyzer
from ..analysis.correlation import CorrelationAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class EntryTimingSignal:
    """Entry timing signal with sentiment optimization."""
    symbol: str
    base_signal: Dict[str, Any]
    sentiment_score: float
    timing_score: float
    optimal_entry_time: datetime
    entry_delay_minutes: int
    confidence: float
    recommendation: str
    risk_adjustment: float
    expected_holding_time: str
    sentiment_context: Dict[str, Any]

class SentimentEntryTimingOptimizer:
    """GOD MODE - Advanced entry timing optimization based on sentiment analysis."""

    def __init__(self):
        self.sentiment_aggregator = SentimentAggregator()
        self.sentiment_trend_analyzer = SentimentTrendAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()

        # Timing optimization parameters
        self.min_entry_delay = 0  # Minimum delay in minutes
        self.max_entry_delay = 120  # Maximum delay in minutes (2 hours)
        self.sentiment_threshold_strong = 0.4  # Strong sentiment threshold
        self.sentiment_threshold_weak = 0.2  # Weak sentiment threshold
        self.momentum_threshold = 0.15  # Momentum threshold for timing
        self.volatility_threshold = 0.25  # Sentiment volatility threshold

        # Entry timing windows (minutes)
        self.optimal_entry_windows = {
            'immediate': (0, 15),      # Enter immediately
            'short_delay': (15, 45),   # Short delay
            'medium_delay': (45, 90),  # Medium delay
            'long_delay': (90, 120)    # Long delay
        }

        logger.info("🤖 GOD MODE - Sentiment Entry Timing Optimizer initialized")

    async def optimize_entry_timing(self, symbol: str, base_signal: Dict[str, Any]) -> EntryTimingSignal:
        """Optimize entry timing for a trading signal based on sentiment analysis."""
        try:
            logger.info(f"🎯 Optimizing entry timing for {symbol}")

            # Get comprehensive sentiment analysis
            sentiment_data = await self.sentiment_aggregator.get_overall_sentiment(symbol)
            sentiment_score = sentiment_data.get('overall_sentiment', 0.0)
            sentiment_confidence = sentiment_data.get('overall_confidence', 0.0)

            # Get trend analysis
            trend_data = await self.sentiment_trend_analyzer.analyze_sentiment_trend(symbol, hours_back=12)

            # Get momentum and timing analysis
            timing_analysis = await self._analyze_entry_timing(symbol, sentiment_score, trend_data)

            # Calculate optimal entry time
            optimal_entry_time, entry_delay = self._calculate_optimal_entry_time(
                sentiment_score, trend_data, timing_analysis
            )

            # Calculate timing score (0-1, higher is better timing)
            timing_score = self._calculate_timing_score(
                sentiment_score, trend_data, timing_analysis, entry_delay
            )

            # Generate recommendation
            recommendation = self._generate_entry_recommendation(
                sentiment_score, trend_data, timing_score, entry_delay
            )

            # Calculate risk adjustment
            risk_adjustment = self._calculate_risk_adjustment(
                sentiment_score, trend_data, timing_analysis
            )

            # Determine expected holding time
            expected_holding_time = self._estimate_holding_time(sentiment_score, trend_data)

            # Calculate overall confidence
            confidence = self._calculate_overall_confidence(
                sentiment_confidence, timing_score, trend_data
            )

            return EntryTimingSignal(
                symbol=symbol,
                base_signal=base_signal,
                sentiment_score=sentiment_score,
                timing_score=timing_score,
                optimal_entry_time=optimal_entry_time,
                entry_delay_minutes=entry_delay,
                confidence=confidence,
                recommendation=recommendation,
                risk_adjustment=risk_adjustment,
                expected_holding_time=expected_holding_time,
                sentiment_context={
                    'trend_analysis': trend_data,
                    'timing_analysis': timing_analysis,
                    'sentiment_data': sentiment_data
                }
            )

        except Exception as e:
            logger.error(f"Error optimizing entry timing for {symbol}: {e}")
            # Return basic signal if optimization fails
            return EntryTimingSignal(
                symbol=symbol,
                base_signal=base_signal,
                sentiment_score=0.0,
                timing_score=0.5,
                optimal_entry_time=datetime.now(),
                entry_delay_minutes=0,
                confidence=0.5,
                recommendation="Enter immediately (timing optimization failed)",
                risk_adjustment=1.0,
                expected_holding_time="4-8 hours",
                sentiment_context={'error': str(e)}
            )

    async def _analyze_entry_timing(self, symbol: str, sentiment_score: float,
                                  trend_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimal entry timing based on sentiment patterns."""
        try:
            # Get recent sentiment history
            if symbol in self.sentiment_trend_analyzer.sentiment_history:
                recent_readings = self.sentiment_trend_analyzer.sentiment_history[symbol][-20:]  # Last 20 readings

                if len(recent_readings) >= 5:
                    # Calculate sentiment momentum
                    recent_scores = [r['sentiment_score'] for r in recent_readings]
                    momentum = self._calculate_sentiment_momentum(recent_scores)

                    # Calculate sentiment acceleration (rate of change of momentum)
                    acceleration = self._calculate_sentiment_acceleration(recent_scores)

                    # Analyze volatility
                    volatility = np.std(recent_scores)

                    # Check for sentiment reversal patterns
                    reversal_signals = self._detect_sentiment_reversals(recent_scores)

                    # Analyze market regime
                    regime = trend_data.get('regime', 'unknown')

                    return {
                        'momentum': momentum,
                        'acceleration': acceleration,
                        'volatility': volatility,
                        'reversal_signals': reversal_signals,
                        'market_regime': regime,
                        'data_points': len(recent_scores),
                        'sentiment_trend': trend_data.get('trend', 'neutral')
                    }

            # Fallback analysis
            return {
                'momentum': 0.0,
                'acceleration': 0.0,
                'volatility': 0.0,
                'reversal_signals': [],
                'market_regime': 'unknown',
                'data_points': 0,
                'sentiment_trend': 'insufficient_data'
            }

        except Exception as e:
            logger.warning(f"Error in entry timing analysis for {symbol}: {e}")
            return {'error': str(e)}

    def _calculate_sentiment_momentum(self, scores: List[float]) -> float:
        """Calculate sentiment momentum (rate of change)."""
        try:
            if len(scores) < 3:
                return 0.0

            # Use exponential moving average for smoother momentum
            recent_avg = np.mean(scores[-3:])  # Last 3 readings
            older_avg = np.mean(scores[-6:-3]) if len(scores) >= 6 else np.mean(scores[:-3])  # Previous 3 readings

            if older_avg == 0:
                return 0.0

            momentum = (recent_avg - older_avg) / abs(older_avg)
            return float(momentum)

        except Exception:
            return 0.0

    def _calculate_sentiment_acceleration(self, scores: List[float]) -> float:
        """Calculate sentiment acceleration (change in momentum)."""
        try:
            if len(scores) < 5:
                return 0.0

            # Calculate momentum over different periods
            momentum1 = self._calculate_sentiment_momentum(scores[-4:])  # Recent momentum
            momentum2 = self._calculate_sentiment_momentum(scores[-8:-4]) if len(scores) >= 8 else 0.0  # Older momentum

            acceleration = momentum1 - momentum2
            return float(acceleration)

        except Exception:
            return 0.0

    def _detect_sentiment_reversals(self, scores: List[float]) -> List[Dict[str, Any]]:
        """Detect potential sentiment reversal patterns."""
        try:
            reversals = []

            if len(scores) < 5:
                return reversals

            # Check for peak/trough patterns
            for i in range(2, len(scores) - 2):
                current = scores[i]
                prev2 = scores[i-2]
                prev1 = scores[i-1]
                next1 = scores[i+1]
                next2 = scores[i+2]

                # Peak detection (local maximum)
                if (current > prev1 and current > prev2 and
                    current > next1 and current > next2):
                    reversals.append({
                        'type': 'peak',
                        'index': i,
                        'value': current,
                        'strength': current - min(prev2, prev1, next1, next2)
                    })

                # Trough detection (local minimum)
                elif (current < prev1 and current < prev2 and
                      current < next1 and current < next2):
                    reversals.append({
                        'type': 'trough',
                        'index': i,
                        'value': current,
                        'strength': max(prev2, prev1, next1, next2) - current
                    })

            return reversals

        except Exception:
            return []

    def _calculate_optimal_entry_time(self, sentiment_score: float, trend_data: Dict[str, Any],
                                    timing_analysis: Dict[str, Any]) -> Tuple[datetime, int]:
        """Calculate optimal entry time and delay."""
        try:
            base_time = datetime.now()
            momentum = timing_analysis.get('momentum', 0.0)
            acceleration = timing_analysis.get('acceleration', 0.0)
            volatility = timing_analysis.get('volatility', 0.0)
            regime = timing_analysis.get('market_regime', 'unknown')
            trend = trend_data.get('trend', 'neutral')

            # Base delay calculation
            delay_minutes = 0

            # Sentiment strength adjustment
            if abs(sentiment_score) > self.sentiment_threshold_strong:
                if sentiment_score > 0:
                    # Strong positive sentiment - can enter sooner
                    delay_minutes -= 10
                else:
                    # Strong negative sentiment - wait for confirmation
                    delay_minutes += 15
            elif abs(sentiment_score) > self.sentiment_threshold_weak:
                # Moderate sentiment - slight delay for confirmation
                delay_minutes += 5

            # Momentum adjustment
            if abs(momentum) > self.momentum_threshold:
                if momentum > 0:
                    # Positive momentum - enter sooner
                    delay_minutes -= 8
                else:
                    # Negative momentum - wait longer
                    delay_minutes += 12

            # Volatility adjustment
            if volatility > self.volatility_threshold:
                # High volatility - wait for stabilization
                delay_minutes += 20
            elif volatility < 0.1:
                # Low volatility - can enter sooner
                delay_minutes -= 5

            # Market regime adjustment
            if 'volatile' in regime:
                delay_minutes += 15  # Wait in volatile conditions
            elif 'stable' in regime:
                delay_minutes -= 10  # Enter faster in stable conditions

            # Trend adjustment
            if trend.startswith('strong_'):
                if 'bullish' in trend and sentiment_score > 0:
                    delay_minutes -= 5  # Aligning with strong trend
                elif 'bearish' in trend and sentiment_score < 0:
                    delay_minutes -= 5  # Aligning with strong trend
                else:
                    delay_minutes += 10  # Against strong trend

            # Ensure delay is within bounds
            delay_minutes = max(self.min_entry_delay, min(self.max_entry_delay, delay_minutes))

            optimal_time = base_time + timedelta(minutes=delay_minutes)

            return optimal_time, delay_minutes

        except Exception as e:
            logger.warning(f"Error calculating optimal entry time: {e}")
            return datetime.now(), 0

    def _calculate_timing_score(self, sentiment_score: float, trend_data: Dict[str, Any],
                              timing_analysis: Dict[str, Any], entry_delay: int) -> float:
        """Calculate timing score (0-1, higher is better)."""
        try:
            score = 0.5  # Base score

            # Sentiment alignment (30% weight)
            sentiment_alignment = min(abs(sentiment_score), 1.0)
            score += 0.3 * sentiment_alignment

            # Momentum alignment (25% weight)
            momentum = timing_analysis.get('momentum', 0.0)
            momentum_alignment = min(abs(momentum), 1.0)
            if (sentiment_score > 0 and momentum > 0) or (sentiment_score < 0 and momentum < 0):
                score += 0.25 * momentum_alignment
            else:
                score -= 0.1 * momentum_alignment

            # Volatility consideration (20% weight)
            volatility = timing_analysis.get('volatility', 0.0)
            if volatility < 0.2:
                score += 0.2 * (1 - volatility / 0.2)  # Bonus for low volatility
            else:
                score -= 0.1 * min(volatility / 0.4, 1.0)  # Penalty for high volatility

            # Delay optimization (15% weight)
            optimal_delay_range = (15, 60)  # Sweet spot for delays
            if optimal_delay_range[0] <= entry_delay <= optimal_delay_range[1]:
                score += 0.15
            elif entry_delay < optimal_delay_range[0]:
                score += 0.1  # Small bonus for immediate entry
            else:
                score -= 0.05 * min((entry_delay - optimal_delay_range[1]) / 30, 1.0)

            # Trend alignment (10% weight)
            trend = trend_data.get('trend', 'neutral')
            if trend != 'neutral' and trend != 'insufficient_data':
                if ((sentiment_score > 0 and 'bullish' in trend) or
                    (sentiment_score < 0 and 'bearish' in trend)):
                    score += 0.1
                else:
                    score -= 0.05

            return max(0.0, min(1.0, score))

        except Exception:
            return 0.5

    def _generate_entry_recommendation(self, sentiment_score: float, trend_data: Dict[str, Any],
                                     timing_score: float, entry_delay: int) -> str:
        """Generate entry recommendation based on analysis."""
        try:
            sentiment_strength = "strong" if abs(sentiment_score) > self.sentiment_threshold_strong else "moderate" if abs(sentiment_score) > self.sentiment_threshold_weak else "weak"
            sentiment_direction = "bullish" if sentiment_score > 0 else "bearish"

            timing_quality = "excellent" if timing_score > 0.8 else "good" if timing_score > 0.6 else "fair" if timing_score > 0.4 else "poor"

            if entry_delay == 0:
                delay_desc = "immediate entry"
            elif entry_delay <= 15:
                delay_desc = f"enter in {entry_delay} minutes"
            elif entry_delay <= 60:
                delay_desc = f"enter in {entry_delay // 15 * 15} minutes"
            else:
                delay_desc = f"wait {entry_delay} minutes before entry"

            trend = trend_data.get('trend', 'neutral')
            regime = trend_data.get('regime', 'unknown')

            if timing_score > 0.7:
                confidence_desc = "High confidence"
            elif timing_score > 0.5:
                confidence_desc = "Moderate confidence"
            else:
                confidence_desc = "Low confidence - consider waiting"

            recommendation = f"{confidence_desc}: {sentiment_strength} {sentiment_direction} sentiment, {timing_quality} timing. {delay_desc.upper()}."

            if 'volatile' in regime:
                recommendation += " High volatility - use tight stops."
            elif 'stable' in regime:
                recommendation += " Stable conditions - consider holding longer."

            return recommendation

        except Exception:
            return "Enter immediately - timing analysis inconclusive"

    def _calculate_risk_adjustment(self, sentiment_score: float, trend_data: Dict[str, Any],
                                 timing_analysis: Dict[str, Any]) -> float:
        """Calculate risk adjustment factor for position sizing."""
        try:
            base_adjustment = 1.0

            # Sentiment strength adjustment
            sentiment_strength = abs(sentiment_score)
            if sentiment_strength > self.sentiment_threshold_strong:
                base_adjustment *= 1.2  # Increase position size for strong sentiment
            elif sentiment_strength < 0.1:
                base_adjustment *= 0.7  # Reduce position size for weak sentiment

            # Volatility adjustment
            volatility = timing_analysis.get('volatility', 0.0)
            if volatility > self.volatility_threshold:
                base_adjustment *= 0.8  # Reduce size in high volatility
            elif volatility < 0.1:
                base_adjustment *= 1.1  # Increase size in low volatility

            # Trend alignment bonus
            trend = trend_data.get('trend', 'neutral')
            if trend.startswith('strong_'):
                if ((sentiment_score > 0 and 'bullish' in trend) or
                    (sentiment_score < 0 and 'bearish' in trend)):
                    base_adjustment *= 1.15  # Bonus for trend alignment

            # Momentum adjustment
            momentum = timing_analysis.get('momentum', 0.0)
            if abs(momentum) > self.momentum_threshold:
                if ((sentiment_score > 0 and momentum > 0) or
                    (sentiment_score < 0 and momentum < 0)):
                    base_adjustment *= 1.1  # Bonus for momentum alignment

            return max(0.3, min(2.0, base_adjustment))  # Clamp between 0.3 and 2.0

        except Exception:
            return 1.0

    def _estimate_holding_time(self, sentiment_score: float, trend_data: Dict[str, Any]) -> str:
        """Estimate expected holding time based on sentiment and trend."""
        try:
            base_time = "4-8 hours"  # Default

            # Sentiment strength adjustment
            sentiment_strength = abs(sentiment_score)
            if sentiment_strength > self.sentiment_threshold_strong:
                base_time = "8-24 hours"  # Hold longer for strong sentiment
            elif sentiment_strength < 0.1:
                base_time = "2-4 hours"  # Hold shorter for weak sentiment

            # Trend strength adjustment
            trend = trend_data.get('trend', 'neutral')
            if trend.startswith('strong_'):
                base_time = "12-48 hours"  # Hold longer in strong trends
            elif trend.startswith('weak_'):
                base_time = "2-6 hours"  # Hold shorter in weak trends

            # Volatility consideration
            volatility = trend_data.get('volatility', 0.0)
            if volatility > self.volatility_threshold:
                # Reduce holding time in high volatility
                if "2-4" in base_time:
                    base_time = "1-2 hours"
                elif "4-8" in base_time:
                    base_time = "2-4 hours"
                elif "8-24" in base_time:
                    base_time = "4-12 hours"

            return base_time

        except Exception:
            return "4-8 hours"

    def _calculate_overall_confidence(self, sentiment_confidence: float, timing_score: float,
                                    trend_data: Dict[str, Any]) -> float:
        """Calculate overall confidence in the entry timing recommendation."""
        try:
            # Weighted combination of factors
            sentiment_weight = 0.4
            timing_weight = 0.4
            trend_weight = 0.2

            # Trend confidence
            trend = trend_data.get('trend', 'neutral')
            trend_confidence = 0.8 if not trend.startswith(('neutral', 'insufficient')) else 0.4

            overall_confidence = (
                sentiment_weight * sentiment_confidence +
                timing_weight * timing_score +
                trend_weight * trend_confidence
            )

            return max(0.0, min(1.0, overall_confidence))

        except Exception:
            return 0.5

    async def get_entry_timing_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """Get summary of entry timing analysis for multiple symbols."""
        try:
            timing_analysis = {}

            for symbol in symbols:
                try:
                    # Get basic sentiment for quick analysis
                    sentiment_data = await self.sentiment_aggregator.get_overall_sentiment(symbol)
                    sentiment_score = sentiment_data.get('overall_sentiment', 0.0)

                    # Quick trend analysis
                    trend_data = await self.sentiment_trend_analyzer.analyze_sentiment_trend(symbol, hours_back=6)

                    # Simple timing recommendation
                    if abs(sentiment_score) > self.sentiment_threshold_strong:
                        if sentiment_score > 0:
                            timing_rec = "Enter soon - strong bullish sentiment"
                        else:
                            timing_rec = "Wait for confirmation - strong bearish sentiment"
                    elif abs(sentiment_score) > self.sentiment_threshold_weak:
                        timing_rec = "Enter with moderate delay - moderate sentiment"
                    else:
                        timing_rec = "Wait for stronger sentiment signal"

                    timing_analysis[symbol] = {
                        'sentiment_score': sentiment_score,
                        'trend': trend_data.get('trend', 'unknown'),
                        'timing_recommendation': timing_rec,
                        'confidence': sentiment_data.get('overall_confidence', 0.0)
                    }

                except Exception as e:
                    timing_analysis[symbol] = {'error': str(e)}

            # Sort by sentiment strength
            sorted_symbols = sorted(
                [(s, data) for s, data in timing_analysis.items() if 'error' not in data],
                key=lambda x: abs(x[1]['sentiment_score']),
                reverse=True
            )

            return {
                'timing_analysis': dict(sorted_symbols),
                'summary': {
                    'total_symbols': len(symbols),
                    'analyzed_symbols': len([s for s in timing_analysis.keys() if 'error' not in timing_analysis[s]]),
                    'strong_sentiment_symbols': len([s for s, data in timing_analysis.items()
                                                   if 'error' not in data and abs(data['sentiment_score']) > self.sentiment_threshold_strong]),
                    'top_opportunities': [s for s, _ in sorted_symbols[:3]]
                }
            }

        except Exception as e:
            logger.error(f"Error generating entry timing summary: {e}")
            return {'error': str(e)}

# Global optimizer instance
entry_timing_optimizer = SentimentEntryTimingOptimizer()

async def optimize_entry_timing(symbol: str, base_signal: Dict[str, Any]) -> EntryTimingSignal:
    """Convenience function to optimize entry timing."""
    return await entry_timing_optimizer.optimize_entry_timing(symbol, base_signal)

async def get_entry_timing_summary(symbols: List[str]) -> Dict[str, Any]:
    """Convenience function to get entry timing summary."""
    return await entry_timing_optimizer.get_entry_timing_summary(symbols)
