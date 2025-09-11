#!/usr/bin/env python3
"""GOD MODE - Sentiment-Based Risk Multiplier System."""

import asyncio
import logging
from typing import Dict, List, Optional, Any
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
from ..monitoring.alerts import SentimentAlertSystem

logger = logging.getLogger(__name__)

@dataclass
class RiskMultiplierSignal:
    """Risk multiplier signal with sentiment-based adjustments."""
    symbol: str
    base_risk_multiplier: float
    sentiment_risk_multiplier: float
    final_risk_multiplier: float
    risk_adjustment_reason: str
    volatility_multiplier: float
    correlation_multiplier: float
    market_regime_multiplier: float
    confidence: float
    risk_level: str
    recommended_max_loss: float
    timestamp: datetime

class SentimentRiskMultiplier:
    """GOD MODE - Advanced risk multiplier system based on sentiment analysis."""

    def __init__(self):
        self.sentiment_aggregator = SentimentAggregator()
        self.sentiment_trend_analyzer = SentimentTrendAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.alert_system = SentimentAlertSystem()

        # Risk multiplier parameters
        self.base_risk_multiplier = 1.0
        self.max_risk_multiplier = 3.0  # Maximum 3x risk in extreme conditions
        self.min_risk_multiplier = 0.1  # Minimum 0.1x risk in extreme conditions

        # Risk adjustment thresholds
        self.extreme_sentiment_threshold = 0.5  # Extreme sentiment threshold
        self.high_sentiment_threshold = 0.3     # High sentiment threshold
        self.low_sentiment_threshold = 0.1      # Low sentiment threshold

        # Volatility risk adjustments
        self.high_volatility_threshold = 0.3
        self.low_volatility_threshold = 0.1

        # Risk level definitions
        self.risk_levels = {
            'very_low': {'multiplier_range': (0.1, 0.3), 'description': 'Very Low Risk'},
            'low': {'multiplier_range': (0.3, 0.6), 'description': 'Low Risk'},
            'moderate': {'multiplier_range': (0.6, 1.2), 'description': 'Moderate Risk'},
            'high': {'multiplier_range': (1.2, 2.0), 'description': 'High Risk'},
            'very_high': {'multiplier_range': (2.0, 3.0), 'description': 'Very High Risk'}
        }

        logger.info("🤖 GOD MODE - Sentiment Risk Multiplier initialized")

    async def calculate_risk_multiplier(self, symbol: str, base_risk: float = 1.0,
                                      account_balance: float = 10000.0) -> RiskMultiplierSignal:
        """Calculate sentiment-based risk multiplier for a symbol."""
        try:
            logger.info(f"🔴 Calculating risk multiplier for {symbol}")

            # Get comprehensive sentiment analysis
            sentiment_data = await self.sentiment_aggregator.get_overall_sentiment(symbol)
            sentiment_score = sentiment_data.get('overall_sentiment', 0.0)
            sentiment_confidence = sentiment_data.get('overall_confidence', 0.0)

            # Get trend analysis
            trend_data = await self.sentiment_trend_analyzer.analyze_sentiment_trend(symbol, hours_back=12)

            # Get correlation risk
            correlation_risk = await self._calculate_correlation_risk(symbol)

            # Calculate individual risk multipliers
            sentiment_multiplier = self._calculate_sentiment_risk_multiplier(sentiment_score, sentiment_confidence)
            volatility_multiplier = self._calculate_volatility_risk_multiplier(trend_data)
            regime_multiplier = self._calculate_regime_risk_multiplier(trend_data)
            correlation_multiplier = self._calculate_correlation_risk_multiplier(correlation_risk)

            # Combine multipliers with weights
            final_multiplier = self._combine_risk_multipliers(
                sentiment_multiplier, volatility_multiplier,
                regime_multiplier, correlation_multiplier
            )

            # Ensure within bounds
            final_multiplier = max(self.min_risk_multiplier, min(self.max_risk_multiplier, final_multiplier))

            # Determine risk level
            risk_level = self._determine_risk_level(final_multiplier)

            # Calculate recommended max loss
            recommended_max_loss = self._calculate_recommended_max_loss(
                final_multiplier, account_balance, base_risk
            )

            # Generate risk adjustment reason
            risk_reason = self._generate_risk_adjustment_reason(
                sentiment_score, trend_data, correlation_risk, final_multiplier
            )

            # Calculate confidence
            confidence = self._calculate_risk_confidence(sentiment_confidence, trend_data)

            return RiskMultiplierSignal(
                symbol=symbol,
                base_risk_multiplier=base_risk,
                sentiment_risk_multiplier=sentiment_multiplier,
                final_risk_multiplier=final_multiplier,
                risk_adjustment_reason=risk_reason,
                volatility_multiplier=volatility_multiplier,
                correlation_multiplier=correlation_multiplier,
                market_regime_multiplier=regime_multiplier,
                confidence=confidence,
                risk_level=risk_level,
                recommended_max_loss=recommended_max_loss,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error calculating risk multiplier for {symbol}: {e}")
            # Return safe default
            return RiskMultiplierSignal(
                symbol=symbol,
                base_risk_multiplier=base_risk,
                sentiment_risk_multiplier=1.0,
                final_risk_multiplier=1.0,
                risk_adjustment_reason="Error in calculation - using base risk",
                volatility_multiplier=1.0,
                correlation_multiplier=1.0,
                market_regime_multiplier=1.0,
                confidence=0.5,
                risk_level="moderate",
                recommended_max_loss=base_risk * account_balance * 0.01,
                timestamp=datetime.now()
            )

    def _calculate_sentiment_risk_multiplier(self, sentiment_score: float, confidence: float) -> float:
        """Calculate risk multiplier based on sentiment strength and confidence."""
        try:
            sentiment_strength = abs(sentiment_score)

            # Base multiplier from sentiment strength
            if sentiment_strength > self.extreme_sentiment_threshold:
                # Extreme sentiment - significantly increase risk (contrarian opportunity)
                base_multiplier = 2.5 if sentiment_score > 0 else 2.2
            elif sentiment_strength > self.high_sentiment_threshold:
                # High sentiment - moderately increase risk
                base_multiplier = 1.8 if sentiment_score > 0 else 1.6
            elif sentiment_strength > self.low_sentiment_threshold:
                # Moderate sentiment - slight risk increase
                base_multiplier = 1.3 if sentiment_score > 0 else 1.2
            else:
                # Low sentiment - reduce risk
                base_multiplier = 0.7

            # Confidence adjustment
            if confidence > 0.8:
                confidence_adjustment = 1.1  # High confidence - slight increase
            elif confidence > 0.6:
                confidence_adjustment = 1.0  # Moderate confidence - no change
            elif confidence > 0.4:
                confidence_adjustment = 0.9  # Low confidence - slight decrease
            else:
                confidence_adjustment = 0.7  # Very low confidence - significant decrease

            return base_multiplier * confidence_adjustment

        except Exception:
            return 1.0

    def _calculate_volatility_risk_multiplier(self, trend_data: Dict[str, Any]) -> float:
        """Calculate risk multiplier based on sentiment volatility."""
        try:
            volatility = trend_data.get('volatility', 0.0)

            if volatility > self.high_volatility_threshold:
                # High volatility - significantly reduce risk
                return 0.6
            elif volatility > self.low_volatility_threshold:
                # Moderate volatility - moderately reduce risk
                return 0.8
            else:
                # Low volatility - can take more risk
                return 1.2

        except Exception:
            return 1.0

    def _calculate_regime_risk_multiplier(self, trend_data: Dict[str, Any]) -> float:
        """Calculate risk multiplier based on market regime."""
        try:
            regime = trend_data.get('regime', 'unknown')

            if 'volatile' in regime:
                if 'bullish' in regime:
                    return 1.4  # Volatile bullish - higher risk but potential reward
                elif 'bearish' in regime:
                    return 1.6  # Volatile bearish - high risk
                else:
                    return 0.7  # General volatility - reduce risk
            elif 'stable' in regime:
                if 'bullish' in regime:
                    return 1.2  # Stable bullish - moderate risk increase
                elif 'bearish' in regime:
                    return 1.3  # Stable bearish - moderate risk increase
                else:
                    return 0.9  # Stable neutral - slight risk reduction
            elif 'consolidation' in regime:
                return 0.8  # Consolidation - reduce risk
            elif 'transitional' in regime:
                return 1.1  # Transitional - slight risk increase
            else:
                return 1.0  # Unknown regime - neutral

        except Exception:
            return 1.0

    async def _calculate_correlation_risk(self, symbol: str) -> Dict[str, Any]:
        """Calculate correlation-based risk for the symbol."""
        try:
            # Update sentiment correlation matrix if needed
            currency_pairs = [symbol]  # Focus on this symbol for now
            await self.correlation_analyzer.update_sentiment_correlation_matrix(currency_pairs)

            # Get correlation summary
            correlation_summary = self.correlation_analyzer.get_sentiment_correlation_summary()

            # Calculate correlation risk score
            avg_correlation = correlation_summary.get('average_sentiment_correlation', 0.0)

            return {
                'average_correlation': avg_correlation,
                'correlation_risk': abs(avg_correlation),  # Higher correlation = higher risk
                'correlation_summary': correlation_summary
            }

        except Exception:
            return {'average_correlation': 0.0, 'correlation_risk': 0.0}

    def _calculate_correlation_risk_multiplier(self, correlation_risk: Dict[str, Any]) -> float:
        """Calculate risk multiplier based on correlation risk."""
        try:
            corr_risk = correlation_risk.get('correlation_risk', 0.0)

            if corr_risk > 0.7:
                # High correlation - significantly reduce risk (diversification needed)
                return 0.5
            elif corr_risk > 0.5:
                # Moderate correlation - moderately reduce risk
                return 0.7
            elif corr_risk > 0.3:
                # Low correlation - slight risk reduction
                return 0.9
            else:
                # Very low correlation - can take more risk
                return 1.1

        except Exception:
            return 1.0

    def _combine_risk_multipliers(self, sentiment_mult: float, volatility_mult: float,
                                regime_mult: float, correlation_mult: float) -> float:
        """Combine individual risk multipliers with appropriate weights."""
        try:
            # Weights for different risk factors
            sentiment_weight = 0.4    # Most important
            volatility_weight = 0.25  # Second most important
            regime_weight = 0.2      # Important for market conditions
            correlation_weight = 0.15 # Important for diversification

            # Weighted combination
            combined = (
                sentiment_weight * sentiment_mult +
                volatility_weight * volatility_mult +
                regime_weight * regime_mult +
                correlation_weight * correlation_mult
            )

            # Apply diminishing returns for extreme values
            if combined > 2.0:
                combined = 2.0 + (combined - 2.0) * 0.5  # Diminish extreme increases
            elif combined < 0.5:
                combined = 0.5 - (0.5 - combined) * 0.5  # Diminish extreme decreases

            return combined

        except Exception:
            return 1.0

    def _determine_risk_level(self, multiplier: float) -> str:
        """Determine risk level based on final multiplier."""
        try:
            for level, info in self.risk_levels.items():
                min_mult, max_mult = info['multiplier_range']
                if min_mult <= multiplier <= max_mult:
                    return level

            # Fallback
            if multiplier > 2.0:
                return 'very_high'
            elif multiplier > 1.2:
                return 'high'
            elif multiplier > 0.6:
                return 'moderate'
            elif multiplier > 0.3:
                return 'low'
            else:
                return 'very_low'

        except Exception:
            return 'moderate'

    def _calculate_recommended_max_loss(self, multiplier: float, account_balance: float,
                                      base_risk: float) -> float:
        """Calculate recommended maximum loss based on risk multiplier."""
        try:
            # Base risk as percentage of account
            base_max_loss = account_balance * (base_risk / 100.0)

            # Adjust based on multiplier
            recommended_max_loss = base_max_loss * multiplier

            # Ensure within reasonable bounds
            max_allowed_loss = account_balance * 0.05  # Max 5% of account
            recommended_max_loss = min(recommended_max_loss, max_allowed_loss)

            return recommended_max_loss

        except Exception:
            return account_balance * 0.01  # Default 1%

    def _generate_risk_adjustment_reason(self, sentiment_score: float, trend_data: Dict[str, Any],
                                       correlation_risk: Dict[str, Any], final_multiplier: float) -> str:
        """Generate human-readable reason for risk adjustment."""
        try:
            reasons = []

            # Sentiment-based reason
            sentiment_strength = abs(sentiment_score)
            if sentiment_strength > self.extreme_sentiment_threshold:
                sentiment_desc = "extreme"
            elif sentiment_strength > self.high_sentiment_threshold:
                sentiment_desc = "high"
            elif sentiment_strength > self.low_sentiment_threshold:
                sentiment_desc = "moderate"
            else:
                sentiment_desc = "low"

            sentiment_direction = "bullish" if sentiment_score > 0 else "bearish"
            reasons.append(f"{sentiment_desc} {sentiment_direction} sentiment")

            # Volatility-based reason
            volatility = trend_data.get('volatility', 0.0)
            if volatility > self.high_volatility_threshold:
                reasons.append("high sentiment volatility")
            elif volatility < self.low_volatility_threshold:
                reasons.append("low sentiment volatility")

            # Regime-based reason
            regime = trend_data.get('regime', 'unknown')
            if regime != 'unknown':
                reasons.append(f"{regime.replace('_', ' ')} market regime")

            # Correlation-based reason
            corr_risk = correlation_risk.get('correlation_risk', 0.0)
            if corr_risk > 0.5:
                reasons.append("high sentiment correlation risk")

            # Final assessment
            if final_multiplier > 1.5:
                assessment = "significantly increased"
            elif final_multiplier > 1.2:
                assessment = "moderately increased"
            elif final_multiplier < 0.7:
                assessment = "significantly reduced"
            elif final_multiplier < 0.9:
                assessment = "moderately reduced"
            else:
                assessment = "maintained"

            reason_str = f"Risk {assessment} due to: {', '.join(reasons)}"
            return reason_str

        except Exception:
            return "Risk adjustment based on sentiment analysis"

    def _calculate_risk_confidence(self, sentiment_confidence: float, trend_data: Dict[str, Any]) -> float:
        """Calculate confidence in risk multiplier recommendation."""
        try:
            # Base confidence on sentiment confidence
            base_confidence = sentiment_confidence

            # Adjust based on trend data quality
            trend = trend_data.get('trend', 'insufficient_data')
            if trend == 'insufficient_data':
                base_confidence *= 0.7
            elif trend.startswith('strong_'):
                base_confidence *= 1.1

            # Adjust based on data points
            data_points = trend_data.get('data_points', 0)
            if data_points > 20:
                base_confidence *= 1.1
            elif data_points < 5:
                base_confidence *= 0.8

            return max(0.0, min(1.0, base_confidence))

        except Exception:
            return 0.5

    async def get_risk_multiplier_alerts(self, symbols: List[str], account_balance: float) -> List[Dict[str, Any]]:
        """Get risk multiplier alerts for symbols with extreme risk adjustments."""
        try:
            alerts = []

            for symbol in symbols:
                try:
                    risk_signal = await self.calculate_risk_multiplier(symbol, account_balance=account_balance)

                    # Check for extreme risk adjustments
                    if risk_signal.final_risk_multiplier > 2.0:
                        alerts.append({
                            'type': 'extreme_high_risk',
                            'symbol': symbol,
                            'risk_multiplier': risk_signal.final_risk_multiplier,
                            'reason': risk_signal.risk_adjustment_reason,
                            'recommended_max_loss': risk_signal.recommended_max_loss,
                            'confidence': risk_signal.confidence,
                            'timestamp': risk_signal.timestamp.isoformat()
                        })
                    elif risk_signal.final_risk_multiplier < 0.4:
                        alerts.append({
                            'type': 'extreme_low_risk',
                            'symbol': symbol,
                            'risk_multiplier': risk_signal.final_risk_multiplier,
                            'reason': risk_signal.risk_adjustment_reason,
                            'recommended_max_loss': risk_signal.recommended_max_loss,
                            'confidence': risk_signal.confidence,
                            'timestamp': risk_signal.timestamp.isoformat()
                        })

                except Exception as e:
                    logger.warning(f"Error checking risk alerts for {symbol}: {e}")
                    continue

            return alerts

        except Exception as e:
            logger.error(f"Error generating risk multiplier alerts: {e}")
            return []

    async def get_portfolio_risk_summary(self, symbols: List[str], account_balance: float) -> Dict[str, Any]:
        """Get portfolio-wide risk summary based on sentiment analysis."""
        try:
            portfolio_risks = []

            for symbol in symbols:
                try:
                    risk_signal = await self.calculate_risk_multiplier(symbol, account_balance=account_balance)
                    portfolio_risks.append({
                        'symbol': symbol,
                        'risk_multiplier': risk_signal.final_risk_multiplier,
                        'risk_level': risk_signal.risk_level,
                        'recommended_max_loss': risk_signal.recommended_max_loss
                    })
                except Exception:
                    continue

            if not portfolio_risks:
                return {'error': 'No risk data available'}

            # Calculate portfolio metrics
            avg_risk_multiplier = np.mean([r['risk_multiplier'] for r in portfolio_risks])
            max_risk_multiplier = max([r['risk_multiplier'] for r in portfolio_risks])
            total_recommended_loss = sum([r['recommended_max_loss'] for r in portfolio_risks])

            # Risk distribution
            risk_distribution = {}
            for risk in portfolio_risks:
                level = risk['risk_level']
                risk_distribution[level] = risk_distribution.get(level, 0) + 1

            # Portfolio risk assessment
            if avg_risk_multiplier > 1.5:
                portfolio_risk_level = 'high'
                portfolio_risk_desc = 'High risk - consider reducing exposure'
            elif avg_risk_multiplier > 1.2:
                portfolio_risk_level = 'moderate_high'
                portfolio_risk_desc = 'Moderately high risk - monitor closely'
            elif avg_risk_multiplier < 0.7:
                portfolio_risk_level = 'low'
                portfolio_risk_desc = 'Low risk - consider increasing exposure'
            elif avg_risk_multiplier < 0.9:
                portfolio_risk_level = 'moderate_low'
                portfolio_risk_desc = 'Moderately low risk - balanced approach'
            else:
                portfolio_risk_level = 'moderate'
                portfolio_risk_desc = 'Moderate risk - standard approach'

            return {
                'portfolio_risk_level': portfolio_risk_level,
                'portfolio_risk_description': portfolio_risk_desc,
                'average_risk_multiplier': avg_risk_multiplier,
                'maximum_risk_multiplier': max_risk_multiplier,
                'total_recommended_max_loss': total_recommended_loss,
                'risk_distribution': risk_distribution,
                'individual_risks': portfolio_risks,
                'symbols_analyzed': len(portfolio_risks),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating portfolio risk summary: {e}")
            return {'error': str(e)}

# Global risk multiplier instance
sentiment_risk_multiplier = SentimentRiskMultiplier()

async def calculate_risk_multiplier(symbol: str, base_risk: float = 1.0,
                                  account_balance: float = 10000.0) -> RiskMultiplierSignal:
    """Convenience function to calculate risk multiplier."""
    return await sentiment_risk_multiplier.calculate_risk_multiplier(symbol, base_risk, account_balance)

async def get_risk_multiplier_alerts(symbols: List[str], account_balance: float) -> List[Dict[str, Any]]:
    """Convenience function to get risk multiplier alerts."""
    return await sentiment_risk_multiplier.get_risk_multiplier_alerts(symbols, account_balance)

async def get_portfolio_risk_summary(symbols: List[str], account_balance: float) -> Dict[str, Any]:
    """Convenience function to get portfolio risk summary."""
    return await sentiment_risk_multiplier.get_portfolio_risk_summary(symbols, account_balance)
