#!/usr/bin/env python3
"""GOD MODE - Sentiment Dashboard for Real-Time Monitoring."""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import aiohttp

from ..config.settings import settings
from ..news.sentiment import SentimentAggregator, SentimentTrendAnalyzer
from .alerts import SentimentAlertSystem

logger = logging.getLogger(__name__)

class SentimentDashboard:
    """GOD MODE - Comprehensive sentiment monitoring dashboard."""

    def __init__(self):
        self.sentiment_aggregator = SentimentAggregator()
        self.sentiment_trend_analyzer = SentimentTrendAnalyzer()
        self.alert_system = SentimentAlertSystem()

        # Dashboard data
        self.dashboard_data: Dict[str, Any] = {}
        self.last_update = None
        self.update_interval = 60  # Update every 60 seconds

        # Dashboard file
        self.dashboard_file = Path("data/dashboard/sentiment_dashboard.json")
        self.dashboard_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info("🤖 GOD MODE - Sentiment Dashboard initialized")

    async def update_dashboard(self, currency_pairs: List[str]) -> Dict[str, Any]:
        """Update the complete sentiment dashboard with latest data."""
        try:
            logger.info("📊 Updating GOD MODE Sentiment Dashboard...")

            current_time = datetime.now()
            dashboard = {
                'timestamp': current_time.isoformat(),
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'currency_pairs': {},
                'market_overview': {},
                'alerts_summary': {},
                'trend_analysis': {},
                'performance_metrics': {},
                'system_health': {}
            }

            # Update sentiment data for each pair
            for symbol in currency_pairs:
                try:
                    pair_data = await self._get_pair_dashboard_data(symbol)
                    dashboard['currency_pairs'][symbol] = pair_data
                except Exception as e:
                    logger.warning(f"Error updating dashboard for {symbol}: {e}")
                    dashboard['currency_pairs'][symbol] = {'error': str(e)}

            # Market overview
            dashboard['market_overview'] = await self._get_market_overview(currency_pairs)

            # Alerts summary
            dashboard['alerts_summary'] = self._get_alerts_summary()

            # Trend analysis
            dashboard['trend_analysis'] = await self._get_trend_analysis(currency_pairs)

            # Performance metrics
            dashboard['performance_metrics'] = self._get_performance_metrics()

            # System health
            dashboard['system_health'] = self._get_system_health()

            # Save dashboard data
            self.dashboard_data = dashboard
            self.last_update = current_time
            await self._save_dashboard()

            logger.info("✅ GOD MODE Sentiment Dashboard updated successfully")
            return dashboard

        except Exception as e:
            logger.error(f"Error updating sentiment dashboard: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    async def _get_pair_dashboard_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive dashboard data for a single currency pair."""
        try:
            # Current sentiment
            sentiment_data = await self.sentiment_aggregator.get_overall_sentiment(symbol)

            # Trend analysis
            trend_data = await self.sentiment_trend_analyzer.analyze_sentiment_trend(symbol, hours_back=24)

            # Forecast
            forecast_data = await self.sentiment_trend_analyzer.get_sentiment_bias_forecast(symbol, forecast_hours=4)

            # Historical data (last 10 readings)
            historical_data = []
            if symbol in self.sentiment_trend_analyzer.sentiment_history:
                recent_readings = self.sentiment_trend_analyzer.sentiment_history[symbol][-10:]
                historical_data = [
                    {
                        'timestamp': reading['timestamp'].isoformat(),
                        'sentiment_score': reading['sentiment_score'],
                        'confidence': reading['confidence']
                    }
                    for reading in recent_readings
                ]

            return {
                'current_sentiment': {
                    'score': sentiment_data.get('overall_sentiment', 0.0),
                    'confidence': sentiment_data.get('overall_confidence', 0.0),
                    'recommendation': sentiment_data.get('recommendation', {}),
                    'sources': {
                        'twitter': sentiment_data.get('twitter_sentiment', {}),
                        'news': sentiment_data.get('base_currency_news', {}),
                        'fmp': sentiment_data.get('quote_currency_news', {})
                    }
                },
                'trend_analysis': trend_data,
                'forecast': forecast_data,
                'historical_data': historical_data,
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting dashboard data for {symbol}: {e}")
            return {'error': str(e)}

    async def _get_market_overview(self, currency_pairs: List[str]) -> Dict[str, Any]:
        """Get market-wide sentiment overview."""
        try:
            # Calculate market sentiment averages
            sentiment_scores = []
            bullish_pairs = 0
            bearish_pairs = 0
            neutral_pairs = 0

            for symbol in currency_pairs:
                try:
                    sentiment_data = await self.sentiment_aggregator.get_overall_sentiment(symbol)
                    score = sentiment_data.get('overall_sentiment', 0.0)
                    confidence = sentiment_data.get('overall_confidence', 0.0)

                    if confidence >= 0.3:  # Only count confident readings
                        sentiment_scores.append(score)

                        if score > 0.2:
                            bullish_pairs += 1
                        elif score < -0.2:
                            bearish_pairs += 1
                        else:
                            neutral_pairs += 1

                except Exception:
                    continue

            # Market sentiment summary
            if sentiment_scores:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                market_sentiment = 'bullish' if avg_sentiment > 0.1 else 'bearish' if avg_sentiment < -0.1 else 'neutral'
            else:
                avg_sentiment = 0.0
                market_sentiment = 'insufficient_data'

            # Volatility assessment
            if len(sentiment_scores) > 1:
                volatility = sum((s - avg_sentiment) ** 2 for s in sentiment_scores) / len(sentiment_scores)
                volatility = volatility ** 0.5
            else:
                volatility = 0.0

            return {
                'market_sentiment': market_sentiment,
                'average_sentiment': avg_sentiment,
                'sentiment_volatility': volatility,
                'pair_breakdown': {
                    'bullish': bullish_pairs,
                    'bearish': bearish_pairs,
                    'neutral': neutral_pairs,
                    'total_analyzed': len(sentiment_scores)
                },
                'confidence_metrics': {
                    'total_pairs': len(currency_pairs),
                    'analyzed_pairs': len(sentiment_scores),
                    'coverage_percentage': len(sentiment_scores) / len(currency_pairs) * 100
                }
            }

        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {'error': str(e)}

    def _get_alerts_summary(self) -> Dict[str, Any]:
        """Get summary of active alerts and recent alert activity."""
        try:
            active_alerts = self.alert_system.get_active_alerts()
            alert_stats = self.alert_system.get_alert_statistics()

            # Categorize active alerts
            alert_categories = {}
            for alert in active_alerts:
                category = alert.alert_type
                if category not in alert_categories:
                    alert_categories[category] = []
                alert_categories[category].append({
                    'symbol': alert.symbol,
                    'severity': alert.severity,
                    'sentiment_score': alert.sentiment_score,
                    'timestamp': alert.timestamp.isoformat()
                })

            return {
                'active_alerts_count': len(active_alerts),
                'alert_categories': alert_categories,
                'alert_statistics': alert_stats,
                'recent_activity': {
                    'alerts_last_24h': alert_stats.get('alerts_last_24h', 0),
                    'alerts_last_7d': alert_stats.get('alerts_last_7d', 0)
                }
            }

        except Exception as e:
            logger.error(f"Error getting alerts summary: {e}")
            return {'error': str(e)}

    async def _get_trend_analysis(self, currency_pairs: List[str]) -> Dict[str, Any]:
        """Get trend analysis summary across all pairs."""
        try:
            trend_summary = {
                'strong_bullish': [],
                'strong_bearish': [],
                'weak_bullish': [],
                'weak_bearish': [],
                'neutral': [],
                'insufficient_data': []
            }

            regime_summary = {}

            for symbol in currency_pairs:
                try:
                    trend_data = await self.sentiment_trend_analyzer.analyze_sentiment_trend(symbol, hours_back=24)
                    trend = trend_data.get('trend', 'insufficient_data')
                    regime = trend_data.get('regime', 'unknown')

                    if trend in trend_summary:
                        trend_summary[trend].append({
                            'symbol': symbol,
                            'strength': trend_data.get('strength', 0.0),
                            'regime': regime
                        })

                    # Count regimes
                    if regime not in regime_summary:
                        regime_summary[regime] = 0
                    regime_summary[regime] += 1

                except Exception:
                    trend_summary['insufficient_data'].append({'symbol': symbol})

            return {
                'trend_distribution': {k: len(v) for k, v in trend_summary.items()},
                'trend_details': trend_summary,
                'regime_distribution': regime_summary,
                'dominant_trend': max(trend_summary.keys(), key=lambda k: len(trend_summary[k])) if trend_summary else 'unknown',
                'dominant_regime': max(regime_summary.keys(), key=lambda k: regime_summary[k]) if regime_summary else 'unknown'
            }

        except Exception as e:
            logger.error(f"Error getting trend analysis: {e}")
            return {'error': str(e)}

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for GOD MODE sentiment features."""
        try:
            # This would integrate with actual performance tracking
            # For now, return placeholder structure
            return {
                'sentiment_accuracy': {
                    'overall': 0.0,  # Would be calculated from actual trade outcomes
                    'bullish_predictions': 0.0,
                    'bearish_predictions': 0.0
                },
                'alert_effectiveness': {
                    'alerts_triggered': 0,
                    'profitable_alerts': 0,
                    'alert_success_rate': 0.0
                },
                'position_adjustments': {
                    'total_adjustments': 0,
                    'profitable_adjustments': 0,
                    'avg_pnl_impact': 0.0
                },
                'stop_loss_optimizations': {
                    'total_modifications': 0,
                    'avg_pips_saved': 0.0,
                    'success_rate': 0.0
                }
            }

        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {'error': str(e)}

    def _get_system_health(self) -> Dict[str, Any]:
        """Get system health and status information."""
        try:
            return {
                'sentiment_analyzer': {
                    'status': 'operational',
                    'last_update': self.last_update.isoformat() if self.last_update else None,
                    'pairs_tracked': len(self.sentiment_trend_analyzer.sentiment_history),
                    'total_readings': sum(len(history) for history in self.sentiment_trend_analyzer.sentiment_history.values())
                },
                'alert_system': {
                    'status': 'operational',
                    'active_alerts': len(self.alert_system.get_active_alerts()),
                    'telegram_enabled': bool(getattr(self.alert_system, 'telegram_token', None))
                },
                'trend_analyzer': {
                    'status': 'operational',
                    'symbols_analyzed': len(self.sentiment_trend_analyzer.sentiment_history)
                },
                'last_dashboard_update': self.last_update.isoformat() if self.last_update else None,
                'update_interval_seconds': self.update_interval
            }

        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {'error': str(e)}

    async def _save_dashboard(self):
        """Save dashboard data to file."""
        try:
            with open(self.dashboard_file, 'w') as f:
                json.dump(self.dashboard_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving dashboard: {e}")

    async def load_dashboard(self) -> Optional[Dict[str, Any]]:
        """Load dashboard data from file."""
        try:
            if self.dashboard_file.exists():
                with open(self.dashboard_file, 'r') as f:
                    data = json.load(f)
                self.dashboard_data = data
                return data
        except Exception as e:
            logger.error(f"Error loading dashboard: {e}")
        return None

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get a concise dashboard summary for quick monitoring."""
        try:
            if not self.dashboard_data:
                return {'status': 'no_data', 'message': 'Dashboard not yet updated'}

            market_overview = self.dashboard_data.get('market_overview', {})
            alerts_summary = self.dashboard_data.get('alerts_summary', {})
            trend_analysis = self.dashboard_data.get('trend_analysis', {})

            return {
                'status': 'active',
                'last_update': self.dashboard_data.get('timestamp'),
                'market_sentiment': market_overview.get('market_sentiment', 'unknown'),
                'average_sentiment': market_overview.get('average_sentiment', 0.0),
                'active_alerts': alerts_summary.get('active_alerts_count', 0),
                'dominant_trend': trend_analysis.get('dominant_trend', 'unknown'),
                'dominant_regime': trend_analysis.get('dominant_regime', 'unknown'),
                'pairs_analyzed': len(self.dashboard_data.get('currency_pairs', {})),
                'system_health': 'good' if self._get_system_health().get('sentiment_analyzer', {}).get('status') == 'operational' else 'issues'
            }

        except Exception as e:
            logger.error(f"Error getting dashboard summary: {e}")
            return {'status': 'error', 'message': str(e)}

    async def generate_dashboard_report(self) -> str:
        """Generate a human-readable dashboard report."""
        try:
            if not self.dashboard_data:
                return "🤖 GOD MODE Dashboard: No data available. Please update dashboard first."

            report = []
            report.append("🤖 GOD MODE SENTIMENT DASHBOARD")
            report.append("=" * 50)
            report.append(f"📅 Last Update: {self.dashboard_data.get('timestamp', 'Unknown')}")
            report.append("")

            # Market Overview
            market = self.dashboard_data.get('market_overview', {})
            report.append("🌍 MARKET OVERVIEW")
            report.append(f"   Overall Sentiment: {market.get('market_sentiment', 'Unknown').upper()}")
            report.append(f"   Average Score: {market.get('average_sentiment', 0.0):.3f}")
            report.append(f"   Sentiment Volatility: {market.get('sentiment_volatility', 0.0):.3f}")

            breakdown = market.get('pair_breakdown', {})
            report.append(f"   Pair Breakdown: {breakdown.get('bullish', 0)} bullish, {breakdown.get('bearish', 0)} bearish, {breakdown.get('neutral', 0)} neutral")
            report.append("")

            # Active Alerts
            alerts = self.dashboard_data.get('alerts_summary', {})
            report.append("🚨 ACTIVE ALERTS")
            report.append(f"   Total Active: {alerts.get('active_alerts_count', 0)}")

            categories = alerts.get('alert_categories', {})
            for category, alert_list in categories.items():
                report.append(f"   {category.replace('_', ' ').title()}: {len(alert_list)} alerts")
            report.append("")

            # Trend Analysis
            trends = self.dashboard_data.get('trend_analysis', {})
            report.append("📈 TREND ANALYSIS")
            report.append(f"   Dominant Trend: {trends.get('dominant_trend', 'Unknown').replace('_', ' ').title()}")
            report.append(f"   Dominant Regime: {trends.get('dominant_regime', 'Unknown').replace('_', ' ').title()}")

            trend_dist = trends.get('trend_distribution', {})
            report.append("   Trend Distribution:")
            for trend, count in trend_dist.items():
                if count > 0:
                    report.append(f"     {trend.replace('_', ' ').title()}: {count} pairs")
            report.append("")

            # Top Pairs by Sentiment
            pairs = self.dashboard_data.get('currency_pairs', {})
            if pairs:
                report.append("🎯 TOP PAIRS BY SENTIMENT")

                # Sort pairs by sentiment score
                sorted_pairs = sorted(
                    [(symbol, data.get('current_sentiment', {}).get('score', 0.0))
                     for symbol, data in pairs.items() if 'error' not in data],
                    key=lambda x: x[1],
                    reverse=True
                )

                # Show top 5 bullish and bearish
                bullish = [p for p in sorted_pairs[:5] if p[1] > 0.1]
                bearish = [p for p in sorted_pairs[-5:] if p[1] < -0.1]

                if bullish:
                    report.append("   Most Bullish:")
                    for symbol, score in bullish:
                        report.append(f"     {symbol}: {score:.3f}")

                if bearish:
                    report.append("   Most Bearish:")
                    for symbol, score in bearish:
                        report.append(f"     {symbol}: {score:.3f}")
                report.append("")

            # System Health
            health = self.dashboard_data.get('system_health', {})
            report.append("⚡ SYSTEM HEALTH")
            report.append(f"   Sentiment Analyzer: {health.get('sentiment_analyzer', {}).get('status', 'Unknown')}")
            report.append(f"   Alert System: {health.get('alert_system', {}).get('status', 'Unknown')}")
            report.append(f"   Pairs Tracked: {health.get('sentiment_analyzer', {}).get('pairs_tracked', 0)}")
            report.append(f"   Total Readings: {health.get('sentiment_analyzer', {}).get('total_readings', 0)}")

            return "\n".join(report)

        except Exception as e:
            logger.error(f"Error generating dashboard report: {e}")
            return f"🤖 GOD MODE Dashboard Error: {str(e)}"

# Global dashboard instance
dashboard = SentimentDashboard()

async def initialize_dashboard():
    """Initialize the global dashboard."""
    await dashboard.load_dashboard()
    logger.info("🤖 GOD MODE - Sentiment Dashboard ready")

async def update_dashboard(currency_pairs: List[str]) -> Dict[str, Any]:
    """Update the global dashboard."""
    return await dashboard.update_dashboard(currency_pairs)

def get_dashboard_summary() -> Dict[str, Any]:
    """Get dashboard summary."""
    return dashboard.get_dashboard_summary()

async def get_dashboard_report() -> str:
    """Get human-readable dashboard report."""
    return await dashboard.generate_dashboard_report()
