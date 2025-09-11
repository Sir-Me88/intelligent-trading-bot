#!/usr/bin/env python3
"""GOD MODE - Sentiment Model Performance Tracking System."""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd

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
from ..analysis.entry_timing_optimizer import SentimentEntryTimingOptimizer
from ..risk.sentiment_risk_multiplier import SentimentRiskMultiplier

logger = logging.getLogger(__name__)

@dataclass
class SentimentTradeRecord:
    """Record of a sentiment-influenced trade for performance tracking."""
    trade_id: str
    symbol: str
    timestamp: datetime
    sentiment_score: float
    sentiment_confidence: float
    trade_direction: str  # 'buy' or 'sell'
    entry_price: float
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    trade_duration: Optional[timedelta] = None
    sentiment_adjustments: Dict[str, Any] = None
    risk_multiplier: float = 1.0
    timing_delay: int = 0
    market_regime: str = 'unknown'
    exit_reason: str = 'unknown'

    def __post_init__(self):
        if self.sentiment_adjustments is None:
            self.sentiment_adjustments = {}

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        if self.exit_timestamp:
            data['exit_timestamp'] = self.exit_timestamp.isoformat()
        if self.trade_duration:
            data['trade_duration'] = self.trade_duration.total_seconds()
        return data

@dataclass
class SentimentPerformanceMetrics:
    """Performance metrics for sentiment analysis."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    avg_trade_duration: float = 0.0
    sentiment_accuracy: float = 0.0
    risk_adjusted_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    recovery_factor: float = 0.0

class SentimentPerformanceTracker:
    """GOD MODE - Advanced performance tracking for sentiment-based trading."""

    def __init__(self):
        self.sentiment_aggregator = SentimentAggregator()
        self.sentiment_trend_analyzer = SentimentTrendAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.entry_timing_optimizer = SentimentEntryTimingOptimizer()
        self.risk_multiplier = SentimentRiskMultiplier()

        # Performance data
        self.trade_records: List[SentimentTradeRecord] = []
        self.performance_history: List[SentimentPerformanceMetrics] = []

        # Persistence
        self.performance_file = Path("data/performance/sentiment_performance.json")
        self.performance_file.parent.mkdir(parents=True, exist_ok=True)

        # Analysis parameters
        self.min_trades_for_analysis = 10
        self.performance_update_interval = 3600  # 1 hour
        self.last_update = None

        logger.info("🤖 GOD MODE - Sentiment Performance Tracker initialized")

    async def record_sentiment_trade(self, trade_id: str, symbol: str, sentiment_score: float,
                                   sentiment_confidence: float, trade_direction: str,
                                   entry_price: float, risk_multiplier: float = 1.0,
                                   timing_delay: int = 0) -> str:
        """Record a new sentiment-influenced trade."""
        try:
            # Get current market regime
            trend_data = await self.sentiment_trend_analyzer.analyze_sentiment_trend(symbol, hours_back=6)
            market_regime = trend_data.get('regime', 'unknown')

            # Get sentiment adjustments
            sentiment_adjustments = {
                'sentiment_score': sentiment_score,
                'sentiment_confidence': sentiment_confidence,
                'market_regime': market_regime,
                'trend_strength': trend_data.get('strength', 0.0),
                'volatility': trend_data.get('volatility', 0.0)
            }

            # Create trade record
            trade_record = SentimentTradeRecord(
                trade_id=trade_id,
                symbol=symbol,
                timestamp=datetime.now(),
                sentiment_score=sentiment_score,
                sentiment_confidence=sentiment_confidence,
                trade_direction=trade_direction,
                entry_price=entry_price,
                risk_multiplier=risk_multiplier,
                timing_delay=timing_delay,
                market_regime=market_regime,
                sentiment_adjustments=sentiment_adjustments
            )

            self.trade_records.append(trade_record)

            # Save to file
            await self._save_trade_records()

            logger.info(f"📊 Recorded sentiment trade: {trade_id} for {symbol}")
            return trade_id

        except Exception as e:
            logger.error(f"Error recording sentiment trade: {e}")
            return ""

    async def update_trade_exit(self, trade_id: str, exit_price: float,
                              exit_reason: str = "manual") -> bool:
        """Update trade record with exit information."""
        try:
            # Find trade record
            trade_record = None
            for record in self.trade_records:
                if record.trade_id == trade_id:
                    trade_record = record
                    break

            if not trade_record:
                logger.warning(f"Trade record not found: {trade_id}")
                return False

            # Update exit information
            exit_timestamp = datetime.now()
            trade_duration = exit_timestamp - trade_record.timestamp

            # Calculate P&L
            if trade_record.trade_direction.lower() == 'buy':
                pnl = exit_price - trade_record.entry_price
            else:  # sell
                pnl = trade_record.entry_price - exit_price

            pnl_percentage = (pnl / trade_record.entry_price) * 100

            # Update record
            trade_record.exit_price = exit_price
            trade_record.exit_timestamp = exit_timestamp
            trade_record.pnl = pnl
            trade_record.pnl_percentage = pnl_percentage
            trade_record.trade_duration = trade_duration
            trade_record.exit_reason = exit_reason

            # Save updated records
            await self._save_trade_records()

            logger.info(f"📊 Updated trade exit: {trade_id}, P&L: {pnl:.2f}")
            return True

        except Exception as e:
            logger.error(f"Error updating trade exit: {e}")
            return False

    async def calculate_performance_metrics(self) -> SentimentPerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        try:
            if len(self.trade_records) < self.min_trades_for_analysis:
                logger.warning("Insufficient trades for performance analysis")
                return SentimentPerformanceMetrics()

            # Filter completed trades
            completed_trades = [t for t in self.trade_records if t.exit_price is not None and t.pnl is not None]

            if not completed_trades:
                return SentimentPerformanceMetrics()

            # Basic metrics
            total_trades = len(completed_trades)
            winning_trades = len([t for t in completed_trades if t.pnl > 0])
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

            # P&L metrics
            winning_pnls = [t.pnl for t in completed_trades if t.pnl > 0]
            losing_pnls = [t.pnl for t in completed_trades if t.pnl < 0]

            avg_win = np.mean(winning_pnls) if winning_pnls else 0.0
            avg_loss = abs(np.mean(losing_pnls)) if losing_pnls else 0.0
            profit_factor = (sum(winning_pnls) / abs(sum(losing_pnls))) if losing_pnls and sum(losing_pnls) != 0 else float('inf')

            total_pnl = sum(t.pnl for t in completed_trades)

            # Duration metrics
            avg_trade_duration = np.mean([t.trade_duration.total_seconds() / 3600 for t in completed_trades if t.trade_duration])  # hours

            # Sentiment accuracy
            sentiment_accuracy = self._calculate_sentiment_accuracy(completed_trades)

            # Risk-adjusted metrics
            returns = [t.pnl_percentage for t in completed_trades]
            risk_adjusted_return = self._calculate_risk_adjusted_return(returns, completed_trades)

            # Sharpe ratio (simplified)
            if len(returns) > 1:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = avg_return / std_return if std_return > 0 else 0.0
            else:
                sharpe_ratio = 0.0

            # Drawdown analysis
            max_drawdown = self._calculate_max_drawdown(completed_trades)

            # Recovery factor
            recovery_factor = total_pnl / max_drawdown if max_drawdown > 0 else float('inf')

            return SentimentPerformanceMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                total_pnl=total_pnl,
                avg_trade_duration=avg_trade_duration,
                sentiment_accuracy=sentiment_accuracy,
                risk_adjusted_return=risk_adjusted_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                recovery_factor=recovery_factor
            )

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return SentimentPerformanceMetrics()

    def _calculate_sentiment_accuracy(self, trades: List[SentimentTradeRecord]) -> float:
        """Calculate how well sentiment predictions aligned with trade outcomes."""
        try:
            correct_predictions = 0

            for trade in trades:
                sentiment_direction = 1 if trade.sentiment_score > 0 else -1
                trade_direction = 1 if trade.trade_direction.lower() == 'buy' else -1
                pnl_direction = 1 if trade.pnl > 0 else -1

                # Check if sentiment aligned with profitable outcome
                if sentiment_direction == pnl_direction:
                    correct_predictions += 1

            return correct_predictions / len(trades) if trades else 0.0

        except Exception:
            return 0.0

    def _calculate_risk_adjusted_return(self, returns: List[float], trades: List[SentimentTradeRecord]) -> float:
        """Calculate risk-adjusted return based on risk multipliers."""
        try:
            if not returns or not trades:
                return 0.0

            # Weight returns by risk multiplier
            risk_weighted_returns = []
            for i, trade in enumerate(trades):
                if i < len(returns):
                    risk_weight = 1.0 / trade.risk_multiplier  # Higher risk = lower weight
                    risk_weighted_returns.append(returns[i] * risk_weight)

            return np.mean(risk_weighted_returns) if risk_weighted_returns else 0.0

        except Exception:
            return 0.0

    def _calculate_max_drawdown(self, trades: List[SentimentTradeRecord]) -> float:
        """Calculate maximum drawdown from peak to trough."""
        try:
            if not trades:
                return 0.0

            # Sort trades by timestamp
            sorted_trades = sorted(trades, key=lambda x: x.timestamp)

            # Calculate cumulative P&L
            cumulative_pnl = 0.0
            peak = 0.0
            max_drawdown = 0.0

            for trade in sorted_trades:
                if trade.pnl is not None:
                    cumulative_pnl += trade.pnl
                    peak = max(peak, cumulative_pnl)
                    drawdown = peak - cumulative_pnl
                    max_drawdown = max(max_drawdown, drawdown)

            return max_drawdown

        except Exception:
            return 0.0

    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            metrics = await self.calculate_performance_metrics()

            # Additional analysis
            sentiment_effectiveness = await self._analyze_sentiment_effectiveness()
            market_regime_performance = self._analyze_market_regime_performance()
            risk_multiplier_performance = self._analyze_risk_multiplier_performance()
            timing_performance = self._analyze_timing_performance()

            report = {
                'timestamp': datetime.now().isoformat(),
                'period': 'all_time',
                'performance_metrics': {
                    'total_trades': metrics.total_trades,
                    'win_rate': f"{metrics.win_rate:.1%}",
                    'total_pnl': f"{metrics.total_pnl:.2f}",
                    'avg_win': f"{metrics.avg_win:.2f}",
                    'avg_loss': f"{metrics.avg_loss:.2f}",
                    'profit_factor': f"{metrics.profit_factor:.2f}",
                    'sentiment_accuracy': f"{metrics.sentiment_accuracy:.1%}",
                    'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
                    'max_drawdown': f"{metrics.max_drawdown:.2f}",
                    'recovery_factor': f"{metrics.recovery_factor:.2f}"
                },
                'sentiment_analysis': sentiment_effectiveness,
                'market_regime_performance': market_regime_performance,
                'risk_multiplier_performance': risk_multiplier_performance,
                'timing_performance': timing_performance,
                'recommendations': self._generate_performance_recommendations(metrics, sentiment_effectiveness)
            }

            return report

        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}

    async def _analyze_sentiment_effectiveness(self) -> Dict[str, Any]:
        """Analyze how effective sentiment analysis has been."""
        try:
            completed_trades = [t for t in self.trade_records if t.exit_price is not None]

            if len(completed_trades) < 5:
                return {'status': 'insufficient_data'}

            # Group by sentiment strength
            strong_sentiment_trades = [t for t in completed_trades if abs(t.sentiment_score) > 0.4]
            moderate_sentiment_trades = [t for t in completed_trades if 0.2 <= abs(t.sentiment_score) <= 0.4]
            weak_sentiment_trades = [t for t in completed_trades if abs(t.sentiment_score) < 0.2]

            def calculate_win_rate(trades):
                if not trades:
                    return 0.0
                winning = len([t for t in trades if t.pnl > 0])
                return winning / len(trades)

            return {
                'strong_sentiment_win_rate': calculate_win_rate(strong_sentiment_trades),
                'moderate_sentiment_win_rate': calculate_win_rate(moderate_sentiment_trades),
                'weak_sentiment_win_rate': calculate_win_rate(weak_sentiment_trades),
                'strong_sentiment_count': len(strong_sentiment_trades),
                'moderate_sentiment_count': len(moderate_sentiment_trades),
                'weak_sentiment_count': len(weak_sentiment_trades),
                'best_sentiment_range': self._find_best_sentiment_range(completed_trades)
            }

        except Exception as e:
            logger.error(f"Error analyzing sentiment effectiveness: {e}")
            return {'error': str(e)}

    def _find_best_sentiment_range(self, trades: List[SentimentTradeRecord]) -> str:
        """Find the sentiment range with best performance."""
        try:
            ranges = {
                'strong_positive': [t for t in trades if t.sentiment_score > 0.4],
                'moderate_positive': [t for t in trades if 0.2 <= t.sentiment_score <= 0.4],
                'weak_positive': [t for t in trades if 0.1 <= t.sentiment_score < 0.2],
                'neutral': [t for t in trades if -0.1 <= t.sentiment_score <= 0.1],
                'weak_negative': [t for t in trades if -0.2 <= t.sentiment_score < -0.1],
                'moderate_negative': [t for t in trades if -0.4 <= t.sentiment_score <= -0.2],
                'strong_negative': [t for t in trades if t.sentiment_score < -0.4]
            }

            best_range = None
            best_win_rate = 0.0

            for range_name, range_trades in ranges.items():
                if len(range_trades) >= 3:  # Minimum trades for significance
                    winning = len([t for t in range_trades if t.pnl > 0])
                    win_rate = winning / len(range_trades)
                    if win_rate > best_win_rate:
                        best_win_rate = win_rate
                        best_range = range_name

            return best_range or 'insufficient_data'

        except Exception:
            return 'error'

    def _analyze_market_regime_performance(self) -> Dict[str, Any]:
        """Analyze performance by market regime."""
        try:
            completed_trades = [t for t in self.trade_records if t.exit_price is not None]

            if len(completed_trades) < 5:
                return {'status': 'insufficient_data'}

            # Group by market regime
            regime_performance = {}
            regimes = set(t.market_regime for t in completed_trades)

            for regime in regimes:
                regime_trades = [t for t in completed_trades if t.market_regime == regime]
                if len(regime_trades) >= 3:
                    winning = len([t for t in regime_trades if t.pnl > 0])
                    win_rate = winning / len(regime_trades)
                    avg_pnl = np.mean([t.pnl for t in regime_trades])

                    regime_performance[regime] = {
                        'win_rate': win_rate,
                        'avg_pnl': avg_pnl,
                        'trade_count': len(regime_trades)
                    }

            return regime_performance

        except Exception as e:
            logger.error(f"Error analyzing market regime performance: {e}")
            return {'error': str(e)}

    def _analyze_risk_multiplier_performance(self) -> Dict[str, Any]:
        """Analyze performance by risk multiplier."""
        try:
            completed_trades = [t for t in self.trade_records if t.exit_price is not None]

            if len(completed_trades) < 5:
                return {'status': 'insufficient_data'}

            # Group by risk multiplier ranges
            risk_ranges = {
                'low_risk': [t for t in completed_trades if t.risk_multiplier < 0.7],
                'moderate_risk': [t for t in completed_trades if 0.7 <= t.risk_multiplier <= 1.3],
                'high_risk': [t for t in completed_trades if t.risk_multiplier > 1.3]
            }

            risk_performance = {}
            for risk_level, trades in risk_ranges.items():
                if len(trades) >= 3:
                    winning = len([t for t in trades if t.pnl > 0])
                    win_rate = winning / len(trades)
                    avg_pnl = np.mean([t.pnl for t in trades])
                    avg_risk = np.mean([t.risk_multiplier for t in trades])

                    risk_performance[risk_level] = {
                        'win_rate': win_rate,
                        'avg_pnl': avg_pnl,
                        'avg_risk_multiplier': avg_risk,
                        'trade_count': len(trades)
                    }

            return risk_performance

        except Exception as e:
            logger.error(f"Error analyzing risk multiplier performance: {e}")
            return {'error': str(e)}

    def _analyze_timing_performance(self) -> Dict[str, Any]:
        """Analyze performance by entry timing."""
        try:
            completed_trades = [t for t in self.trade_records if t.exit_price is not None]

            if len(completed_trades) < 5:
                return {'status': 'insufficient_data'}

            # Group by timing delay
            timing_ranges = {
                'immediate': [t for t in completed_trades if t.timing_delay <= 15],
                'short_delay': [t for t in completed_trades if 15 < t.timing_delay <= 60],
                'long_delay': [t for t in completed_trades if t.timing_delay > 60]
            }

            timing_performance = {}
            for timing_type, trades in timing_ranges.items():
                if len(trades) >= 3:
                    winning = len([t for t in trades if t.pnl > 0])
                    win_rate = winning / len(trades)
                    avg_pnl = np.mean([t.pnl for t in trades])
                    avg_delay = np.mean([t.timing_delay for t in trades])

                    timing_performance[timing_type] = {
                        'win_rate': win_rate,
                        'avg_pnl': avg_pnl,
                        'avg_delay_minutes': avg_delay,
                        'trade_count': len(trades)
                    }

            return timing_performance

        except Exception as e:
            logger.error(f"Error analyzing timing performance: {e}")
            return {'error': str(e)}

    def _generate_performance_recommendations(self, metrics: SentimentPerformanceMetrics,
                                            sentiment_analysis: Dict[str, Any]) -> List[str]:
        """Generate performance-based recommendations."""
        try:
            recommendations = []

            # Win rate analysis
            if metrics.win_rate < 0.4:
                recommendations.append("Consider adjusting sentiment thresholds - win rate below 40%")
            elif metrics.win_rate > 0.65:
                recommendations.append("Excellent win rate - consider maintaining current sentiment strategy")

            # Sentiment effectiveness
            best_range = sentiment_analysis.get('best_sentiment_range', '')
            if best_range and best_range != 'insufficient_data':
                recommendations.append(f"Focus on {best_range.replace('_', ' ')} sentiment conditions for best results")

            # Risk management
            if metrics.profit_factor < 1.5:
                recommendations.append("Consider improving risk management - profit factor below 1.5")
            elif metrics.profit_factor > 2.5:
                recommendations.append("Strong risk management - profit factor above 2.5")

            # Drawdown analysis
            if metrics.max_drawdown > abs(metrics.total_pnl) * 0.5:
                recommendations.append("High drawdown detected - consider implementing stricter risk controls")

            # Sample size
            if metrics.total_trades < 50:
                recommendations.append("Limited sample size - continue gathering more performance data")

            return recommendations if recommendations else ["Performance analysis shows balanced results - maintain current strategy"]

        except Exception:
            return ["Unable to generate specific recommendations"]

    async def _save_trade_records(self):
        """Save trade records to file."""
        try:
            records_data = [record.to_dict() for record in self.trade_records]
            with open(self.performance_file, 'w') as f:
                json.dump(records_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving trade records: {e}")

    async def load_trade_records(self):
        """Load trade records from file."""
        try:
            if self.performance_file.exists():
                with open(self.performance_file, 'r') as f:
                    records_data = json.load(f)

                self.trade_records = []
                for record_data in records_data:
                    # Convert timestamps back to datetime
                    record_data['timestamp'] = datetime.fromisoformat(record_data['timestamp'])
                    if record_data.get('exit_timestamp'):
                        record_data['exit_timestamp'] = datetime.fromisoformat(record_data['exit_timestamp'])
                    if record_data.get('trade_duration'):
                        record_data['trade_duration'] = timedelta(seconds=record_data['trade_duration'])

                    # Create SentimentTradeRecord
                    trade_record = SentimentTradeRecord(**record_data)
                    self.trade_records.append(trade_record)

                logger.info(f"✅ Loaded {len(self.trade_records)} trade records")

        except Exception as e:
            logger.error(f"Error loading trade records: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get quick performance summary."""
        try:
            if not self.trade_records:
                return {'status': 'no_data', 'message': 'No performance data available'}

            completed_trades = [t for t in self.trade_records if t.exit_price is not None]
            active_trades = len(self.trade_records) - len(completed_trades)

            if completed_trades:
                total_pnl = sum(t.pnl for t in completed_trades if t.pnl)
                win_rate = len([t for t in completed_trades if t.pnl and t.pnl > 0]) / len(completed_trades)
            else:
                total_pnl = 0.0
                win_rate = 0.0

            return {
                'status': 'active',
                'total_trades': len(self.trade_records),
                'completed_trades': len(completed_trades),
                'active_trades': active_trades,
                'total_pnl': round(total_pnl, 2),
                'win_rate': round(win_rate, 3),
                'last_update': self.last_update.isoformat() if self.last_update else None
            }

        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {'status': 'error', 'message': str(e)}

# Global performance tracker instance
sentiment_performance_tracker = SentimentPerformanceTracker()

async def record_sentiment_trade(trade_id: str, symbol: str, sentiment_score: float,
                               sentiment_confidence: float, trade_direction: str,
                               entry_price: float, risk_multiplier: float = 1.0,
                               timing_delay: int = 0) -> str:
    """Convenience function to record a sentiment trade."""
    return await sentiment_performance_tracker.record_sentiment_trade(
        trade_id, symbol, sentiment_score, sentiment_confidence,
        trade_direction, entry_price, risk_multiplier, timing_delay
    )

async def update_trade_exit(trade_id: str, exit_price: float, exit_reason: str = "manual") -> bool:
    """Convenience function to update trade exit."""
    return await sentiment_performance_tracker.update_trade_exit(trade_id, exit_price, exit_reason)

async def get_performance_report() -> Dict[str, Any]:
    """Convenience function to get performance report."""
    return await sentiment_performance_tracker.generate_performance_report()

def get_performance_summary() -> Dict[str, Any]:
    """Convenience function to get performance summary."""
    return sentiment_performance_tracker.get_performance_summary()
