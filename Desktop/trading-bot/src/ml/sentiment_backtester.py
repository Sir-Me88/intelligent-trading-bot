#!/usr/bin/env python3
"""GOD MODE - Sentiment-Based Backtesting System."""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd

from ..news.sentiment import SentimentAggregator, SentimentTrendAnalyzer
from ..analysis.correlation import CorrelationAnalyzer
from ..analysis.entry_timing_optimizer import SentimentEntryTimingOptimizer
from ..risk.sentiment_risk_multiplier import SentimentRiskMultiplier
from ..ml.sentiment_performance_tracker import SentimentTradeRecord

logger = logging.getLogger(__name__)

@dataclass
class BacktestTrade:
    """Trade record for backtesting."""
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    direction: str = 'buy'  # 'buy' or 'sell'
    size: float = 1.0
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    sentiment_score: float = 0.0
    sentiment_confidence: float = 0.0
    risk_multiplier: float = 1.0
    timing_delay: int = 0
    market_regime: str = 'unknown'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_reason: str = 'unknown'

    def calculate_pnl(self):
        """Calculate P&L for the trade."""
        if self.exit_price is None:
            return

        if self.direction.lower() == 'buy':
            self.pnl = (self.exit_price - self.entry_price) * self.size
        else:  # sell
            self.pnl = (self.entry_price - self.exit_price) * self.size

        if self.entry_price != 0:
            self.pnl_percentage = (self.pnl / (self.entry_price * self.size)) * 100

@dataclass
class BacktestResult:
    """Results from a backtest run."""
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_percentage: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percentage: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    avg_trade_duration: float = 0.0
    sentiment_accuracy: float = 0.0
    risk_adjusted_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    trades: List[BacktestTrade] = None

    def __post_init__(self):
        if self.trades is None:
            self.trades = []

class SentimentBacktester:
    """GOD MODE - Advanced sentiment-based backtesting system."""

    def __init__(self):
        self.sentiment_aggregator = SentimentAggregator()
        self.sentiment_trend_analyzer = SentimentTrendAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.entry_timing_optimizer = SentimentEntryTimingOptimizer()
        self.risk_multiplier = SentimentRiskMultiplier()

        # Backtest parameters
        self.commission_per_trade = 0.0002  # 0.02% commission
        self.spread_cost = 0.0001  # 0.01% spread
        self.min_trade_size = 0.01
        self.max_trade_size = 100.0

        # Risk management
        self.max_drawdown_limit = 0.20  # 20% max drawdown
        self.daily_trade_limit = 10
        self.max_open_positions = 5

        logger.info("🤖 GOD MODE - Sentiment Backtester initialized")

    async def run_backtest(self, symbol: str, start_date: datetime, end_date: datetime,
                          strategy_config: Dict[str, Any]) -> BacktestResult:
        """Run a comprehensive sentiment-based backtest."""
        try:
            logger.info(f"🔬 Running GOD MODE backtest for {symbol}: {start_date.date()} to {end_date.date()}")

            # Initialize backtest result
            result = BacktestResult(
                strategy_name=strategy_config.get('name', 'sentiment_strategy'),
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )

            # Get historical data
            historical_data = await self._get_historical_data(symbol, start_date, end_date)
            if historical_data.empty:
                logger.warning(f"No historical data available for {symbol}")
                return result

            # Generate sentiment signals
            signals = await self._generate_sentiment_signals(symbol, historical_data, strategy_config)

            # Execute trades
            trades = await self._execute_backtest_trades(symbol, historical_data, signals, strategy_config)

            # Calculate performance metrics
            result = self._calculate_backtest_metrics(result, trades, historical_data)

            logger.info(f"✅ Backtest completed: {len(trades)} trades, P&L: {result.total_pnl:.2f}")
            return result

        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return BacktestResult(
                strategy_name=strategy_config.get('name', 'error'),
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )

    async def _get_historical_data(self, symbol: str, start_date: datetime,
                                  end_date: datetime) -> pd.DataFrame:
        """Get historical price and sentiment data."""
        try:
            # Get price data
            price_data = await self.sentiment_aggregator.data_manager.get_candles(
                symbol, "H1", limit=1000
            )

            if price_data is None or price_data.empty:
                return pd.DataFrame()

            # Filter by date range
            price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
            price_data = price_data[
                (price_data['timestamp'] >= start_date) &
                (price_data['timestamp'] <= end_date)
            ]

            # Add sentiment data (simplified for backtesting)
            price_data['sentiment_score'] = 0.0
            price_data['sentiment_confidence'] = 0.0

            # Generate synthetic sentiment data for backtesting
            np.random.seed(42)  # For reproducible results
            for idx in price_data.index:
                # Create realistic sentiment patterns
                base_sentiment = np.sin(idx * 0.1) * 0.3  # Cyclical component
                noise = np.random.normal(0, 0.2)  # Random noise
                trend = (idx / len(price_data)) * 0.1  # Slight upward trend

                sentiment = base_sentiment + noise + trend
                sentiment = max(-1.0, min(1.0, sentiment))  # Clamp to [-1, 1]

                price_data.at[idx, 'sentiment_score'] = sentiment
                price_data.at[idx, 'sentiment_confidence'] = 0.7 + np.random.normal(0, 0.1)

            return price_data

        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()

    async def _generate_sentiment_signals(self, symbol: str, data: pd.DataFrame,
                                        strategy_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals based on sentiment analysis."""
        try:
            signals = []
            sentiment_threshold = strategy_config.get('sentiment_threshold', 0.2)
            confidence_threshold = strategy_config.get('confidence_threshold', 0.5)

            for idx, row in data.iterrows():
                sentiment_score = row.get('sentiment_score', 0.0)
                confidence = row.get('sentiment_confidence', 0.0)
                price = row.get('close', 0.0)
                timestamp = row.get('timestamp')

                # Generate signals based on sentiment
                if (abs(sentiment_score) > sentiment_threshold and
                    confidence > confidence_threshold):

                    signal = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'price': price,
                        'sentiment_score': sentiment_score,
                        'sentiment_confidence': confidence,
                        'direction': 'buy' if sentiment_score > 0 else 'sell',
                        'strength': abs(sentiment_score),
                        'type': 'sentiment_signal'
                    }

                    signals.append(signal)

            logger.info(f"Generated {len(signals)} sentiment signals")
            return signals

        except Exception as e:
            logger.error(f"Error generating sentiment signals: {e}")
            return []

    async def _execute_backtest_trades(self, symbol: str, data: pd.DataFrame,
                                     signals: List[Dict[str, Any]],
                                     strategy_config: Dict[str, Any]) -> List[BacktestTrade]:
        """Execute trades based on signals."""
        try:
            trades = []
            open_positions = []
            capital = strategy_config.get('initial_capital', 10000.0)
            risk_per_trade = strategy_config.get('risk_per_trade', 0.02)  # 2% risk per trade

            for signal in signals:
                # Check if we should enter a trade
                if len(open_positions) >= self.max_open_positions:
                    continue

                # Calculate position size based on risk
                stop_loss_pips = strategy_config.get('stop_loss_pips', 50)
                position_size = self._calculate_position_size(capital, risk_per_trade, stop_loss_pips)

                # Create trade
                trade = BacktestTrade(
                    symbol=symbol,
                    entry_time=signal['timestamp'],
                    entry_price=signal['price'],
                    direction=signal['direction'],
                    size=position_size,
                    sentiment_score=signal['sentiment_score'],
                    sentiment_confidence=signal['sentiment_confidence']
                )

                # Set stop loss and take profit
                if signal['direction'] == 'buy':
                    trade.stop_loss = signal['price'] - (stop_loss_pips * 0.0001)  # Assuming 1 pip = 0.0001
                    trade.take_profit = signal['price'] + (stop_loss_pips * 0.0002)  # 2:1 reward ratio
                else:
                    trade.stop_loss = signal['price'] + (stop_loss_pips * 0.0001)
                    trade.take_profit = signal['price'] - (stop_loss_pips * 0.0002)

                open_positions.append(trade)
                trades.append(trade)

                # Simulate trade exit (simplified)
                exit_price = self._simulate_trade_exit(trade, data, signal['timestamp'])
                if exit_price:
                    trade.exit_price = exit_price
                    trade.exit_time = signal['timestamp'] + timedelta(hours=4)  # Assume 4 hour hold
                    trade.calculate_pnl()

                    # Remove from open positions
                    open_positions.remove(trade)

            logger.info(f"Executed {len(trades)} backtest trades")
            return trades

        except Exception as e:
            logger.error(f"Error executing backtest trades: {e}")
            return []

    def _calculate_position_size(self, capital: float, risk_per_trade: float,
                               stop_loss_pips: int) -> float:
        """Calculate position size based on risk management."""
        try:
            # Risk amount in account currency
            risk_amount = capital * risk_per_trade

            # Stop loss in price terms (simplified)
            stop_loss_price = stop_loss_pips * 0.0001  # Assuming EURUSD

            # Position size = Risk amount / Stop loss
            position_size = risk_amount / stop_loss_price

            # Ensure within bounds
            position_size = max(self.min_trade_size, min(self.max_trade_size, position_size))

            return position_size

        except Exception:
            return self.min_trade_size

    def _simulate_trade_exit(self, trade: BacktestTrade, data: pd.DataFrame,
                           entry_time: datetime) -> Optional[float]:
        """Simulate trade exit based on stop loss or take profit."""
        try:
            # Find exit conditions in historical data
            entry_idx = data[data['timestamp'] >= entry_time].index[0]

            # Check next 24 hours (24 candles at H1)
            for i in range(min(24, len(data) - entry_idx)):
                current_idx = entry_idx + i
                current_price = data.iloc[current_idx]['close']

                # Check stop loss
                if trade.direction == 'buy':
                    if current_price <= trade.stop_loss:
                        return trade.stop_loss
                    elif current_price >= trade.take_profit:
                        return trade.take_profit
                else:  # sell
                    if current_price >= trade.stop_loss:
                        return trade.stop_loss
                    elif current_price <= trade.take_profit:
                        return trade.take_profit

            # If no exit triggered, close at current price
            return data.iloc[min(entry_idx + 23, len(data) - 1)]['close']

        except Exception:
            return None

    def _calculate_backtest_metrics(self, result: BacktestResult, trades: List[BacktestTrade],
                                  data: pd.DataFrame) -> BacktestResult:
        """Calculate comprehensive backtest performance metrics."""
        try:
            if not trades:
                return result

            # Filter completed trades
            completed_trades = [t for t in trades if t.exit_price is not None and t.pnl is not None]

            if not completed_trades:
                return result

            # Basic metrics
            result.total_trades = len(completed_trades)
            result.winning_trades = len([t for t in completed_trades if t.pnl > 0])
            result.losing_trades = result.total_trades - result.winning_trades
            result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0.0

            # P&L metrics
            result.total_pnl = sum(t.pnl for t in completed_trades if t.pnl)
            result.total_pnl_percentage = (result.total_pnl / 10000.0) * 100  # Assuming $10k starting capital

            winning_pnls = [t.pnl for t in completed_trades if t.pnl > 0]
            losing_pnls = [t.pnl for t in completed_trades if t.pnl < 0]

            result.avg_win = np.mean(winning_pnls) if winning_pnls else 0.0
            result.avg_loss = abs(np.mean(losing_pnls)) if losing_pnls else 0.0

            if losing_pnls and sum(losing_pnls) != 0:
                result.profit_factor = sum(winning_pnls) / abs(sum(losing_pnls))
            else:
                result.profit_factor = float('inf')

            # Drawdown analysis
            result.max_drawdown, result.max_drawdown_percentage = self._calculate_drawdown(completed_trades)

            # Risk-adjusted metrics
            returns = [t.pnl_percentage for t in completed_trades if t.pnl_percentage]
            if returns:
                result.sharpe_ratio = self._calculate_sharpe_ratio(returns)
                result.sortino_ratio = self._calculate_sortino_ratio(returns)
                result.calmar_ratio = result.total_pnl_percentage / result.max_drawdown_percentage if result.max_drawdown_percentage > 0 else 0.0

            # Duration metrics
            durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in completed_trades if t.exit_time]
            result.avg_trade_duration = np.mean(durations) if durations else 0.0

            # Sentiment accuracy
            result.sentiment_accuracy = self._calculate_backtest_sentiment_accuracy(completed_trades)

            # Risk-adjusted return
            result.risk_adjusted_return = self._calculate_backtest_risk_adjusted_return(completed_trades)

            # Market beta (simplified)
            result.beta = self._calculate_market_beta(completed_trades, data)

            result.trades = completed_trades

            return result

        except Exception as e:
            logger.error(f"Error calculating backtest metrics: {e}")
            return result

    def _calculate_drawdown(self, trades: List[BacktestTrade]) -> Tuple[float, float]:
        """Calculate maximum drawdown."""
        try:
            if not trades:
                return 0.0, 0.0

            # Sort trades by exit time
            sorted_trades = sorted(trades, key=lambda x: x.exit_time or datetime.now())

            cumulative_pnl = 0.0
            peak = 0.0
            max_drawdown = 0.0
            max_drawdown_percentage = 0.0

            for trade in sorted_trades:
                if trade.pnl:
                    cumulative_pnl += trade.pnl
                    peak = max(peak, cumulative_pnl)
                    drawdown = peak - cumulative_pnl
                    drawdown_percentage = (drawdown / peak) * 100 if peak > 0 else 0.0

                    max_drawdown = max(max_drawdown, drawdown)
                    max_drawdown_percentage = max(max_drawdown_percentage, drawdown_percentage)

            return max_drawdown, max_drawdown_percentage

        except Exception:
            return 0.0, 0.0

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio."""
        try:
            if len(returns) < 2:
                return 0.0

            avg_return = np.mean(returns)
            std_return = np.std(returns)

            # Assume risk-free rate of 2% annually (0.02/252 for daily, adjusted for hourly)
            risk_free_rate = 0.02 / (252 * 24)  # Hourly risk-free rate

            if std_return == 0:
                return 0.0

            return (avg_return - risk_free_rate) / std_return

        except Exception:
            return 0.0

    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        try:
            if len(returns) < 2:
                return 0.0

            avg_return = np.mean(returns)
            downside_returns = [r for r in returns if r < 0]
            downside_std = np.std(downside_returns) if downside_returns else 0.0

            risk_free_rate = 0.02 / (252 * 24)

            if downside_std == 0:
                return 0.0

            return (avg_return - risk_free_rate) / downside_std

        except Exception:
            return 0.0

    def _calculate_backtest_sentiment_accuracy(self, trades: List[BacktestTrade]) -> float:
        """Calculate sentiment prediction accuracy in backtest."""
        try:
            correct_predictions = 0

            for trade in trades:
                if trade.pnl and trade.sentiment_score:
                    # Check if sentiment direction aligned with profitable outcome
                    sentiment_direction = 1 if trade.sentiment_score > 0 else -1
                    pnl_direction = 1 if trade.pnl > 0 else -1

                    if sentiment_direction == pnl_direction:
                        correct_predictions += 1

            return correct_predictions / len(trades) if trades else 0.0

        except Exception:
            return 0.0

    def _calculate_backtest_risk_adjusted_return(self, trades: List[BacktestTrade]) -> float:
        """Calculate risk-adjusted return for backtest."""
        try:
            if not trades:
                return 0.0

            total_return = sum(t.pnl for t in trades if t.pnl) / 10000.0  # Assuming $10k starting
            total_risk = sum(t.risk_multiplier for t in trades) / len(trades)

            if total_risk == 0:
                return 0.0

            return total_return / total_risk

        except Exception:
            return 0.0

    def _calculate_market_beta(self, trades: List[BacktestTrade], data: pd.DataFrame) -> float:
        """Calculate market beta (simplified)."""
        try:
            if not trades or data.empty:
                return 1.0

            # Use price returns as market proxy
            market_returns = data['close'].pct_change().dropna().values

            if len(market_returns) < len(trades):
                return 1.0

            # Strategy returns
            strategy_returns = []
            for trade in trades:
                if trade.pnl_percentage:
                    strategy_returns.append(trade.pnl_percentage / 100.0)

            if len(strategy_returns) != len(market_returns[:len(strategy_returns)]):
                return 1.0

            # Calculate beta
            covariance = np.cov(strategy_returns, market_returns[:len(strategy_returns)])[0, 1]
            market_variance = np.var(market_returns[:len(strategy_returns)])

            if market_variance == 0:
                return 1.0

            return covariance / market_variance

        except Exception:
            return 1.0

    async def compare_strategies(self, symbol: str, start_date: datetime, end_date: datetime,
                               strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple sentiment-based strategies."""
        try:
            results = {}

            for strategy in strategies:
                result = await self.run_backtest(symbol, start_date, end_date, strategy)
                results[strategy['name']] = result

            # Generate comparison report
            comparison = self._generate_strategy_comparison(results)

            return {
                'individual_results': {name: self._backtest_result_to_dict(result)
                                     for name, result in results.items()},
                'comparison': comparison
            }

        except Exception as e:
            logger.error(f"Error comparing strategies: {e}")
            return {'error': str(e)}

    def _generate_strategy_comparison(self, results: Dict[str, BacktestResult]) -> Dict[str, Any]:
        """Generate comparison metrics between strategies."""
        try:
            if not results:
                return {}

            # Find best performing strategy for each metric
            metrics = ['win_rate', 'total_pnl', 'profit_factor', 'sharpe_ratio', 'sentiment_accuracy']

            comparison = {}
            for metric in metrics:
                best_strategy = None
                best_value = float('-inf')

                for name, result in results.items():
                    value = getattr(result, metric, 0.0)
                    if value > best_value:
                        best_value = value
                        best_strategy = name

                comparison[f'best_{metric}'] = {
                    'strategy': best_strategy,
                    'value': best_value
                }

            # Overall ranking
            rankings = {}
            for name, result in results.items():
                score = (
                    result.win_rate * 0.3 +
                    min(result.profit_factor / 3.0, 1.0) * 0.3 +  # Cap at 3.0
                    (result.sharpe_ratio + 2.0) / 4.0 * 0.2 +  # Normalize around 0
                    result.sentiment_accuracy * 0.2
                )
                rankings[name] = score

            best_overall = max(rankings.items(), key=lambda x: x[1])

            comparison['overall_ranking'] = {
                'best_strategy': best_overall[0],
                'score': best_overall[1],
                'all_scores': rankings
            }

            return comparison

        except Exception as e:
            logger.error(f"Error generating strategy comparison: {e}")
            return {}

    def _backtest_result_to_dict(self, result: BacktestResult) -> Dict[str, Any]:
        """Convert BacktestResult to dictionary."""
        try:
            data = asdict(result)
            # Convert datetime objects to ISO strings
            data['start_date'] = result.start_date.isoformat()
            data['end_date'] = result.end_date.isoformat()
            # Remove trades list to reduce size (can be added back if needed)
            data.pop('trades', None)
            return data
        except Exception:
            return {}

    async def generate_backtest_report(self, result: BacktestResult) -> str:
        """Generate human-readable backtest report."""
        try:
            report = []
            report.append("🤖 GOD MODE SENTIMENT BACKTEST REPORT")
            report.append("=" * 50)
            report.append(f"Strategy: {result.strategy_name}")
            report.append(f"Symbol: {result.symbol}")
            report.append(f"Period: {result.start_date.date()} to {result.end_date.date()}")
            report.append("")

            report.append("📊 PERFORMANCE METRICS")
            report.append(f"   Total Trades: {result.total_trades}")
            report.append(f"   Win Rate: {result.win_rate:.1%}")
            report.append(f"   Total P&L: ${result.total_pnl:.2f}")
            report.append(f"   Total Return: {result.total_pnl_percentage:.1f}%")
            report.append(f"   Profit Factor: {result.profit_factor:.2f}")
            report.append(f"   Average Win: ${result.avg_win:.2f}")
            report.append(f"   Average Loss: ${result.avg_loss:.2f}")
            report.append("")

            report.append("🎯 RISK METRICS")
            report.append(f"   Max Drawdown: ${result.max_drawdown:.2f}")
            report.append(f"   Max Drawdown %: {result.max_drawdown_percentage:.1f}%")
            report.append(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
            report.append(f"   Sortino Ratio: {result.sortino_ratio:.2f}")
            report.append(f"   Calmar Ratio: {result.calmar_ratio:.2f}")
            report.append("")

            report.append("🧠 SENTIMENT ANALYSIS")
            report.append(f"   Sentiment Accuracy: {result.sentiment_accuracy:.1%}")
            report.append(f"   Risk-Adjusted Return: {result.risk_adjusted_return:.2f}")
            report.append(f"   Market Beta: {result.beta:.2f}")
            report.append(f"   Average Trade Duration: {result.avg_trade_duration:.1f} hours")
            report.append("")

            # Trade analysis
            if result.trades:
                report.append("📈 TRADE ANALYSIS")
                profitable_trades = [t for t in result.trades if t.pnl and t.pnl > 0]
                unprofitable_trades = [t for t in result.trades if t.pnl and t.pnl <= 0]

                if profitable_trades:
                    best_trade = max(profitable_trades, key=lambda x: x.pnl)
                    report.append(f"   Best Trade: ${best_trade.pnl:.2f} ({best_trade.pnl_percentage:.1f}%)")

                if unprofitable_trades:
                    worst_trade = min(unprofitable_trades, key=lambda x: x.pnl)
                    report.append(f"   Worst Trade: ${worst_trade.pnl:.2f} ({worst_trade.pnl_percentage:.1f}%)")

                # Sentiment distribution
                sentiment_ranges = {
                    'Strong Positive': len([t for t in result.trades if t.sentiment_score > 0.4]),
                    'Moderate Positive': len([t for t in result.trades if 0.2 <= t.sentiment_score <= 0.4]),
                    'Neutral': len([t for t in result.trades if -0.2 <= t.sentiment_score <= 0.2]),
                    'Moderate Negative': len([t for t in result.trades if -0.4 <= t.sentiment_score <= -0.2]),
                    'Strong Negative': len([t for t in result.trades if t.sentiment_score < -0.4])
                }

                report.append("   Sentiment Distribution:")
                for sentiment_range, count in sentiment_ranges.items():
                    if count > 0:
                        report.append(f"     {sentiment_range}: {count} trades")

            return "\n".join(report)

        except Exception as e:
            logger.error(f"Error generating backtest report: {e}")
            return f"🤖 GOD MODE Backtest Error: {str(e)}"

# Global backtester instance
sentiment_backtester = SentimentBacktester()

async def run_sentiment_backtest(symbol: str, start_date: datetime, end_date: datetime,
                               strategy_config: Dict[str, Any]) -> BacktestResult:
    """Convenience function to run sentiment backtest."""
    return await sentiment_backtester.run_backtest(symbol, start_date, end_date, strategy_config)

async def compare_sentiment_strategies(symbol: str, start_date: datetime, end_date: datetime,
                                     strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convenience function to compare sentiment strategies."""
    return await sentiment_backtester.compare_strategies(symbol, start_date, end_date, strategies)

async def generate_backtest_report(result: BacktestResult) -> str:
    """Convenience function to generate backtest report."""
    return await sentiment_backtester.generate_backtest_report(result)
