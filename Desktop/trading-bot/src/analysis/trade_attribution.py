"""Advanced trade attribution and performance analysis system."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """Container for attribution analysis results."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    strategy_contributions: Dict[str, float]
    market_condition_performance: Dict[str, Dict[str, float]]
    parameter_impact: Dict[str, Dict[str, float]]
    time_based_returns: Dict[str, float]
    risk_adjusted_contributions: Dict[str, float]


class TradeAttributionAnalyzer:
    """Advanced trade attribution and performance analysis system."""

    def __init__(self):
        self.trade_history = []
        self.performance_cache = {}
        self.attribution_cache = {}

    def add_trade(self, trade_data: Dict[str, Any]):
        """Add a trade to the attribution analysis."""
        try:
            # Standardize trade data
            standardized_trade = {
                'timestamp': trade_data.get('timestamp'),
                'ticket': trade_data.get('ticket'),
                'symbol': trade_data.get('symbol'),
                'direction': trade_data.get('direction'),
                'entry_price': trade_data.get('entry_price'),
                'exit_price': trade_data.get('exit_price', 0),
                'stop_loss': trade_data.get('stop_loss'),
                'take_profit': trade_data.get('take_profit'),
                'volume': trade_data.get('volume', 0),
                'profit': trade_data.get('profit', 0),
                'commission': trade_data.get('commission', 0),
                'swap': trade_data.get('swap', 0),
                'exit_reason': trade_data.get('exit_reason', 'manual'),
                'hold_duration': trade_data.get('hold_duration', 0),
                'confidence': trade_data.get('confidence', 0),
                'adaptive_params_used': trade_data.get('adaptive_params_used', {}),
                'market_conditions': trade_data.get('market_conditions', {}),
                'strategy_signals': trade_data.get('strategy_signals', []),
                'sentiment_data': trade_data.get('sentiment_data', {}),
                'correlation_data': trade_data.get('correlation_data', {}),
                'volatility_level': trade_data.get('volatility_level', 'normal'),
                'session': trade_data.get('session', 'normal'),
                'spread_pips': trade_data.get('spread_pips', 0)
            }

            self.trade_history.append(standardized_trade)
            logger.debug(f"Added trade #{standardized_trade['ticket']} to attribution analysis")

        except Exception as e:
            logger.error(f"Error adding trade to attribution analysis: {e}")

    def analyze_performance_attribution(self, start_date: Optional[str] = None,
                                      end_date: Optional[str] = None) -> AttributionResult:
        """Perform comprehensive performance attribution analysis."""
        try:
            # Filter trades by date range
            filtered_trades = self._filter_trades_by_date(start_date, end_date)

            if not filtered_trades:
                logger.warning("No trades found for the specified date range")
                return self._create_empty_attribution_result()

            # Calculate basic performance metrics
            basic_metrics = self._calculate_basic_performance_metrics(filtered_trades)

            # Strategy attribution analysis
            strategy_contributions = self._analyze_strategy_contributions(filtered_trades)

            # Market condition performance
            market_condition_performance = self._analyze_market_condition_performance(filtered_trades)

            # Parameter impact analysis
            parameter_impact = self._analyze_parameter_impact(filtered_trades)

            # Time-based returns
            time_based_returns = self._analyze_time_based_returns(filtered_trades)

            # Risk-adjusted contributions
            risk_adjusted_contributions = self._calculate_risk_adjusted_contributions(filtered_trades)

            # Create comprehensive result
            result = AttributionResult(
                total_return=basic_metrics['total_return'],
                annualized_return=basic_metrics['annualized_return'],
                volatility=basic_metrics['volatility'],
                sharpe_ratio=basic_metrics['sharpe_ratio'],
                max_drawdown=basic_metrics['max_drawdown'],
                win_rate=basic_metrics['win_rate'],
                profit_factor=basic_metrics['profit_factor'],
                avg_win=basic_metrics['avg_win'],
                avg_loss=basic_metrics['avg_loss'],
                largest_win=basic_metrics['largest_win'],
                largest_loss=basic_metrics['largest_loss'],
                total_trades=basic_metrics['total_trades'],
                winning_trades=basic_metrics['winning_trades'],
                losing_trades=basic_metrics['losing_trades'],
                strategy_contributions=strategy_contributions,
                market_condition_performance=market_condition_performance,
                parameter_impact=parameter_impact,
                time_based_returns=time_based_returns,
                risk_adjusted_contributions=risk_adjusted_contributions
            )

            return result

        except Exception as e:
            logger.error(f"Error in performance attribution analysis: {e}")
            return self._create_empty_attribution_result()

    def _filter_trades_by_date(self, start_date: Optional[str],
                              end_date: Optional[str]) -> List[Dict[str, Any]]:
        """Filter trades by date range."""
        if not start_date and not end_date:
            return self.trade_history

        filtered_trades = []
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None

        for trade in self.trade_history:
            trade_dt = datetime.fromisoformat(trade['timestamp'])

            if start_dt and trade_dt < start_dt:
                continue
            if end_dt and trade_dt > end_dt:
                continue

            filtered_trades.append(trade)

        return filtered_trades

    def _calculate_basic_performance_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate basic performance metrics."""
        if not trades:
            return {
                'total_return': 0.0, 'annualized_return': 0.0, 'volatility': 0.0,
                'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0,
                'profit_factor': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
                'largest_win': 0.0, 'largest_loss': 0.0, 'total_trades': 0,
                'winning_trades': 0, 'losing_trades': 0
            }

        # Extract profits
        profits = [trade['profit'] for trade in trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]

        # Basic metrics
        total_return = sum(profits)
        total_trades = len(trades)
        winning_count = len(winning_trades)
        losing_count = len(losing_trades)
        win_rate = winning_count / total_trades if total_trades > 0 else 0

        # Average win/loss
        avg_win = sum(winning_trades) / winning_count if winning_count > 0 else 0
        avg_loss = sum(losing_trades) / losing_count if losing_count > 0 else 0

        # Largest win/loss
        largest_win = max(winning_trades) if winning_trades else 0
        largest_loss = min(losing_trades) if losing_trades else 0

        # Profit factor
        total_wins = sum(winning_trades)
        total_losses = abs(sum(losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Calculate returns over time for volatility and drawdown
        cumulative_returns = np.cumsum(profits)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - peak
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0

        # Volatility (annualized)
        if len(profits) > 1:
            volatility = np.std(profits) * np.sqrt(252)  # Assuming daily returns
        else:
            volatility = 0

        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        if volatility > 0:
            sharpe_ratio = (total_return / total_trades - risk_free_rate) / volatility
        else:
            sharpe_ratio = 0

        # Annualized return (simplified)
        if total_trades > 0:
            annualized_return = total_return / total_trades * 252
        else:
            annualized_return = 0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'total_trades': total_trades,
            'winning_trades': winning_count,
            'losing_trades': losing_count
        }

    def _analyze_strategy_contributions(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze contribution by strategy/exit reason."""
        strategy_returns = defaultdict(float)
        strategy_counts = defaultdict(int)

        for trade in trades:
            exit_reason = trade.get('exit_reason', 'unknown')
            profit = trade['profit']

            strategy_returns[exit_reason] += profit
            strategy_counts[exit_reason] += 1

        # Calculate percentage contributions
        total_return = sum(strategy_returns.values())
        if total_return == 0:
            return {strategy: 0.0 for strategy in strategy_returns.keys()}

        contributions = {}
        for strategy, return_val in strategy_returns.items():
            contributions[strategy] = (return_val / total_return) * 100

        return dict(contributions)

    def _analyze_market_condition_performance(self, trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by market conditions."""
        conditions = ['volatility_level', 'session']
        results = {}

        for condition in conditions:
            condition_performance = defaultdict(list)

            for trade in trades:
                cond_value = trade.get(condition, 'unknown')
                profit = trade['profit']
                condition_performance[cond_value].append(profit)

            # Calculate metrics for each condition value
            condition_results = {}
            for cond_value, profits in condition_performance.items():
                if profits:
                    total_return = sum(profits)
                    win_rate = len([p for p in profits if p > 0]) / len(profits)
                    avg_profit = total_return / len(profits)
                    condition_results[cond_value] = {
                        'total_return': total_return,
                        'win_rate': win_rate,
                        'avg_profit': avg_profit,
                        'trade_count': len(profits)
                    }

            results[condition] = dict(condition_results)

        return results

    def _analyze_parameter_impact(self, trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze impact of different parameters on performance."""
        parameter_ranges = {
            'confidence': [(0, 0.7), (0.7, 0.85), (0.85, 1.0)],
            'hold_duration': [(0, 1), (1, 4), (4, 24), (24, float('inf'))],  # hours
            'spread_pips': [(0, 5), (5, 15), (15, float('inf'))]
        }

        results = {}

        for param, ranges in parameter_ranges.items():
            param_performance = {}

            for min_val, max_val in ranges:
                range_trades = []
                for trade in trades:
                    value = trade.get(param, 0)
                    if min_val <= value < max_val:
                        range_trades.append(trade['profit'])

                if range_trades:
                    total_return = sum(range_trades)
                    win_rate = len([p for p in range_trades if p > 0]) / len(range_trades)
                    avg_profit = total_return / len(range_trades)

                    range_key = f"{min_val}-{max_val if max_val != float('inf') else 'inf'}"
                    param_performance[range_key] = {
                        'total_return': total_return,
                        'win_rate': win_rate,
                        'avg_profit': avg_profit,
                        'trade_count': len(range_trades)
                    }

            results[param] = param_performance

        return results

    def _analyze_time_based_returns(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze returns by time periods."""
        time_periods = {
            'hourly': '%Y-%m-%d %H',
            'daily': '%Y-%m-%d',
            'weekly': '%Y-%U',
            'monthly': '%Y-%m'
        }

        results = {}

        for period_name, period_format in time_periods.items():
            period_returns = defaultdict(float)
            period_counts = defaultdict(int)

            for trade in trades:
                timestamp = datetime.fromisoformat(trade['timestamp'])
                period_key = timestamp.strftime(period_format)
                period_returns[period_key] += trade['profit']
                period_counts[period_key] += 1

            # Calculate average return per period
            if period_returns:
                avg_return = sum(period_returns.values()) / len(period_returns)
                results[period_name] = avg_return
            else:
                results[period_name] = 0.0

        return results

    def _calculate_risk_adjusted_contributions(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate risk-adjusted contributions by strategy."""
        strategy_stats = defaultdict(lambda: {'returns': [], 'volatility': 0, 'sharpe': 0})

        for trade in trades:
            exit_reason = trade.get('exit_reason', 'unknown')
            strategy_stats[exit_reason]['returns'].append(trade['profit'])

        # Calculate risk-adjusted metrics
        risk_adjusted = {}
        for strategy, stats in strategy_stats.items():
            returns = stats['returns']
            if len(returns) > 1:
                volatility = np.std(returns)
                avg_return = np.mean(returns)
                sharpe = avg_return / volatility if volatility > 0 else 0
                risk_adjusted[strategy] = sharpe
            else:
                risk_adjusted[strategy] = 0.0

        return dict(risk_adjusted)

    def _create_empty_attribution_result(self) -> AttributionResult:
        """Create an empty attribution result."""
        return AttributionResult(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            strategy_contributions={},
            market_condition_performance={},
            parameter_impact={},
            time_based_returns={},
            risk_adjusted_contributions={}
        )

    def generate_attribution_report(self, start_date: Optional[str] = None,
                                  end_date: Optional[str] = None) -> Dict[str, Any]:
        """Generate a comprehensive attribution report."""
        try:
            result = self.analyze_performance_attribution(start_date, end_date)

            report = {
                'summary': {
                    'total_return': result.total_return,
                    'annualized_return': result.annualized_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'win_rate': result.win_rate,
                    'profit_factor': result.profit_factor,
                    'total_trades': result.total_trades
                },
                'trade_analysis': {
                    'winning_trades': result.winning_trades,
                    'losing_trades': result.losing_trades,
                    'avg_win': result.avg_win,
                    'avg_loss': result.avg_loss,
                    'largest_win': result.largest_win,
                    'largest_loss': result.largest_loss
                },
                'strategy_attribution': result.strategy_contributions,
                'market_condition_performance': result.market_condition_performance,
                'parameter_impact': result.parameter_impact,
                'time_based_returns': result.time_based_returns,
                'risk_adjusted_contributions': result.risk_adjusted_contributions,
                'generated_at': datetime.now().isoformat(),
                'date_range': {
                    'start': start_date,
                    'end': end_date
                }
            }

            return report

        except Exception as e:
            logger.error(f"Error generating attribution report: {e}")
            return {'error': str(e)}

    def get_top_performing_strategies(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top performing strategies by return."""
        if not self.trade_history:
            return []

        strategy_returns = defaultdict(float)

        for trade in self.trade_history:
            exit_reason = trade.get('exit_reason', 'unknown')
            strategy_returns[exit_reason] += trade['profit']

        sorted_strategies = sorted(strategy_returns.items(), key=lambda x: x[1], reverse=True)
        return sorted_strategies[:n]

    def get_worst_performing_strategies(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get worst performing strategies by return."""
        if not self.trade_history:
            return []

        strategy_returns = defaultdict(float)

        for trade in self.trade_history:
            exit_reason = trade.get('exit_reason', 'unknown')
            strategy_returns[exit_reason] += trade['profit']

        sorted_strategies = sorted(strategy_returns.items(), key=lambda x: x[1])
        return sorted_strategies[:n]

    def analyze_seasonal_performance(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance by season/month."""
        seasonal_performance = defaultdict(lambda: {'returns': [], 'trades': 0})

        for trade in self.trade_history:
            timestamp = datetime.fromisoformat(trade['timestamp'])
            month = timestamp.strftime('%B')
            seasonal_performance[month]['returns'].append(trade['profit'])
            seasonal_performance[month]['trades'] += 1

        results = {}
        for month, data in seasonal_performance.items():
            returns = data['returns']
            if returns:
                total_return = sum(returns)
                avg_return = total_return / len(returns)
                win_rate = len([r for r in returns if r > 0]) / len(returns)
                results[month] = {
                    'total_return': total_return,
                    'avg_return': avg_return,
                    'win_rate': win_rate,
                    'trade_count': data['trades']
                }

        return dict(results)

    def calculate_attribution_confidence(self) -> Dict[str, float]:
        """Calculate confidence intervals for attribution results."""
        if len(self.trade_history) < 30:
            return {'confidence_level': 'low', 'minimum_trades_needed': 30}

        # Bootstrap analysis for confidence intervals
        n_bootstrap = 1000
        strategy_returns = defaultdict(list)

        for _ in range(n_bootstrap):
            sample = np.random.choice(self.trade_history, size=len(self.trade_history), replace=True)
            sample_strategy_returns = defaultdict(float)

            for trade in sample:
                exit_reason = trade.get('exit_reason', 'unknown')
                sample_strategy_returns[exit_reason] += trade['profit']

            for strategy, return_val in sample_strategy_returns.items():
                strategy_returns[strategy].append(return_val)

        # Calculate confidence intervals
        confidence_intervals = {}
        for strategy, returns in strategy_returns.items():
            if returns:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                ci_lower = mean_return - 1.96 * std_return
                ci_upper = mean_return + 1.96 * std_return

                confidence_intervals[strategy] = {
                    'mean': mean_return,
                    'std': std_return,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'confidence_level': 'high' if len(returns) > 100 else 'medium'
                }

        return confidence_intervals
