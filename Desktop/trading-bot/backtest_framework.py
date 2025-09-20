#!/usr/bin/env python3
"""
Simple Backtesting Framework for Trading Strategies.

This framework allows testing trading strategies on historical data
before deploying them live.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import logging
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

class BacktestResult:
    """Container for backtest results."""

    def __init__(self):
        self.trades = []
        self.equity_curve = []
        self.drawdowns = []
        self.metrics = {}

    def add_trade(self, trade: Dict):
        """Add a trade to the results."""
        self.trades.append(trade)

    def calculate_metrics(self):
        """Calculate performance metrics."""
        if not self.trades:
            return

        df = pd.DataFrame(self.trades)

        # Basic metrics
        total_trades = len(df)
        winning_trades = len(df[df['profit'] > 0])
        losing_trades = len(df[df['profit'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Profit metrics
        total_profit = df['profit'].sum()
        avg_profit = df['profit'].mean()
        max_profit = df['profit'].max()
        max_loss = df['profit'].min()

        # Risk metrics
        profit_factor = abs(df[df['profit'] > 0]['profit'].sum() / df[df['profit'] < 0]['profit'].sum()) if len(df[df['profit'] < 0]) > 0 else float('inf')

        # Sharpe ratio (simplified)
        if len(df) > 1:
            returns = df['profit'].pct_change().fillna(0)
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        # Maximum drawdown
        cumulative = (1 + df['profit'].cumsum()).fillna(1)
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        self.metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_balance': 10000 + total_profit  # Assuming $10k starting balance
        }

class SimpleBacktester:
    """Simple backtesting framework for trading strategies."""

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.position_size = 0.1  # 0.1 lots
        self.spread_pips = 2  # 2 pip spread
        self.commission_per_lot = 5.0  # $5 per lot round trip

    def load_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Load historical data for backtesting."""
        try:
            # This would normally load from a data source
            # For now, we'll generate synthetic data
            dates = pd.date_range(end=datetime.now(), periods=days*24, freq='h')
            np.random.seed(42)  # For reproducible results

            # Generate realistic price movements
            base_price = 1.0850 if 'EUR' in symbol else 150.00
            price_changes = np.random.normal(0, 0.001, len(dates))
            prices = base_price * (1 + np.cumsum(price_changes))

            df = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': prices * (1 + np.random.uniform(0, 0.002, len(dates))),
                'low': prices * (1 - np.random.uniform(0, 0.002, len(dates))),
                'close': prices,
                'volume': np.random.randint(1000, 10000, len(dates))
            })

            return df

        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return pd.DataFrame()

    def run_backtest(self, strategy_func: Callable, symbol: str = "EURUSD",
                    days: int = 30) -> BacktestResult:
        """Run backtest with given strategy function."""
        logger.info(f"🧪 Starting backtest for {symbol} over {days} days")

        # Load historical data
        data = self.load_historical_data(symbol, days)
        if data.empty:
            logger.error("No historical data available")
            return BacktestResult()

        result = BacktestResult()
        self.current_balance = self.initial_balance

        # Reset position tracking
        open_position = None

        # Iterate through each candle
        for idx, row in data.iterrows():
            current_price = row['close']
            timestamp = row['timestamp']

            # Get strategy signal
            signal = strategy_func(row, open_position)

            if signal['action'] == 'BUY' and open_position is None:
                # Open long position
                entry_price = current_price + (self.spread_pips / 100000)  # Add spread
                open_position = {
                    'type': 'BUY',
                    'entry_price': entry_price,
                    'entry_time': timestamp,
                    'size': self.position_size
                }

            elif signal['action'] == 'SELL' and open_position is None:
                # Open short position
                entry_price = current_price - (self.spread_pips / 100000)  # Add spread
                open_position = {
                    'type': 'SELL',
                    'entry_price': entry_price,
                    'entry_time': timestamp,
                    'size': self.position_size
                }

            elif signal['action'] == 'CLOSE' and open_position is not None:
                # Close position
                exit_price = current_price
                if open_position['type'] == 'BUY':
                    exit_price -= (self.spread_pips / 100000)  # Subtract spread for selling
                else:
                    exit_price += (self.spread_pips / 100000)  # Add spread for buying back

                # Calculate profit/loss
                if open_position['type'] == 'BUY':
                    profit = (exit_price - open_position['entry_price']) * (100000 / self.position_size)  # Pips to dollars
                else:
                    profit = (open_position['entry_price'] - exit_price) * (100000 / self.position_size)

                # Subtract commission
                profit -= self.commission_per_lot

                # Record trade
                trade = {
                    'symbol': symbol,
                    'type': open_position['type'],
                    'entry_price': open_position['entry_price'],
                    'exit_price': exit_price,
                    'entry_time': open_position['entry_time'],
                    'exit_time': timestamp,
                    'profit': profit,
                    'size': open_position['size']
                }

                result.add_trade(trade)
                self.current_balance += profit

                # Reset position
                open_position = None

        # Calculate final metrics
        result.calculate_metrics()

        logger.info(f"✅ Backtest completed: {result.metrics.get('total_trades', 0)} trades")
        return result

def simple_momentum_strategy(candle: pd.Series, open_position: Optional[Dict]) -> Dict:
    """Simple momentum-based strategy for testing."""
    # Calculate simple moving averages
    # In a real implementation, you'd have access to historical data
    current_price = candle['close']

    # Simple trend-following logic
    if open_position is None:
        # Look for breakout signals
        if current_price > candle['open'] * 1.001:  # Price up 0.1%
            return {'action': 'BUY', 'reason': 'uptrend'}
        elif current_price < candle['open'] * 0.999:  # Price down 0.1%
            return {'action': 'SELL', 'reason': 'downtrend'}
    else:
        # Exit logic
        if open_position['type'] == 'BUY':
            # Exit if price drops 0.2% from entry
            if current_price < open_position['entry_price'] * 0.998:
                return {'action': 'CLOSE', 'reason': 'stop_loss'}
            # Take profit at 0.5%
            elif current_price > open_position['entry_price'] * 1.005:
                return {'action': 'CLOSE', 'reason': 'take_profit'}
        else:  # SELL position
            if current_price > open_position['entry_price'] * 1.002:
                return {'action': 'CLOSE', 'reason': 'stop_loss'}
            elif current_price < open_position['entry_price'] * 0.995:
                return {'action': 'CLOSE', 'reason': 'take_profit'}

    return {'action': 'HOLD', 'reason': 'no_signal'}

def run_sample_backtest():
    """Run a sample backtest to demonstrate the framework."""
    print("🚀 RUNNING SAMPLE BACKTEST")
    print("=" * 50)

    # Initialize backtester
    backtester = SimpleBacktester(initial_balance=10000.0)

    # Run backtest
    result = backtester.run_backtest(
        strategy_func=simple_momentum_strategy,
        symbol="EURUSD",
        days=30
    )

    # Display results
    print("\n📊 BACKTEST RESULTS:")
    print("=" * 30)

    metrics = result.metrics

    if not metrics:
        print("❌ No trades executed during backtest")
        return

    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.1%}")
    print(f"Total Profit: ${metrics['total_profit']:.2f}")
    print(f"Average Profit: ${metrics['avg_profit']:.2f}")
    print(f"Max Profit: ${metrics['max_profit']:.2f}")
    print(f"Max Loss: ${metrics['max_loss']:.2f}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.1%}")
    print(f"Final Balance: ${metrics['final_balance']:.2f}")

    # Performance assessment
    print("\n🎯 PERFORMANCE ASSESSMENT:")
    if metrics['win_rate'] > 0.6 and metrics['profit_factor'] > 1.5:
        print("✅ STRATEGY PERFORMANCE: EXCELLENT")
    elif metrics['win_rate'] > 0.5 and metrics['profit_factor'] > 1.2:
        print("✅ STRATEGY PERFORMANCE: GOOD")
    elif metrics['win_rate'] > 0.4 and metrics['profit_factor'] > 1.0:
        print("⚠️ STRATEGY PERFORMANCE: MODERATE")
    else:
        print("❌ STRATEGY PERFORMANCE: POOR")

    print("\n💡 RECOMMENDATIONS:")
    if metrics['max_drawdown'] > 0.2:
        print("• Consider reducing position sizes to limit drawdowns")
    if metrics['win_rate'] < 0.5:
        print("• Review entry/exit criteria for better win rate")
    if metrics['profit_factor'] < 1.2:
        print("• Focus on improving reward-to-risk ratios")

if __name__ == "__main__":
    run_sample_backtest()
