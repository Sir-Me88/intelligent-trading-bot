#!/usr/bin/env python3
"""Backtesting broker interface for simulating trades."""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone
import pandas as pd

logger = logging.getLogger(__name__)

class BacktestPosition:
    """Represents a position in backtesting."""

    def __init__(self, ticket: int, symbol: str, order_type: str, volume: float,
                 entry_price: float, sl: float = None, tp: float = None,
                 entry_time: datetime = None):
        self.ticket = ticket
        self.symbol = symbol
        self.type = 0 if order_type.upper() == 'BUY' else 1  # 0=BUY, 1=SELL
        self.volume = volume
        self.price_open = entry_price
        self.price_current = entry_price
        self.sl = sl
        self.tp = tp
        self.entry_time = entry_time or datetime.now(timezone.utc)
        self.profit = 0.0

    def update_profit(self, current_price: float):
        """Update position profit based on current price."""
        if self.type == 0:  # BUY
            self.profit = (current_price - self.price_open) * self.volume * 100000  # Assuming 5-digit broker
        else:  # SELL
            self.profit = (self.price_open - current_price) * self.volume * 100000

        self.price_current = current_price

    def should_close(self, current_price: float) -> Optional[str]:
        """Check if position should be closed based on SL/TP."""
        if self.type == 0:  # BUY
            if self.sl and current_price <= self.sl:
                return 'SL'
            if self.tp and current_price >= self.tp:
                return 'TP'
        else:  # SELL
            if self.sl and current_price >= self.sl:
                return 'SL'
            if self.tp and current_price <= self.tp:
                return 'TP'
        return None

class BacktestBrokerManager:
    """Backtesting broker manager that simulates trading operations."""

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.margin = 0.0
        self.margin_free = initial_balance

        self.positions: Dict[int, BacktestPosition] = {}
        self.next_ticket = 1000
        self.trades_history = []

        # Backtest statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance

        logger.info(f"🧪 Backtest broker initialized with ${initial_balance:.2f} balance")

    async def initialize(self) -> bool:
        """Initialize backtest broker."""
        logger.info("✅ Backtest broker ready")
        return True

    async def get_account_info(self) -> Dict:
        """Get simulated account information."""
        return {
            'balance': self.balance,
            'equity': self.equity,
            'margin': self.margin,
            'margin_free': self.margin_free
        }

    async def get_positions(self) -> List[Dict]:
        """Get open positions."""
        return [{
            'ticket': pos.ticket,
            'symbol': pos.symbol,
            'type': pos.type,
            'volume': pos.volume,
            'price_open': pos.price_open,
            'price_current': pos.price_current,
            'profit': pos.profit,
            'sl': pos.sl,
            'tp': pos.tp
        } for pos in self.positions.values()]

    async def place_order(self, symbol: str, order_type: str, volume: float,
                         sl: float = None, tp: float = None) -> Dict:
        """Simulate placing an order."""
        try:
            # Get current price (this would be provided by the backtest engine)
            current_price = self._get_current_price(symbol)
            if current_price is None:
                return {
                    'status': 'FAILED',
                    'ticket': None,
                    'retcode': -1,
                    'comment': f'No price data for {symbol}'
                }

            # Create position
            ticket = self.next_ticket
            self.next_ticket += 1

            position = BacktestPosition(
                ticket=ticket,
                symbol=symbol,
                order_type=order_type,
                volume=volume,
                entry_price=current_price,
                sl=sl,
                tp=tp
            )

            self.positions[ticket] = position
            self.total_trades += 1

            # Log trade
            trade_record = {
                'ticket': ticket,
                'symbol': symbol,
                'type': order_type,
                'volume': volume,
                'entry_price': current_price,
                'sl': sl,
                'tp': tp,
                'entry_time': position.entry_time.isoformat()
            }
            self.trades_history.append(trade_record)

            logger.info(f"🧪 SIMULATED TRADE: {order_type} {volume} {symbol} at {current_price:.5f} (Ticket: {ticket})")

            return {
                'status': 'SUCCESS',
                'ticket': ticket,
                'retcode': 0,
                'comment': 'Order placed successfully'
            }

        except Exception as e:
            logger.error(f"Error placing backtest order: {e}")
            return {
                'status': 'FAILED',
                'ticket': None,
                'retcode': -1,
                'comment': str(e)
            }

    async def close_position(self, ticket: int) -> Dict:
        """Simulate closing a position."""
        try:
            if ticket not in self.positions:
                return {
                    'status': 'FAILED',
                    'ticket': ticket,
                    'retcode': -1,
                    'comment': f'Position {ticket} not found'
                }

            position = self.positions[ticket]
            exit_price = position.price_current
            profit = position.profit

            # Update account balance
            self.balance += profit
            self.equity = self.balance

            # Update statistics
            if profit > 0:
                self.winning_trades += 1
                self.total_profit += profit
            else:
                self.losing_trades += 1
                self.total_loss += abs(profit)

            # Update drawdown
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance
            else:
                current_drawdown = (self.peak_balance - self.balance) / self.peak_balance
                self.max_drawdown = max(self.max_drawdown, current_drawdown)

            # Log exit
            exit_record = {
                'ticket': ticket,
                'symbol': position.symbol,
                'exit_price': exit_price,
                'profit': profit,
                'exit_time': datetime.now(timezone.utc).isoformat()
            }
            self.trades_history.append(exit_record)

            logger.info(f"🧪 CLOSED POSITION: {position.symbol} #{ticket} at {exit_price:.5f}, P/L: ${profit:.2f}")

            # Remove position
            del self.positions[ticket]

            return {
                'status': 'SUCCESS',
                'ticket': ticket,
                'retcode': 0,
                'comment': 'Position closed successfully'
            }

        except Exception as e:
            logger.error(f"Error closing backtest position: {e}")
            return {
                'status': 'FAILED',
                'ticket': ticket,
                'retcode': -1,
                'comment': str(e)
            }

    def update_positions(self, price_data: Dict[str, float]):
        """Update all positions with current prices."""
        for ticket, position in list(self.positions.items()):
            current_price = price_data.get(position.symbol)
            if current_price is not None:
                position.update_profit(current_price)

                # Check for SL/TP hits
                close_reason = position.should_close(current_price)
                if close_reason:
                    logger.info(f"🧪 {close_reason} HIT: {position.symbol} #{ticket}")
                    asyncio.create_task(self.close_position(ticket))

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol (placeholder - would be provided by backtest engine)."""
        # This would be overridden by the backtest engine
        return 1.0  # Default price

    def get_backtest_stats(self) -> Dict:
        """Get backtesting statistics."""
        win_rate = (self.winning_trades / self.total_trades) if self.total_trades > 0 else 0
        avg_win = (self.total_profit / self.winning_trades) if self.winning_trades > 0 else 0
        avg_loss = (self.total_loss / self.losing_trades) if self.losing_trades > 0 else 0
        profit_factor = (self.total_profit / self.total_loss) if self.total_loss > 0 else float('inf')

        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'net_profit': self.total_profit - self.total_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': self.max_drawdown,
            'final_balance': self.balance,
            'return_pct': ((self.balance - self.initial_balance) / self.initial_balance) * 100
        }

    async def get_spread_pips(self, symbol: str) -> Optional[Dict]:
        """Get simulated spread (always 0 for backtesting)."""
        return {
            'spread_pips': 0,
            'pip_value': 10
        }

    def is_connected(self) -> bool:
        """Check if backtest broker is connected."""
        return True
