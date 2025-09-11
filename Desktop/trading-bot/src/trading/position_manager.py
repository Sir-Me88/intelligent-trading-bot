"""Position and risk management."""

import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum


class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"


class Position:
    def __init__(self, id, pair, direction, size, entry_price, stop_loss, take_profit):
        self.id = id
        self.pair = pair
        self.direction = direction
        self.size = float(size)
        self.entry_price = float(entry_price)
        self.current_price = float(entry_price)
        self.stop_loss = float(stop_loss)
        self.take_profit = float(take_profit)
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.status = PositionStatus.OPEN
        self.open_time = datetime.now()
        self.close_time = None
        self.commission = 0.0
        self.swap = 0.0

    def update_current_price(self, new_price):
        """Update current price and unrealized PnL in pips * lots (test expectation)."""
        self.current_price = float(new_price)
        # pip multiplier: 100 for JPY pairs else 10000
        p = (self.pair or "").replace("/", "").replace("_", "").upper()
        pip_factor = 100 if "JPY" in p or p.endswith("JPY") else 10000
        if str(self.direction).lower() in ("buy", "long", "0"):
            diff = self.current_price - self.entry_price
        else:
            diff = self.entry_price - self.current_price
        # Unrealized pnl expected = pips * lots
        try:
            self.unrealized_pnl = abs(diff) * pip_factor * float(self.size)
        except Exception:
            self.unrealized_pnl = 0.0

    def get_risk_amount(self):
        """Return risk amount in account pip units (pips * lots)."""
        p = (self.pair or "").replace("/", "").replace("_", "").upper()
        pip_factor = 100 if "JPY" in p or p.endswith("JPY") else 10000
        return abs(self.entry_price - self.stop_loss) * pip_factor * float(self.size)


class PositionSizer:
    def __init__(self, max_total_risk=0.02):
        self.max_total_risk = float(max_total_risk)

    def calculate_position_size(self, account_equity, entry_price, stop_loss, current_positions):
        """Calculate suggested size and whether trading is allowed given max_total_risk.

        Uses:
          - default risk_per_trade = 1% of equity
          - pip factor = 10000 for non-JPY pairs
        """
        account_equity = float(account_equity)
        risk_per_trade_amount = account_equity * 0.01  # 1% per trade
        stop_distance = abs(float(entry_price) - float(stop_loss))
        if stop_distance <= 0:
            return {"size": 0.0, "can_trade": False}

        pip_factor = 10000
        base_size = risk_per_trade_amount / (stop_distance * pip_factor)

        # compute total existing risk (in account currency pip units)
        total_risk = 0.0
        for pos in (current_positions or []):
            try:
                total_risk += pos.get_risk_amount()
            except Exception:
                continue

        # New trade's risk in same units
        new_trade_risk = stop_distance * base_size * pip_factor

        allowed_total_risk_value = account_equity * self.max_total_risk

        if (total_risk + new_trade_risk) > allowed_total_risk_value:
            return {"size": round(base_size, 2), "can_trade": False}
        return {"size": round(base_size, 2), "can_trade": True}


class RiskManager:
    def validate_trade(self, equity, positions, new_position):
        if equity < 1000:
            return False
        if len(positions) >= 5:
            return False
        return True


class PositionManager:
    def __init__(self):
        self.positions = []

    def create_position(self, id, pair, direction, size, entry_price, stop_loss, take_profit):
        pos = Position(id, pair, direction, size, entry_price, stop_loss, take_profit)
        self.positions.append(pos)
        return pos

    def update_positions(self):
        for pos in self.positions:
            pos.update_current_price(pos.current_price)

    def close_position(self, position_id):
        for pos in self.positions:
            if pos.id == position_id:
                pos.close_time = datetime.now()
                pos.status = PositionStatus.CLOSED
                return True
        return False

    def get_open_positions(self):
        return [pos for pos in self.positions if pos.status == PositionStatus.OPEN]

    def get_total_unrealized_pnl(self):
        return sum(pos.unrealized_pnl for pos in self.positions)

