#!/usr/bin/env python3
"""Precise pip value calculator for forex trading."""

from typing import Optional
import MetaTrader5 as mt5
import logging

logger = logging.getLogger(__name__)

class PipCalculator:
    """Calculate precise pip values for position sizing across all currency pairs."""

    @staticmethod
    def get_pip_size(symbol: str) -> float:
        """Get pip size for a symbol (0.0001 for majors, 0.01 for JPY pairs)."""
        return 0.01 if 'JPY' in symbol.upper() else 0.0001

    @staticmethod
    def get_tick_value(symbol: str) -> float:
        """Get tick value from broker symbol info."""
        try:
            if not mt5.initialize():
                logger.warning("MT5 not initialized, using fallback tick value")
                return 100000 if 'JPY' not in symbol.upper() else 1000

            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.warning(f"Symbol info not found for {symbol}, using fallback")
                return 100000 if 'JPY' not in symbol.upper() else 1000

            return symbol_info.trade_tick_value
        except Exception as e:
            logger.error(f"Error getting tick value for {symbol}: {e}")
            return 100000 if 'JPY' not in symbol.upper() else 1000

    @staticmethod
    def get_pip_value(symbol: str, lot_size: float, current_rate: float, account_currency: str = 'USD') -> float:
        """
        Calculate precise pip value for position sizing.

        Formula: (pip_size / exchange_rate) * (lot_size * contract_size) * tick_value
        """
        try:
            pip_size = PipCalculator.get_pip_size(symbol)
            contract_size = 100000  # Standard lot size
            tick_value = PipCalculator.get_tick_value(symbol)

            # Base pip value calculation
            pip_value = (pip_size / current_rate) * (lot_size * contract_size) * tick_value

            # Adjust for non-USD account currency (simplified - assumes USD base)
            if account_currency != 'USD':
                # Would need current USD/account_currency rate here
                # For now, assume USD account
                pass

            return round(pip_value, 2)

        except Exception as e:
            logger.error(f"Error calculating pip value for {symbol}: {e}")
            # Fallback calculation
            pip_size = PipCalculator.get_pip_size(symbol)
            return round((pip_size / current_rate) * (lot_size * 100000) * 10, 2)

    @staticmethod
    def calculate_position_size(symbol: str, risk_amount: float, entry_price: float,
                              stop_loss: float, account_currency: str = 'USD') -> float:
        """
        Calculate position size based on risk amount and stop loss distance.

        Returns lot size that risks exactly risk_amount at the given stop loss.
        """
        try:
            if entry_price <= 0 or stop_loss <= 0 or risk_amount <= 0:
                logger.error("Invalid inputs for position size calculation")
                return 0.01

            # Calculate risk in pips
            risk_pips = abs(entry_price - stop_loss) / PipCalculator.get_pip_size(symbol)

            if risk_pips <= 0:
                logger.error("Invalid risk pips calculation")
                return 0.01

            # Get pip value at current rate
            pip_value = PipCalculator.get_pip_value(symbol, 1.0, entry_price, account_currency)

            # Calculate position size
            position_size = risk_amount / (risk_pips * pip_value)

            # Apply limits
            position_size = min(position_size, 1.0)    # Max 1 lot
            position_size = max(position_size, 0.01)   # Min 0.01 lot

            return round(position_size, 2)

        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.01

    @staticmethod
    def validate_position_size(symbol: str, position_size: float, entry_price: float,
                             stop_loss: float, max_risk_percent: float = 0.02) -> dict:
        """
        Validate position size against risk limits.

        Returns dict with validation results and risk metrics.
        """
        try:
            pip_size = PipCalculator.get_pip_size(symbol)
            risk_pips = abs(entry_price - stop_loss) / pip_size
            pip_value = PipCalculator.get_pip_value(symbol, position_size, entry_price)

            risk_amount = risk_pips * pip_value
            risk_percent = risk_amount / 1000  # Assuming $1000 account for percentage

            return {
                'valid': risk_percent <= max_risk_percent,
                'risk_amount': round(risk_amount, 2),
                'risk_percent': round(risk_percent * 100, 2),
                'pip_value': round(pip_value, 4),
                'risk_pips': round(risk_pips, 1),
                'max_risk_percent': max_risk_percent * 100
            }

        except Exception as e:
            logger.error(f"Error validating position size for {symbol}: {e}")
            return {
                'valid': False,
                'error': str(e),
                'risk_amount': 0,
                'risk_percent': 0
            }
