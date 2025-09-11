"""Enhanced Risk Management System for Production Trading."""

import MetaTrader5 as mt5
from typing import Dict, List, Optional, Tuple
import logging
import asyncio
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import json

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk assessment levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class NewsImpact(Enum):
    """News impact levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class EnhancedRiskManager:
    """Enhanced risk management with news detection and volatility analysis."""
    
    def __init__(self, broker_interface):
        self.broker = broker_interface
        
        # Risk parameters
        self.max_daily_loss = -0.02  # -2% daily loss limit
        self.max_position_risk = 0.05  # 5% per trade
        self.max_spread_pips = 20  # Maximum allowed spread
        self.max_total_positions = 10  # Maximum open positions
        self.max_correlation_exposure = 0.15  # 15% max correlated exposure
        
        # Volatility thresholds
        self.high_volatility_threshold = 0.003  # 0.3%
        self.extreme_volatility_threshold = 0.005  # 0.5%
        
        # News API configuration (placeholder for future implementation)
        self.news_api_enabled = False
        self.high_impact_news_buffer = 30  # minutes before/after news
        
        # Daily tracking
        self.daily_start_balance = None
        self.daily_trades_count = 0
        self.max_daily_trades = 50
        
    async def validate_trade(self, symbol: str, volume: float, direction: str) -> Dict:
        """Comprehensive trade validation with all risk checks."""
        validation = {
            'approved': False,
            'risk_level': RiskLevel.LOW,
            'position_size': volume,
            'warnings': [],
            'reasons': []
        }
        
        try:
            # Get account information
            account_info = await self.broker.get_account_info()
            if not account_info:
                validation['reasons'].append('Failed to get account information')
                return validation
            
            # 1. Daily loss limit check
            daily_check = await self._check_daily_loss_limit(account_info)
            if not daily_check['passed']:
                validation['reasons'].append(daily_check['reason'])
                validation['risk_level'] = RiskLevel.CRITICAL
                return validation
            
            # 2. Spread validation
            spread_check = await self._check_spread(symbol)
            if not spread_check['passed']:
                validation['reasons'].append(spread_check['reason'])
                validation['risk_level'] = RiskLevel.HIGH
                return validation
            
            # 3. Position risk calculation
            position_risk_check = await self._check_position_risk(symbol, volume, account_info)
            if not position_risk_check['passed']:
                validation['reasons'].append(position_risk_check['reason'])
                return validation
            
            # 4. Maximum positions check
            positions_check = await self._check_max_positions()
            if not positions_check['passed']:
                validation['reasons'].append(positions_check['reason'])
                return validation
            
            # 5. Volatility assessment
            volatility_check = await self._check_volatility(symbol)
            if volatility_check['risk_level'] == RiskLevel.HIGH:
                validation['warnings'].append(f"High volatility detected for {symbol}")
                validation['risk_level'] = RiskLevel.HIGH
                # Reduce position size for high volatility
                validation['position_size'] = volume * 0.5
            elif volatility_check['risk_level'] == RiskLevel.CRITICAL:
                validation['reasons'].append(f"Extreme volatility for {symbol} - trading suspended")
                return validation
            
            # 6. News impact check (if enabled)
            if self.news_api_enabled:
                news_check = await self._check_news_impact(symbol)
                if news_check['impact'] == NewsImpact.HIGH:
                    validation['warnings'].append(f"High impact news expected for {symbol}")
                    validation['risk_level'] = RiskLevel.MEDIUM
                    # Reduce position size during news
                    validation['position_size'] = min(validation['position_size'], volume * 0.3)
            
            # 7. Daily trade limit check
            if self.daily_trades_count >= self.max_daily_trades:
                validation['reasons'].append(f'Daily trade limit reached ({self.max_daily_trades})')
                return validation
            
            # 8. Correlation exposure check
            correlation_check = await self._check_correlation_exposure(symbol, validation['position_size'])
            if not correlation_check['passed']:
                validation['warnings'].append(correlation_check['reason'])
                validation['position_size'] = correlation_check['adjusted_size']
            
            # All checks passed
            validation['approved'] = True
            validation['reasons'].append('All risk checks passed')
            
            # Final position size validation
            if validation['position_size'] < 0.01:  # Minimum lot size
                validation['approved'] = False
                validation['reasons'].append('Position size too small after risk adjustments')
                return validation
            
            logger.info(f"✅ Trade approved for {symbol}: Size={validation['position_size']:.2f}, Risk={validation['risk_level'].value}")
            
        except Exception as e:
            logger.error(f"Error in trade validation: {e}")
            validation['reasons'].append(f'Validation error: {e}')
            validation['risk_level'] = RiskLevel.CRITICAL
        
        return validation
    
    async def _check_daily_loss_limit(self, account_info: Dict) -> Dict:
        """Check if daily loss limit is exceeded."""
        try:
            current_equity = account_info['equity']
            
            # Initialize daily start balance if not set
            if self.daily_start_balance is None:
                self.daily_start_balance = current_equity
            
            # Calculate daily P&L percentage
            daily_pnl_pct = (current_equity - self.daily_start_balance) / self.daily_start_balance
            
            if daily_pnl_pct <= self.max_daily_loss:
                return {
                    'passed': False,
                    'reason': f'Daily loss limit exceeded: {daily_pnl_pct:.2%} <= {self.max_daily_loss:.2%}'
                }
            
            return {'passed': True, 'daily_pnl': daily_pnl_pct}
            
        except Exception as e:
            logger.error(f"Error checking daily loss limit: {e}")
            return {'passed': False, 'reason': f'Daily loss check error: {e}'}
    
    async def _check_spread(self, symbol: str) -> Dict:
        """Check if spread is within acceptable limits."""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {'passed': False, 'reason': f'Could not get symbol info for {symbol}'}
            
            spread_pips = symbol_info.spread
            if spread_pips > self.max_spread_pips:
                return {
                    'passed': False,
                    'reason': f'Spread too high: {spread_pips} pips > {self.max_spread_pips} pips'
                }
            
            return {'passed': True, 'spread': spread_pips}
            
        except Exception as e:
            logger.error(f"Error checking spread for {symbol}: {e}")
            return {'passed': False, 'reason': f'Spread check error: {e}'}
    
    async def _check_position_risk(self, symbol: str, volume: float, account_info: Dict) -> Dict:
        """Calculate and validate position risk."""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {'passed': False, 'reason': f'Could not get symbol info for {symbol}'}
            
            # Calculate position value
            tick_value = symbol_info.trade_tick_value
            position_value = volume * tick_value * 100  # Approximate position value
            
            # Calculate risk as percentage of account
            account_balance = account_info['balance']
            position_risk_pct = position_value / account_balance
            
            if position_risk_pct > self.max_position_risk:
                return {
                    'passed': False,
                    'reason': f'Position risk too high: {position_risk_pct:.2%} > {self.max_position_risk:.2%}'
                }
            
            return {'passed': True, 'risk_pct': position_risk_pct}
            
        except Exception as e:
            logger.error(f"Error calculating position risk: {e}")
            return {'passed': False, 'reason': f'Position risk calculation error: {e}'}
    
    async def _check_max_positions(self) -> Dict:
        """Check if maximum number of positions is exceeded."""
        try:
            positions = mt5.positions_get()
            if positions is None:
                positions = []
            
            open_positions = len(positions)
            if open_positions >= self.max_total_positions:
                return {
                    'passed': False,
                    'reason': f'Maximum positions reached: {open_positions}/{self.max_total_positions}'
                }
            
            return {'passed': True, 'open_positions': open_positions}
            
        except Exception as e:
            logger.error(f"Error checking max positions: {e}")
            return {'passed': False, 'reason': f'Max positions check error: {e}'}
    
    async def _check_volatility(self, symbol: str) -> Dict:
        """Assess market volatility for the symbol."""
        try:
            # Get recent price data to calculate volatility
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 20)
            if rates is None or len(rates) < 10:
                return {'risk_level': RiskLevel.MEDIUM, 'reason': 'Insufficient data for volatility check'}
            
            # Calculate price changes
            closes = [rate[4] for rate in rates]  # Close prices
            price_changes = []
            for i in range(1, len(closes)):
                change = abs(closes[i] - closes[i-1]) / closes[i-1]
                price_changes.append(change)
            
            # Calculate average volatility
            avg_volatility = sum(price_changes) / len(price_changes)
            
            if avg_volatility >= self.extreme_volatility_threshold:
                return {'risk_level': RiskLevel.CRITICAL, 'volatility': avg_volatility}
            elif avg_volatility >= self.high_volatility_threshold:
                return {'risk_level': RiskLevel.HIGH, 'volatility': avg_volatility}
            else:
                return {'risk_level': RiskLevel.LOW, 'volatility': avg_volatility}
                
        except Exception as e:
            logger.error(f"Error checking volatility for {symbol}: {e}")
            return {'risk_level': RiskLevel.MEDIUM, 'reason': f'Volatility check error: {e}'}
    
    async def _check_news_impact(self, symbol: str) -> Dict:
        """Check for upcoming high-impact news (placeholder for future implementation)."""
        # This is a placeholder for news API integration
        # In production, this would connect to economic calendar APIs
        return {'impact': NewsImpact.LOW, 'reason': 'News check not implemented'}
    
    async def _check_correlation_exposure(self, symbol: str, volume: float) -> Dict:
        """Check correlation exposure to prevent overexposure to correlated pairs."""
        try:
            # Get current positions
            positions = mt5.positions_get()
            if positions is None:
                positions = []
            
            # Simple correlation check (USD exposure)
            base_currency = symbol[:3]
            quote_currency = symbol[3:]
            
            usd_exposure = 0
            for pos in positions:
                pos_symbol = pos.symbol
                pos_base = pos_symbol[:3]
                pos_quote = pos_symbol[3:]
                
                # Calculate USD exposure
                if pos_base == 'USD':
                    usd_exposure += pos.volume if pos.type == 0 else -pos.volume
                elif pos_quote == 'USD':
                    usd_exposure += -pos.volume if pos.type == 0 else pos.volume
            
            # Add proposed trade USD exposure
            if base_currency == 'USD':
                proposed_usd_exposure = volume
            elif quote_currency == 'USD':
                proposed_usd_exposure = -volume
            else:
                proposed_usd_exposure = 0
            
            total_usd_exposure = abs(usd_exposure + proposed_usd_exposure)
            
            # Check if exposure exceeds limit (simplified check)
            max_usd_exposure = 5.0  # 5 lots max USD exposure
            
            if total_usd_exposure > max_usd_exposure:
                # Reduce position size to stay within limits
                adjusted_size = max(0.01, volume - (total_usd_exposure - max_usd_exposure))
                return {
                    'passed': False,
                    'reason': f'USD correlation exposure too high: {total_usd_exposure:.2f} lots',
                    'adjusted_size': adjusted_size
                }
            
            return {'passed': True, 'usd_exposure': total_usd_exposure}
            
        except Exception as e:
            logger.error(f"Error checking correlation exposure: {e}")
            return {'passed': True, 'reason': f'Correlation check error: {e}'}
    
    def reset_daily_counters(self):
        """Reset daily tracking counters (call at start of each trading day)."""
        self.daily_start_balance = None
        self.daily_trades_count = 0
        logger.info("🔄 Daily risk counters reset")
    
    def increment_trade_count(self):
        """Increment daily trade counter."""
        self.daily_trades_count += 1
        logger.debug(f"Daily trades: {self.daily_trades_count}/{self.max_daily_trades}")
