"""
Small Capital Risk Manager - Optimized for $100 accounts
Features:
- Practical position sizing for small accounts
- Dynamic risk adjustment based on performance
- Maximum loss protection
- Position correlation management
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)


class SmallCapitalRiskManager:
    """Risk manager optimized for small capital accounts ($100-$1000)."""
    
    def __init__(self, account_balance: float = 100.0):
        self.account_balance = account_balance
        self.initial_balance = account_balance
        
        # Risk parameters for small capital
        self.max_risk_per_trade = 0.05      # 5% = $5 per $100
        self.max_daily_loss = 0.10          # 10% = $10 per day
        self.max_weekly_loss = 0.20         # 20% = $20 per week
        self.max_total_risk = 0.15          # 15% = $15 total exposure
        
        # Position management
        self.max_positions = 3               # Maximum 3 open positions
        self.min_position_size = 0.01       # Minimum 0.01 lots
        self.max_position_size = 0.5        # Maximum 0.5 lots
        
        # Risk/reward requirements
        self.min_risk_reward = 2.0          # Minimum 2:1 ratio
        self.target_risk_reward = 3.0       # Target 3:1 ratio
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.total_pnl = 0.0
        self.trades_today = 0
        self.trades_this_week = 0
        
        # Risk adjustment based on performance
        self.risk_multiplier = 1.0          # Dynamic risk adjustment
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        
        # Time-based restrictions
        self.trading_hours = {
            'start': 8,    # 8 AM UTC
            'end': 20      # 8 PM UTC
        }
        
        logger.info(f"💰 Small Capital Risk Manager initialized for ${account_balance:.2f}")
        logger.info(f"🎯 Max risk per trade: ${account_balance * self.max_risk_per_trade:.2f}")
        logger.info(f"🛡️ Max daily loss: ${account_balance * self.max_daily_loss:.2f}")
    
    def update_account_balance(self, new_balance: float):
        """Update account balance and recalculate risk parameters."""
        old_balance = self.account_balance
        self.account_balance = new_balance
        
        # Calculate PnL
        balance_change = new_balance - old_balance
        self.total_pnl += balance_change
        
        # Update daily PnL
        current_date = datetime.now().date()
        if not hasattr(self, 'last_date') or self.last_date != current_date:
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.last_date = current_date
        
        self.daily_pnl += balance_change
        
        # Update weekly PnL
        current_week = datetime.now().isocalendar()[1]
        if not hasattr(self, 'last_week') or self.last_week != current_week:
            self.weekly_pnl = 0.0
            self.trades_this_week = 0
            self.last_week = current_week
        
        self.weekly_pnl += balance_change
        
        # Adjust risk based on performance
        self._adjust_risk_multiplier()
        
        logger.debug(f"Balance updated: ${new_balance:.2f} (Change: ${balance_change:+.2f})")
    
    def _adjust_risk_multiplier(self):
        """Dynamically adjust risk based on performance."""
        old_multiplier = self.risk_multiplier
        
        # Reduce risk after losses
        if self.consecutive_losses >= 2:
            self.risk_multiplier = max(0.5, self.risk_multiplier * 0.8)
        elif self.consecutive_losses >= 3:
            self.risk_multiplier = max(0.3, self.risk_multiplier * 0.7)
        
        # Increase risk after wins (cautiously)
        if self.consecutive_wins >= 3 and self.consecutive_losses == 0:
            self.risk_multiplier = min(1.2, self.risk_multiplier * 1.1)
        
        # Reset after significant losses
        if self.daily_pnl <= -(self.initial_balance * 0.05):  # -5% daily
            self.risk_multiplier = 0.5
        
        if self.risk_multiplier != old_multiplier:
            logger.info(f"🔄 Risk multiplier adjusted: {old_multiplier:.2f} → {self.risk_multiplier:.2f}")
    
    def can_trade(self, current_positions: List[Dict] = None) -> Dict[str, any]:
        """Check if trading is allowed based on current risk status."""
        result = {
            'can_trade': True,
            'reason': 'OK',
            'risk_level': 'LOW',
            'recommendations': []
        }
        
        # Check daily loss limit
        if self.daily_pnl <= -(self.account_balance * self.max_daily_loss):
            result['can_trade'] = False
            result['reason'] = f"Daily loss limit reached: ${self.daily_pnl:.2f}"
            result['risk_level'] = 'CRITICAL'
            return result
        
        # Check weekly loss limit
        if self.weekly_pnl <= -(self.account_balance * self.max_weekly_loss):
            result['can_trade'] = False
            result['reason'] = f"Weekly loss limit reached: ${self.weekly_pnl:.2f}"
            result['risk_level'] = 'CRITICAL'
            return result
        
        # Check position limits
        if current_positions and len(current_positions) >= self.max_positions:
            result['can_trade'] = False
            result['reason'] = f"Maximum positions reached: {len(current_positions)}"
            result['risk_level'] = 'HIGH'
            return result
        
        # Check total exposure
        if current_positions:
            total_exposure = sum(pos.get('risk_amount', 0) for pos in current_positions)
            max_exposure = self.account_balance * self.max_total_risk
            
            if total_exposure >= max_exposure:
                result['can_trade'] = False
                result['reason'] = f"Maximum exposure reached: ${total_exposure:.2f}"
                result['risk_level'] = 'HIGH'
                return result
        
        # Check trading hours
        current_hour = datetime.utcnow().hour
        if not (self.trading_hours['start'] <= current_hour <= self.trading_hours['end']):
            result['can_trade'] = False
            result['reason'] = f"Outside trading hours: {current_hour}:00 UTC"
            result['risk_level'] = 'MEDIUM'
            return result
        
        # Determine risk level
        if self.daily_pnl <= -(self.account_balance * 0.05):  # -5%
            result['risk_level'] = 'HIGH'
            result['recommendations'].append("Reduce position sizes")
        elif self.daily_pnl <= -(self.account_balance * 0.02):  # -2%
            result['risk_level'] = 'MEDIUM'
            result['recommendations'].append("Consider reducing risk")
        elif self.consecutive_losses >= 2:
            result['risk_level'] = 'MEDIUM'
            result['recommendations'].append("Risk reduced due to consecutive losses")
        
        return result
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              risk_amount: float = None) -> Dict[str, any]:
        """Calculate optimal position size for small capital."""
        try:
            if risk_amount is None:
                # Calculate risk amount based on account balance and risk multiplier
                base_risk = self.account_balance * self.max_risk_per_trade
                risk_amount = base_risk * self.risk_multiplier
            
            # Ensure minimum risk amount
            min_risk = self.account_balance * 0.01  # Minimum 1% risk
            risk_amount = max(risk_amount, min_risk)
            
            # Calculate pip value (simplified for major pairs)
            pip_value = 0.0001  # Standard pip size for most pairs
            
            # Calculate stop loss in pips
            sl_pips = abs(entry_price - stop_loss) / pip_value
            
            if sl_pips == 0:
                return {
                    'position_size': 0.01,
                    'risk_amount': min_risk,
                    'pip_value': 0.0,
                    'warning': 'Invalid stop loss'
                }
            
            # Calculate position size based on risk
            # Simplified calculation: risk_amount / (sl_pips * 10)
            position_size = risk_amount / (sl_pips * 10)
            
            # Apply position size limits
            position_size = max(self.min_position_size, 
                              min(position_size, self.max_position_size))
            
            # Round to 2 decimal places
            position_size = round(position_size, 2)
            
            # Recalculate actual risk amount
            actual_risk = sl_pips * 10 * position_size
            
            # Calculate pip value for this position
            pip_value_usd = position_size * 10  # Simplified: 1 lot = $10 per pip
            
            result = {
                'position_size': position_size,
                'risk_amount': round(actual_risk, 2),
                'pip_value': round(pip_value_usd, 2),
                'sl_pips': round(sl_pips, 1),
                'risk_percentage': round((actual_risk / self.account_balance) * 100, 2)
            }
            
            logger.debug(f"Position size calculated: {position_size} lots, Risk: ${actual_risk:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Position size calculation error: {e}")
            return {
                'position_size': 0.01,
                'risk_amount': min_risk,
                'pip_value': 0.0,
                'error': str(e)
            }
    
    def validate_trade(self, signal: Dict, current_positions: List[Dict] = None) -> Dict[str, any]:
        """Validate a trading signal with comprehensive risk checks."""
        try:
            result = {
                'approved': False,
                'reason': '',
                'position_size': 0.0,
                'risk_amount': 0.0,
                'warnings': [],
                'recommendations': []
            }
            
            # Check if trading is allowed
            trade_status = self.can_trade(current_positions)
            if not trade_status['can_trade']:
                result['reason'] = trade_status['reason']
                return result
            
            # Validate signal data
            required_fields = ['entry_price', 'stop_loss', 'take_profit', 'direction']
            for field in required_fields:
                if field not in signal or signal[field] == 0:
                    result['reason'] = f"Missing or invalid {field}"
                    return result
            
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']
            
            # Check risk/reward ratio
            if signal['direction'] == 'buy':
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
            else:  # sell
                risk = stop_loss - entry_price
                reward = entry_price - take_profit
            
            if risk <= 0 or reward <= 0:
                result['reason'] = "Invalid risk/reward calculation"
                return result
            
            risk_reward_ratio = reward / risk
            if risk_reward_ratio < self.min_risk_reward:
                result['reason'] = f"Insufficient risk/reward ratio: {risk_reward_ratio:.2f}"
                return result
            
            # Calculate position size
            size_calc = self.calculate_position_size(entry_price, stop_loss)
            if 'error' in size_calc:
                result['reason'] = f"Position size calculation error: {size_calc['error']}"
                return result
            
            # Check if position size is too small
            if size_calc['position_size'] < self.min_position_size:
                result['reason'] = f"Position size too small: {size_calc['position_size']}"
                return result
            
            # Check risk percentage
            if size_calc['risk_percentage'] > (self.max_risk_per_trade * 100):
                result['warnings'].append(f"Risk per trade ({size_calc['risk_percentage']}%) exceeds target ({self.max_risk_per_trade * 100}%)")
            
            # Check if this would exceed daily loss limit
            potential_daily_loss = self.daily_pnl - size_calc['risk_amount']
            if potential_daily_loss < -(self.account_balance * self.max_daily_loss):
                result['warnings'].append("Trade would exceed daily loss limit")
            
            # Approve trade with calculated parameters
            result['approved'] = True
            result['position_size'] = size_calc['position_size']
            result['risk_amount'] = size_calc['risk_amount']
            result['risk_percentage'] = size_calc['risk_percentage']
            result['pip_value'] = size_calc['pip_value']
            
            # Add recommendations based on risk level
            if trade_status['risk_level'] == 'HIGH':
                result['recommendations'].append("Consider reducing position size due to high risk level")
            elif trade_status['risk_level'] == 'MEDIUM':
                result['recommendations'].append("Monitor risk closely")
            
            if risk_reward_ratio >= self.target_risk_reward:
                result['recommendations'].append("Excellent risk/reward ratio")
            
            logger.info(f"✅ Trade validated: {signal.get('pair', 'Unknown')} - {size_calc['position_size']} lots, Risk: ${size_calc['risk_amount']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Trade validation error: {e}")
            result['reason'] = f"Validation error: {e}"
            return result
    
    def record_trade_result(self, profit_loss: float):
        """Record trade result and update performance metrics."""
        try:
            # Update PnL
            self.total_pnl += profit_loss
            self.daily_pnl += profit_loss
            self.weekly_pnl += profit_loss
            
            # Update consecutive wins/losses
            if profit_loss > 0:
                self.consecutive_wins += 1
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
                self.consecutive_wins = 0
            
            # Update trade counts
            self.trades_today += 1
            self.trades_this_week += 1
            
            # Log performance
            logger.info(f"📊 Trade result: ${profit_loss:+.2f}")
            logger.info(f"   Daily PnL: ${self.daily_pnl:+.2f}")
            logger.info(f"   Total PnL: ${self.total_pnl:+.2f}")
            logger.info(f"   Consecutive: {self.consecutive_wins} wins, {self.consecutive_losses} losses")
            
        except Exception as e:
            logger.error(f"Error recording trade result: {e}")
    
    def get_risk_summary(self) -> Dict[str, any]:
        """Get comprehensive risk summary."""
        try:
            return {
                'account_balance': self.account_balance,
                'total_pnl': self.total_pnl,
                'daily_pnl': self.daily_pnl,
                'weekly_pnl': self.weekly_pnl,
                'pnl_percentage': (self.total_pnl / self.initial_balance) * 100,
                'risk_multiplier': self.risk_multiplier,
                'consecutive_wins': self.consecutive_wins,
                'consecutive_losses': self.consecutive_losses,
                'trades_today': self.trades_today,
                'trades_this_week': self.trades_this_week,
                'max_risk_per_trade': self.max_risk_per_trade * 100,
                'max_daily_loss': self.max_daily_loss * 100,
                'max_weekly_loss': self.max_weekly_loss * 100,
                'max_positions': self.max_positions,
                'min_risk_reward': self.min_risk_reward,
                'target_risk_reward': self.target_risk_reward
            }
        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {} 