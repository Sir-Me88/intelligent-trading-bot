#!/usr/bin/env python3
"""Advanced Position Management System with ML integration."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class PositionTracker:
    """Track position performance over time."""
    ticket: int
    symbol: str
    direction: str
    volume: float
    open_price: float
    open_time: datetime
    peak_profit: float = 0.0
    current_profit: float = 0.0
    max_drawdown: float = 0.0
    profit_history: List[Tuple[datetime, float]] = None
    
    def __post_init__(self):
        if self.profit_history is None:
            self.profit_history = []

class AdvancedPositionManager:
    """Advanced position management with ML-driven decisions."""
    
    def __init__(self, broker_manager, ml_analyzer=None):
        self.broker_manager = broker_manager
        self.ml_analyzer = ml_analyzer
        
        # Position tracking
        self.position_trackers: Dict[int, PositionTracker] = {}
        
        # Management settings
        self.loss_threshold = -30.0  # Close positions losing more than $30
        self.profit_protection_threshold = 30.0  # Protect profits when declining by $30
        self.scaling_profit_threshold = 50.0  # Add positions when profit > $50
        self.max_positions_per_pair = 5  # Maximum positions per currency pair
        
        # Tracking files
        self.data_dir = Path("data/positions")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance metrics
        self.positions_closed_loss = 0
        self.positions_closed_profit = 0
        self.total_loss_prevented = 0.0
        self.total_profit_protected = 0.0
        
    async def update_position_tracking(self):
        """Update tracking for all open positions."""
        try:
            positions = await self.broker_manager.get_positions()
            current_time = datetime.now()
            
            for position in positions:
                ticket = position.get('ticket')
                current_profit = position.get('profit', 0)
                
                if ticket not in self.position_trackers:
                    # Create new tracker
                    self.position_trackers[ticket] = PositionTracker(
                        ticket=ticket,
                        symbol=position.get('symbol', ''),
                        direction=position.get('type', ''),
                        volume=position.get('volume', 0),
                        open_price=position.get('price_open', 0),
                        open_time=current_time,
                        current_profit=current_profit,
                        peak_profit=max(0, current_profit)
                    )
                else:
                    # Update existing tracker
                    tracker = self.position_trackers[ticket]
                    tracker.current_profit = current_profit
                    
                    # Update peak profit
                    if current_profit > tracker.peak_profit:
                        tracker.peak_profit = current_profit
                        
                    # Update max drawdown from peak
                    drawdown = tracker.peak_profit - current_profit
                    if drawdown > tracker.max_drawdown:
                        tracker.max_drawdown = drawdown
                        
                    # Add to profit history
                    tracker.profit_history.append((current_time, current_profit))
                    
                    # Keep only last 100 entries
                    if len(tracker.profit_history) > 100:
                        tracker.profit_history = tracker.profit_history[-100:]
                        
            # Remove trackers for closed positions
            open_tickets = {pos.get('ticket') for pos in positions}
            closed_tickets = set(self.position_trackers.keys()) - open_tickets
            
            for ticket in closed_tickets:
                tracker = self.position_trackers.pop(ticket)
                await self.record_closed_position(tracker)
                
        except Exception as e:
            logger.error(f"Error updating position tracking: {e}")
            
    async def record_closed_position(self, tracker: PositionTracker):
        """Record closed position for ML analysis."""
        try:
            if self.ml_analyzer:
                trade_data = {
                    'ticket': tracker.ticket,
                    'symbol': tracker.symbol,
                    'direction': tracker.direction,
                    'volume': tracker.volume,
                    'open_price': tracker.open_price,
                    'open_time': tracker.open_time.isoformat(),
                    'close_time': datetime.now().isoformat(),
                    'final_profit': tracker.current_profit,
                    'peak_profit': tracker.peak_profit,
                    'max_drawdown': tracker.max_drawdown,
                    'is_profitable': tracker.current_profit > 0,
                    'duration_minutes': (datetime.now() - tracker.open_time).total_seconds() / 60
                }
                
                self.ml_analyzer.record_trade(trade_data)
                
        except Exception as e:
            logger.error(f"Error recording closed position: {e}")
            
    async def manage_losing_positions(self) -> List[Dict]:
        """Close positions losing more than threshold."""
        closed_positions = []
        
        try:
            for ticket, tracker in list(self.position_trackers.items()):
                if tracker.current_profit <= self.loss_threshold:
                    logger.info(f"🔴 AUTO-CLOSING LOSING POSITION: {tracker.symbol} Ticket {ticket}")
                    logger.info(f"   Loss: ${tracker.current_profit:.2f} (threshold: ${self.loss_threshold})")
                    
                    try:
                        close_result = await self.broker_manager.close_position(ticket)
                        
                        if close_result:
                            self.positions_closed_loss += 1
                            self.total_loss_prevented += abs(tracker.current_profit)
                            
                            closed_positions.append({
                                'ticket': ticket,
                                'symbol': tracker.symbol,
                                'reason': 'auto_loss_management',
                                'profit': tracker.current_profit
                            })
                            
                            logger.info(f"✅ Auto loss management: {tracker.symbol} closed at ${tracker.current_profit:.2f}")
                        else:
                            logger.error(f"❌ Failed to close losing position: {tracker.symbol}")
                            
                    except Exception as e:
                        logger.error(f"Error closing losing position {ticket}: {e}")
                        
        except Exception as e:
            logger.error(f"Error managing losing positions: {e}")
            
        return closed_positions
        
    async def manage_profit_protection(self) -> List[Dict]:
        """Close winning positions when they decline by threshold from peak."""
        protected_positions = []
        
        try:
            for ticket, tracker in list(self.position_trackers.items()):
                # Only protect positions that have been profitable
                if tracker.peak_profit > 0 and tracker.max_drawdown >= self.profit_protection_threshold:
                    logger.info(f"💎 AUTO-PROTECTING PROFITS: {tracker.symbol} Ticket {ticket}")
                    logger.info(f"   Peak Profit: ${tracker.peak_profit:.2f}")
                    logger.info(f"   Current Profit: ${tracker.current_profit:.2f}")
                    logger.info(f"   Drawdown: ${tracker.max_drawdown:.2f} (threshold: ${self.profit_protection_threshold})")
                    
                    try:
                        close_result = await self.broker_manager.close_position(ticket)
                        
                        if close_result:
                            self.positions_closed_profit += 1
                            self.total_profit_protected += tracker.current_profit
                            
                            protected_positions.append({
                                'ticket': ticket,
                                'symbol': tracker.symbol,
                                'reason': 'auto_profit_protection',
                                'profit': tracker.current_profit,
                                'peak_profit': tracker.peak_profit,
                                'drawdown': tracker.max_drawdown
                            })
                            
                            logger.info(f"✅ Profit protected: {tracker.symbol} closed at ${tracker.current_profit:.2f}")
                        else:
                            logger.error(f"❌ Failed to close position for profit protection: {tracker.symbol}")
                            
                    except Exception as e:
                        logger.error(f"Error closing position for profit protection {ticket}: {e}")
                        
        except Exception as e:
            logger.error(f"Error managing profit protection: {e}")
            
        return protected_positions
        
    async def identify_scaling_opportunities(self) -> List[Dict]:
        """Identify positions suitable for scaling up."""
        scaling_opportunities = []
        
        try:
            # Count positions per pair
            pair_counts = {}
            for tracker in self.position_trackers.values():
                pair_counts[tracker.symbol] = pair_counts.get(tracker.symbol, 0) + 1
                
            for ticket, tracker in self.position_trackers.items():
                # Check if position is profitable enough for scaling
                if (tracker.current_profit >= self.scaling_profit_threshold and
                    pair_counts[tracker.symbol] < self.max_positions_per_pair):
                    
                    # Calculate suggested scaling size
                    profit_multiple = tracker.current_profit / self.scaling_profit_threshold
                    suggested_volume = min(tracker.volume * 1.5, tracker.volume + 0.3)  # Max 1.5x or +0.3 lots
                    
                    scaling_opportunities.append({
                        'base_ticket': ticket,
                        'symbol': tracker.symbol,
                        'direction': tracker.direction,
                        'current_profit': tracker.current_profit,
                        'suggested_volume': suggested_volume,
                        'profit_multiple': profit_multiple,
                        'current_positions': pair_counts[tracker.symbol]
                    })
                    
                    logger.info(f"🚀 SCALING OPPORTUNITY DETECTED: {tracker.symbol}")
                    logger.info(f"   Current Profit: ${tracker.current_profit:.2f}")
                    logger.info(f"   Suggested Volume: {suggested_volume} lots")
                    logger.info(f"   Current Positions: {pair_counts[tracker.symbol]}")
                    
        except Exception as e:
            logger.error(f"Error identifying scaling opportunities: {e}")
            
        return scaling_opportunities

    async def execute_position_scaling(self, opportunity: Dict, signal_data: Dict) -> Optional[Dict]:
        """Execute position scaling based on opportunity."""
        try:
            symbol = opportunity['symbol']
            direction = opportunity['direction']
            volume = opportunity['suggested_volume']

            logger.info(f"🚀 EXECUTING POSITION SCALING: {symbol}")
            logger.info(f"   Direction: {direction}")
            logger.info(f"   Volume: {volume} lots")
            logger.info(f"   Base Position Profit: ${opportunity['current_profit']:.2f}")

            # Use ML prediction if available
            if self.ml_analyzer:
                prediction = self.ml_analyzer.predict_trade_outcome({
                    'symbol': symbol,
                    'direction': direction,
                    'volume': volume,
                    'confidence': signal_data.get('confidence', 0.75),
                    'rr_ratio': signal_data.get('rr_ratio', 3.0),
                    'timestamp': datetime.now().isoformat()
                })

                if prediction.get('success_probability', 0.5) < 0.6:
                    logger.info(f"⚠️ ML prediction suggests low success probability: {prediction.get('success_probability', 0):.2f}")
                    return None

            # Execute the scaling trade
            order_result = await self.broker_manager.place_order(
                symbol=symbol,
                order_type=direction,
                volume=volume,
                price=signal_data['entry_price'],
                sl=signal_data['stop_loss'],
                tp=signal_data['take_profit']
            )

            if order_result and order_result.get('ticket'):
                logger.info(f"✅ SCALING TRADE EXECUTED!")
                logger.info(f"📋 New Ticket: {order_result['ticket']}")
                logger.info(f"💼 Scaled {direction} {volume} lots {symbol}")

                return {
                    'ticket': order_result['ticket'],
                    'symbol': symbol,
                    'direction': direction,
                    'volume': volume,
                    'base_ticket': opportunity['base_ticket'],
                    'scaling_reason': 'high_profit_performance'
                }
            else:
                logger.error(f"❌ Scaling trade execution failed: {order_result}")
                return None

        except Exception as e:
            logger.error(f"Error executing position scaling: {e}")
            return None

    async def get_management_statistics(self) -> Dict:
        """Get position management statistics."""
        try:
            positions = await self.broker_manager.get_positions()
            total_unrealized = sum(pos.get('profit', 0) for pos in positions)

            # Calculate average profit per position type
            profitable_positions = [p for p in positions if p.get('profit', 0) > 0]
            losing_positions = [p for p in positions if p.get('profit', 0) < 0]

            stats = {
                'total_positions': len(positions),
                'profitable_positions': len(profitable_positions),
                'losing_positions': len(losing_positions),
                'total_unrealized_pnl': total_unrealized,
                'avg_profit_per_winner': sum(p.get('profit', 0) for p in profitable_positions) / max(len(profitable_positions), 1),
                'avg_loss_per_loser': sum(p.get('profit', 0) for p in losing_positions) / max(len(losing_positions), 1),
                'positions_closed_loss': self.positions_closed_loss,
                'positions_closed_profit': self.positions_closed_profit,
                'total_loss_prevented': self.total_loss_prevented,
                'total_profit_protected': self.total_profit_protected,
                'win_rate': len(profitable_positions) / max(len(positions), 1) * 100
            }

            return stats

        except Exception as e:
            logger.error(f"Error getting management statistics: {e}")
            return {}

    async def comprehensive_position_management(self) -> Dict:
        """Run comprehensive position management cycle."""
        try:
            logger.info("🔄 Running comprehensive position management...")

            # Update position tracking
            await self.update_position_tracking()

            # Manage losing positions
            closed_losses = await self.manage_losing_positions()

            # Manage profit protection
            protected_profits = await self.manage_profit_protection()

            # Identify scaling opportunities
            scaling_opportunities = await self.identify_scaling_opportunities()

            # Get statistics
            stats = await self.get_management_statistics()

            management_report = {
                'timestamp': datetime.now().isoformat(),
                'closed_losses': closed_losses,
                'protected_profits': protected_profits,
                'scaling_opportunities': scaling_opportunities,
                'statistics': stats
            }

            # Log summary
            logger.info(f"📊 Position Management Summary:")
            logger.info(f"   Positions Closed (Loss): {len(closed_losses)}")
            logger.info(f"   Positions Protected (Profit): {len(protected_profits)}")
            logger.info(f"   Scaling Opportunities: {len(scaling_opportunities)}")
            logger.info(f"   Total Positions: {stats.get('total_positions', 0)}")
            logger.info(f"   Win Rate: {stats.get('win_rate', 0):.1f}%")

            return management_report

        except Exception as e:
            logger.error(f"Error in comprehensive position management: {e}")
            return {"error": str(e)}
