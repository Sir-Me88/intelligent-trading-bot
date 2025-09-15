#!/usr/bin/env python3
"""Adaptive Intelligent Trading Bot with ML Learning and Real-time Reversal Detection."""

import os
import sys
import asyncio
import logging
import signal
import traceback
import time
import warnings
from collections import deque
import numpy as np  # Added for RL state calculations
from datetime import datetime, timedelta, timezone  # Added timezone
from pathlib import Path
from typing import Dict, List, Optional, Any  # Added type hints
from dotenv import load_dotenv
import json
import requests

# Suppress gym deprecation warnings
warnings.filterwarnings("ignore", message=".*Gym.*unmaintained.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*gym.*deprecated.*", category=DeprecationWarning)

# Add src to path
sys.path.append('src')

from src.config.settings import settings
from src.data.market_data import MarketDataManager
from src.trading.broker_interface import BrokerManager
from src.analysis.technical import TechnicalAnalyzer, SignalDirection
from src.analysis.correlation import CorrelationAnalyzer

# Advanced components (lazy-loaded to prevent import errors)
try:
    from src.analysis.trend_reversal_detector import TrendReversalDetector
    REVERSAL_DETECTOR_AVAILABLE = True
except ImportError:
    REVERSAL_DETECTOR_AVAILABLE = False

try:
    from src.ml.trading_ml_engine import TradingMLEngine
    ML_ENGINE_AVAILABLE = True
except ImportError:
    ML_ENGINE_AVAILABLE = False

try:
    from src.ml.trade_analyzer import TradeAnalyzer
    TRADE_ANALYZER_AVAILABLE = True
except ImportError:
    TRADE_ANALYZER_AVAILABLE = False

try:
    from src.scheduling.intelligent_scheduler import IntelligentTradingScheduler
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False

try:
    from src.news.sentiment import SentimentAggregator
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

try:
    from src.monitoring.metrics import MetricsCollector
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

try:
    from src.analysis.trade_attribution import TradeAttributionAnalyzer
    ATTRIBUTION_AVAILABLE = True
except ImportError:
    ATTRIBUTION_AVAILABLE = False

# Setup logging
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / "adaptive_intelligent_bot.log", encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

class AdaptiveIntelligentBot:
    """Advanced adaptive trading bot with ML learning and real-time reversal detection."""
    
    def __init__(self):
        logger.info("🤖 INITIALIZING ADAPTIVE INTELLIGENT BOT")
        print("[INIT] AdaptiveIntelligentBot initializing...")

        # Core components (always available)
        self.data_manager = MarketDataManager()
        self.broker_manager = BrokerManager()
        self.technical_analyzer = TechnicalAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer(self.data_manager)

        # Advanced components (lazy-loaded to prevent startup delays)
        self.reversal_detector = None
        self.ml_engine = None
        self.trade_analyzer = None
        self.scheduler = None
        self.sentiment_aggregator = None
        self.metrics_collector = None
        self.attribution_analyzer = None

        # Load adaptive parameters from file if exists, otherwise use defaults
        self.adaptive_params = self._load_adaptive_params()

        # Trading state
        self.running = True
        self.current_mode = 'TRADING'
        self.position_trackers = {}
        self.paper_mode = os.getenv('PAPER_MODE', 'false').lower() == 'true'

        # Capped buffers to prevent memory leaks
        self.daily_trade_data = deque(maxlen=100)   # Max 100 trades per day
        self.weekly_trade_data = deque(maxlen=500)  # Max 500 trades per week
        self.rl_state_buffer = deque(maxlen=1000)   # Max 1000 states for RL

        # Performance tracking
        self.heartbeat_file = Path("logs") / "adaptive_bot_heartbeat.json"
        self.scan_count = 0
        self.trades_executed = 0
        self.signals_analyzed = 0
        self.signals_rejected = 0
        self.reversals_detected = 0
        self.positions_closed_reversal = 0
        self.positions_closed_profit = 0
        self.positions_closed_loss = 0
        self.ml_analyses_performed = 0

        # Learning data
        self.last_daily_analysis = None
        self.last_weekly_analysis = None

        logger.info("✅ Adaptive intelligent bot initialized with capped buffers")
        print("[INIT] Initialization complete.")

    def _load_adaptive_params(self) -> Dict:
        """Load adaptive parameters from file or use defaults."""
        param_file = Path('data/adaptive_params.json')
        default_params = {
            'min_confidence': 0.75,  # Conservative default
            'min_rr_ratio': 2.5,     # Reasonable risk/reward
            'profit_protection_percentage': 0.25,
            'reversal_confidence_threshold': 0.75,
            'max_volatility': 0.002,
            'minimum_profit_to_protect': 20.0,
            'atr_multiplier_low_vol': 2.0,
            'atr_multiplier_normal_vol': 2.5,
            'atr_multiplier_high_vol': 3.0,
            'volatility_threshold_low': 0.001,
            'volatility_threshold_high': 0.003
        }

        if param_file.exists():
            try:
                with open(param_file, 'r') as f:
                    file_params = json.load(f)
                # Merge file params with defaults
                merged_params = {**default_params, **file_params}
                logger.info(f"📄 Loaded adaptive parameters from {param_file}")
                return merged_params
            except Exception as e:
                logger.warning(f"Failed to load parameters from {param_file}: {e}")

        logger.info("📄 Using default adaptive parameters")
        return default_params
        
    def signal_handler(self, signum: int, frame: Optional[Any]) -> None:
        """Handle system signals gracefully."""
        logger.info(f"SIGNAL RECEIVED: {signum} - Shutting down gracefully...")
        self.running = False
        
    async def write_heartbeat(self):
        """Write comprehensive heartbeat with adaptive parameters."""
        try:
            # Safely get scheduler info
            schedule_info = "Scheduler not available"
            if self.scheduler:
                try:
                    schedule_info = str(self.scheduler.get_trading_schedule_info())
                except:
                    schedule_info = "Scheduler error"

            heartbeat_data = {
                'timestamp': datetime.now().isoformat(),
                'mode': self.current_mode,
                'scan_count': self.scan_count,
                'trades_executed': self.trades_executed,
                'signals_analyzed': self.signals_analyzed,
                'signals_rejected': self.signals_rejected,
                'reversals_detected': self.reversals_detected,
                'positions_closed_reversal': self.positions_closed_reversal,
                'positions_closed_profit': self.positions_closed_profit,
                'positions_closed_loss': self.positions_closed_loss,
                'ml_analyses_performed': self.ml_analyses_performed,
                'adaptive_parameters': self.adaptive_params,
                'running': self.running,
                'schedule_info': schedule_info,
                'components_available': {
                    'reversal_detector': self.reversal_detector is not None,
                    'ml_engine': self.ml_engine is not None,
                    'scheduler': self.scheduler is not None,
                    'sentiment_aggregator': self.sentiment_aggregator is not None,
                    'metrics_collector': self.metrics_collector is not None,
                    'attribution_analyzer': self.attribution_analyzer is not None
                }
            }

            with open(self.heartbeat_file, 'w') as f:
                json.dump(heartbeat_data, f, indent=2)

        except Exception as e:
            logger.error(f"Error writing heartbeat: {e}")
            
    async def mode_change_handler(self, old_mode: str, new_mode: str):
        """Handle mode changes from scheduler."""
        logger.info(f"🔄 MODE CHANGE: {old_mode} → {new_mode}")
        self.current_mode = new_mode
        
        if new_mode == 'DAILY_ANALYSIS':
            await self.perform_daily_ml_analysis()
        elif new_mode == 'WEEKEND_ANALYSIS':
            await self.perform_weekend_ml_analysis()
        elif new_mode == 'TRADING':
            logger.info("🚀 RESUMING TRADING MODE")
            
    async def real_time_reversal_monitoring(self):
        """Enhanced monitoring with trailing stops and Chandelier Exit integration."""
        try:
            # Skip if reversal detector not available
            if not REVERSAL_DETECTOR_AVAILABLE or self.reversal_detector is None:
                logger.debug("Reversal detector not available - skipping advanced monitoring")
                return

            positions = await self.broker_manager.get_positions()

            for position in positions:
                ticket = position.get('ticket')
                symbol = position.get('symbol')
                profit = position.get('profit', 0)
                position_type = position.get('type')  # 0=BUY, 1=SELL
                entry_price = position.get('price_open', 0)
                current_sl = position.get('sl', 0)

                # Get current market data for reversal analysis - ALL TIMEFRAMES
                try:
                    df_1m = await self.data_manager.get_candles(symbol, "M1", 100)
                    df_5m = await self.data_manager.get_candles(symbol, "M5", 100)
                    df_15m = await self.data_manager.get_candles(symbol, "M15", 50)
                    df_1h = await self.data_manager.get_candles(symbol, "H1", 50)
                    df_h4 = await self.data_manager.get_candles(symbol, "H4", 30)

                    if df_15m is None or df_1h is None:
                        continue

                    # Prepare comprehensive data for reversal detection
                    reversal_data = {
                        'df_1m': df_1m,
                        'df_5m': df_5m,
                        'df_15m': df_15m,
                        'df_1h': df_1h,
                        'df_h4': df_h4,
                        'current_position': 'LONG' if position_type == 0 else 'SHORT'
                    }

                    # ENHANCED TRAILING STOPS WITH CHANDELIER EXIT
                    await self._apply_trailing_stops(ticket, symbol, position_type, profit, entry_price, current_sl, df_15m)

                    # Detect trend reversal
                    reversal_result = self.reversal_detector.detect_trend_reversal(symbol, reversal_data)

                    # Check Chandelier Exit signals
                    chandelier_result = self.reversal_detector.detect_chandelier_exit_signal(
                        symbol, reversal_data, 'LONG' if position_type == 0 else 'SHORT'
                    )

                    # Combine reversal and Chandelier signals
                    should_exit = False
                    exit_reason = None
                    exit_confidence = 0.0

                    if reversal_result.get('reversal_detected', False):
                        self.reversals_detected += 1
                        should_exit = True
                        exit_reason = 'TREND_REVERSAL'
                        exit_confidence = reversal_result['confidence']

                        logger.info(f"🔄 TREND REVERSAL DETECTED: {symbol} #{ticket}")
                        logger.info(f"   Confidence: {reversal_result['confidence']:.2f}")
                        logger.info(f"   Direction: {reversal_result['direction']}")
                        logger.info(f"   Action: {reversal_result['immediate_action']}")

                    # Check Chandelier Exit (dynamic trailing stop)
                    if chandelier_result.get('chandelier_signal', 'HOLD').startswith('EXIT'):
                        if not should_exit or chandelier_result['confidence'] > exit_confidence:
                            should_exit = True
                            exit_reason = 'CHANDELIER_EXIT'
                            exit_confidence = chandelier_result['confidence']

                        logger.info(f"🚪 CHANDELIER EXIT TRIGGERED: {symbol} #{ticket}")
                        logger.info(f"   Confidence: {chandelier_result['confidence']:.2f}")
                        logger.info(f"   Exit Level: {chandelier_result['details']['exit_level_15m']:.5f}")
                        logger.info(f"   Distance: {chandelier_result['details']['distance_pips_15m']:.1f} pips")

                    # Execute exit if either signal triggered
                    if should_exit:
                        logger.info(f"🚨 EXECUTING {exit_reason}: {symbol} #{ticket}")

                        try:
                            close_result = await self.broker_manager.close_position(ticket)
                            if close_result:
                                self.positions_closed_reversal += 1

                                # Enhanced trade logging for ML learning
                                exit_data = {
                                    'timestamp': datetime.now().isoformat(),
                                    'ticket': ticket,
                                    'symbol': symbol,
                                    'exit_reason': exit_reason,
                                    'exit_confidence': exit_confidence,
                                    'profit_at_exit': profit,
                                    'position_type': 'LONG' if position_type == 0 else 'SHORT',
                                    'hold_duration': (datetime.now() - datetime.fromisoformat(self.position_trackers.get(ticket, {}).get('entry_time', datetime.now().isoformat()))).total_seconds() / 3600,
                                    'market_conditions': {
                                        'volatility_15m': df_15m['close'].pct_change().std() if len(df_15m) > 1 else 0,
                                        'volatility_1h': df_1h['close'].pct_change().std() if len(df_1h) > 1 else 0,
                                        'price_at_exit': df_15m['close'].iloc[-1] if len(df_15m) > 0 else 0
                                    }
                                }

                                # Add specific data based on exit reason
                                if exit_reason == 'TREND_REVERSAL':
                                    exit_data.update({
                                        'reversal_direction': reversal_result['direction'],
                                        'reversal_signals': reversal_result.get('signals', [])
                                    })
                                elif exit_reason == 'CHANDELIER_EXIT':
                                    exit_data.update({
                                        'chandelier_exit_level': chandelier_result['details']['exit_level_15m'],
                                        'distance_pips': chandelier_result['details']['distance_pips_15m'],
                                        'atr_value': chandelier_result['details']['atr_15m']
                                    })

                                self.daily_trade_data.append(exit_data)
                                self.weekly_trade_data.append(exit_data)

                                logger.info(f"✅ {exit_reason} SUCCESSFUL: {symbol} at ${profit:.2f}")

                                # Clean up position tracker
                                if ticket in self.position_trackers:
                                    del self.position_trackers[ticket]

                        except Exception as e:
                            logger.error(f"Error executing {exit_reason} for {ticket}: {e}")

                except Exception as e:
                    logger.error(f"Error in reversal monitoring for {symbol}: {e}")

        except Exception as e:
            logger.error(f"Error in real-time reversal monitoring: {e}")
    
    async def _apply_trailing_stops(self, ticket: int, symbol: str, position_type: int, 
                                  profit: float, entry_price: float, current_sl: float, df_15m) -> None:
        """Apply trailing stops using Chandelier Exit after position reaches minimum profit."""
        try:
            # Only apply trailing stops if position is profitable above minimum threshold
            if profit < self.adaptive_params['minimum_profit_to_protect']:
                return
            
            # Calculate Chandelier Exit levels
            chandelier_levels = self.technical_analyzer.calculate_chandelier_exit(df_15m, multiplier=3.0)
            
            if position_type == 0:  # LONG position
                new_sl = chandelier_levels['long_exit']
                
                # Only move stop loss up (never down for long positions)
                if new_sl > current_sl and new_sl < entry_price + (profit * 0.8):  # Don't trail too aggressively
                    try:
                        # Note: This would require implementing modify_position in broker_interface
                        # For now, we'll log the trailing stop action
                        logger.info(f"🔄 TRAILING STOP UPDATE: {symbol} #{ticket}")
                        logger.info(f"   Current SL: {current_sl:.5f} → New SL: {new_sl:.5f}")
                        logger.info(f"   Chandelier Exit Level: {new_sl:.5f}")
                        logger.info(f"   Profit Protection: ${profit:.2f}")
                        
                        # TODO: Implement broker_manager.modify_position(ticket, sl=new_sl)
                        # await self.broker_manager.modify_position(ticket, sl=new_sl)
                        
                    except Exception as e:
                        logger.error(f"Error updating trailing stop for {ticket}: {e}")
                        
            else:  # SHORT position
                new_sl = chandelier_levels['short_exit']
                
                # Only move stop loss down (never up for short positions)
                if new_sl < current_sl and new_sl > entry_price - (profit * 0.8):  # Don't trail too aggressively
                    try:
                        logger.info(f"🔄 TRAILING STOP UPDATE: {symbol} #{ticket}")
                        logger.info(f"   Current SL: {current_sl:.5f} → New SL: {new_sl:.5f}")
                        logger.info(f"   Chandelier Exit Level: {new_sl:.5f}")
                        logger.info(f"   Profit Protection: ${profit:.2f}")
                        
                        # TODO: Implement broker_manager.modify_position(ticket, sl=new_sl)
                        # await self.broker_manager.modify_position(ticket, sl=new_sl)
                        
                    except Exception as e:
                        logger.error(f"Error updating trailing stop for {ticket}: {e}")
                        
        except Exception as e:
            logger.error(f"Error in trailing stops for {ticket}: {e}")
            
    async def adaptive_percentage_protection(self):
        """Enhanced percentage-based protection with adaptive parameters."""
        try:
            positions = await self.broker_manager.get_positions()
            
            for position in positions:
                ticket = position.get('ticket')
                profit = position.get('profit', 0)
                symbol = position.get('symbol', '')
                
                # Initialize or update position tracker
                if ticket not in self.position_trackers:
                    self.position_trackers[ticket] = {
                        'peak_profit': max(0, profit),
                        'symbol': symbol,
                        'entry_time': datetime.now().isoformat()
                    }
                else:
                    if profit > self.position_trackers[ticket]['peak_profit']:
                        self.position_trackers[ticket]['peak_profit'] = profit
                        
                tracker = self.position_trackers[ticket]
                peak_profit = tracker['peak_profit']
                
                # Adaptive loss protection
                if profit <= -self.adaptive_params['minimum_profit_to_protect']:
                    try:
                        close_result = await self.broker_manager.close_position(ticket)
                        if close_result:
                            self.positions_closed_loss += 1
                            logger.info(f"🔴 ADAPTIVE LOSS PROTECTION: {symbol} closed at ${profit:.2f}")
                            del self.position_trackers[ticket]
                    except Exception as e:
                        logger.error(f"Error in adaptive loss protection {ticket}: {e}")
                        
                # Adaptive profit protection
                elif peak_profit >= self.adaptive_params['minimum_profit_to_protect']:
                    drawdown_amount = peak_profit - profit
                    drawdown_percentage = (drawdown_amount / peak_profit) if peak_profit > 0 else 0
                    
                    if drawdown_percentage >= self.adaptive_params['profit_protection_percentage']:
                        try:
                            close_result = await self.broker_manager.close_position(ticket)
                            if close_result:
                                self.positions_closed_profit += 1
                                
                                # Enhanced trade logging for ML learning
                                protection_data = {
                                    'timestamp': datetime.now().isoformat(),
                                    'ticket': ticket,
                                    'symbol': symbol,
                                    'exit_reason': 'ADAPTIVE_PROFIT_PROTECTION',
                                    'peak_profit': peak_profit,
                                    'exit_profit': profit,
                                    'drawdown_percentage': drawdown_percentage,
                                    'protection_threshold': self.adaptive_params['profit_protection_percentage'],
                                    'hold_duration': (datetime.now() - datetime.fromisoformat(tracker.get('entry_time', datetime.now().isoformat()))).total_seconds() / 3600,
                                    'market_conditions': {
                                        'current_time': datetime.now().isoformat(),
                                        'protection_trigger': 'PROFIT_DRAWDOWN'
                                    }
                                }
                                
                                self.daily_trade_data.append(protection_data)
                                self.weekly_trade_data.append(protection_data)
                                
                                logger.info(f"🛡️ ADAPTIVE PROFIT PROTECTION: {symbol} closed at ${profit:.2f}")
                                logger.info(f"   Peak: ${peak_profit:.2f}, Drawdown: {drawdown_percentage:.1%}")
                                
                                del self.position_trackers[ticket]
                        except Exception as e:
                            logger.error(f"Error in adaptive profit protection {ticket}: {e}")
                            
        except Exception as e:
            logger.error(f"Error in adaptive percentage protection: {e}")
            
    async def get_dynamic_atr_multiplier(self, pair: str, df_recent) -> float:
        """Calculate dynamic ATR multiplier based on current market conditions."""
        try:
            if df_recent is None or len(df_recent) < 10:
                return self.adaptive_params['atr_multiplier_normal_vol']
            
            # Calculate current volatility
            current_volatility = df_recent['close'].pct_change().std()
            
            # Get market session info for additional context
            current_hour = datetime.now().hour
            
            # Session-based adjustments
            session_multiplier = 1.0
            if 8 <= current_hour <= 10:  # London open
                session_multiplier = 1.2
            elif 13 <= current_hour <= 15:  # NY open
                session_multiplier = 1.3
            elif 20 <= current_hour <= 22:  # Asian session
                session_multiplier = 0.9
            
            # Volatility-based ATR multiplier
            if current_volatility < self.adaptive_params['volatility_threshold_low']:
                base_multiplier = self.adaptive_params['atr_multiplier_low_vol']
                vol_condition = "LOW_VOLATILITY"
            elif current_volatility > self.adaptive_params['volatility_threshold_high']:
                base_multiplier = self.adaptive_params['atr_multiplier_high_vol']
                vol_condition = "HIGH_VOLATILITY"
            else:
                base_multiplier = self.adaptive_params['atr_multiplier_normal_vol']
                vol_condition = "NORMAL_VOLATILITY"
            
            # Apply session adjustment
            final_multiplier = base_multiplier * session_multiplier
            
            logger.debug(f"   {pair}: Dynamic ATR multiplier = {final_multiplier:.2f} ({vol_condition}, session={session_multiplier:.1f})")
            
            return final_multiplier
            
        except Exception as e:
            logger.error(f"Error calculating dynamic ATR multiplier for {pair}: {e}")
            return self.adaptive_params['atr_multiplier_normal_vol']

    async def check_news_events(self, pair: str) -> bool:
        """Check for high-impact news events that could affect the pair."""
        try:
            # Skip if scheduler not available
            if not SCHEDULER_AVAILABLE or self.scheduler is None:
                logger.debug("Scheduler not available - skipping news check")
                return True

            events = await self.scheduler.get_upcoming_events(pair)
            now = datetime.now(timezone.utc)

            for event in events:
                if (event['impact'].lower() == 'high' and
                    abs((datetime.fromisoformat(event['time']) - now).total_seconds()) < 3600):
                    logger.warning(f"🚨 HIGH-IMPACT NEWS for {pair}: {event['title']}")
                    return False
            return True
        except Exception as e:
            logger.error(f"News check failed: {e}")
            return True

    def get_atr_multiplier(self, volatility: float, pair: str) -> float:
        """Get ATR multiplier based on volatility and pair."""
        if pair.lower() == 'xauusd':
            return self.adaptive_params['atr_multiplier_high_vol']
        if volatility < self.adaptive_params['volatility_threshold_low']:
            return self.adaptive_params['atr_multiplier_low_vol']
        elif volatility > self.adaptive_params['volatility_threshold_high']:
            return self.adaptive_params['atr_multiplier_high_vol']
        return self.adaptive_params['atr_multiplier_normal_vol']

    def get_volatility_level(self, volatility: float, pair: str) -> str:
        """Get volatility level classification."""
        if pair.lower() == 'xauusd':
            return 'HIGH'
        if volatility < self.adaptive_params['volatility_threshold_low']:
            return 'LOW'
        elif volatility > self.adaptive_params['volatility_threshold_high']:
            return 'HIGH'
        return 'NORMAL'
            
    async def adaptive_signal_analysis(self, pair: str, df_15m, df_1h, df_h4=None):
        """Enhanced signal analysis with adaptive parameters and multi-timeframe confirmation."""
        try:
            # Generate base signal with dynamic stops
            signal = self.technical_analyzer.generate_signal(
                df_15m, df_1h, 
                adaptive_params=self.adaptive_params,
                pair=pair,
                correlation_analyzer=self.correlation_analyzer,
                economic_calendar_filter=None  # Could be added later
            )
            
            if signal['direction'] == SignalDirection.NONE:
                return None
                
            # MULTI-TIMEFRAME CONFIRMATION - Check H4 alignment
            h4_confirmation = True
            h4_details = "No H4 data"
            
            if df_h4 is not None and len(df_h4) >= 20:
                try:
                    # Generate H4 signal for confirmation
                    h4_signal = self.technical_analyzer.generate_signal(df_h4, df_h4)  # Use H4 for both timeframes
                    
                    if h4_signal['direction'] != SignalDirection.NONE:
                        # Check if H4 direction aligns with our signal
                        if h4_signal['direction'] == signal['direction']:
                            h4_confirmation = True
                            h4_details = f"H4 ALIGNED ({h4_signal['confidence']:.1%})"
                            # Boost confidence for aligned signals
                            signal['confidence'] = min(signal['confidence'] * 1.1, 0.95)
                        else:
                            h4_confirmation = False
                            h4_details = f"H4 CONFLICT ({h4_signal['direction'].value} vs {signal['direction'].value})"
                    else:
                        # H4 is neutral - check trend direction
                        h4_sma_20 = df_h4['close'].rolling(20).mean()
                        h4_sma_50 = df_h4['close'].rolling(50).mean()
                        
                        if len(h4_sma_20) >= 2 and len(h4_sma_50) >= 2:
                            h4_trend_up = h4_sma_20.iloc[-1] > h4_sma_50.iloc[-1]
                            
                            if (signal['direction'] == SignalDirection.BUY and h4_trend_up) or \
                               (signal['direction'] == SignalDirection.SELL and not h4_trend_up):
                                h4_confirmation = True
                                h4_details = f"H4 TREND ALIGNED ({'UP' if h4_trend_up else 'DOWN'})"
                            else:
                                h4_confirmation = False
                                h4_details = f"H4 TREND CONFLICT ({'UP' if h4_trend_up else 'DOWN'} vs {signal['direction'].value})"
                        else:
                            h4_details = "H4 insufficient data for trend"
                            
                except Exception as e:
                    logger.error(f"Error in H4 confirmation for {pair}: {e}")
                    h4_details = "H4 analysis error"
            
            # Reject signals without H4 confirmation (unless very high confidence)
            if not h4_confirmation and signal['confidence'] < 0.90:
                self.signals_rejected += 1
                logger.info(f"   {pair}: Signal rejected - H4 timeframe conflict")
                logger.info(f"   H4 Status: {h4_details}")
                return None
                
            # Apply adaptive confidence threshold with market condition scaling
            confidence = signal.get('confidence', 0)

            # Get market volatility for adaptive threshold
            df_recent = await self.data_manager.get_candles(pair, "M15", 20)
            if df_recent is not None and len(df_recent) >= 10:
                recent_volatility = df_recent['close'].pct_change().std()

                # Adaptive confidence based on market conditions
                if recent_volatility > 0.003:  # High volatility
                    adaptive_threshold = 0.80  # RAISED from 0.75 for high volatility
                    condition = "HIGH_VOLATILITY"
                elif recent_volatility > 0.0015:  # Normal volatility
                    adaptive_threshold = self.adaptive_params['min_confidence']  # 0.85
                    condition = "NORMAL_VOLATILITY"
                else:  # Low volatility
                    adaptive_threshold = 0.75  # RAISED from 0.70 for low volatility
                    condition = "LOW_VOLATILITY"
            else:
                adaptive_threshold = self.adaptive_params['min_confidence']
                condition = "DEFAULT"

            if confidence < adaptive_threshold:
                self.signals_rejected += 1
                logger.info(f"   {pair}: Signal rejected - Low confidence ({confidence:.1%} < {adaptive_threshold:.1%}) [{condition}]")
                return None
                
            # Apply adaptive R/R ratio
            if signal['direction'] == SignalDirection.BUY:
                risk = abs(signal['entry_price'] - signal['stop_loss'])
                reward = abs(signal['take_profit'] - signal['entry_price'])
            else:
                risk = abs(signal['stop_loss'] - signal['entry_price'])
                reward = abs(signal['entry_price'] - signal['take_profit'])
                
            rr_ratio = reward / risk if risk > 0 else 0
            
            if rr_ratio < self.adaptive_params['min_rr_ratio']:
                self.signals_rejected += 1
                logger.info(f"   {pair}: Signal rejected - Poor R/R ({rr_ratio:.2f} < {self.adaptive_params['min_rr_ratio']:.2f})")
                return None
                
            logger.info(f"   {pair}: ADAPTIVE SIGNAL ACCEPTED!")
            logger.info(f"   Confidence: {confidence:.1%} (adaptive threshold: {adaptive_threshold:.1%}) [{condition}]")
            logger.info(f"   R/R Ratio: {rr_ratio:.2f} (threshold: {self.adaptive_params['min_rr_ratio']:.2f})")
            logger.info(f"   H4 Confirmation: {h4_details}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in adaptive signal analysis for {pair}: {e}")
            return None
            
    async def perform_daily_ml_analysis(self):
        """Perform daily ML analysis during US-Japan gap."""
        try:
            # Skip if ML engine not available
            if not ML_ENGINE_AVAILABLE or self.ml_engine is None:
                logger.debug("ML engine not available - skipping daily analysis")
                return

            logger.info("🧠 STARTING DAILY ML ANALYSIS...")

            # Prepare daily data
            daily_data = {
                'session_date': datetime.now().date().isoformat(),
                'executed_trades': list(self.daily_trade_data),  # Convert deque to list
                'signals_analyzed': self.signals_analyzed,
                'signals_rejected': self.signals_rejected,
                'reversals_detected': self.reversals_detected,
                'adaptive_parameters_used': self.adaptive_params.copy()
            }

            # Perform ML analysis
            analysis_result = self.ml_engine.perform_daily_analysis(daily_data)
            self.last_daily_analysis = analysis_result
            self.ml_analyses_performed += 1

            # ENHANCED ML FEEDBACK LOOP - Apply recommended parameter adjustments
            adjustments = analysis_result.get('strategy_adjustments', [])
            insights = analysis_result.get('ml_insights', {})

            # Apply ML-driven parameter updates
            for adjustment in adjustments:
                param = adjustment.get('parameter')
                new_value = adjustment.get('recommended_value')

                if param in self.adaptive_params:
                    old_value = self.adaptive_params[param]
                    self.adaptive_params[param] = new_value

                    logger.info(f"🔧 ADAPTIVE PARAMETER UPDATE:")
                    logger.info(f"   {param}: {old_value} → {new_value}")
                    logger.info(f"   Reason: {adjustment.get('reason')}")

            # ADDITIONAL ML-DRIVEN ADJUSTMENTS based on insights
            win_rate = insights.get('win_rate', 0.5)
            profit_factor = insights.get('profit_factor', 1.0)
            confidence_correlation = insights.get('confidence_correlation', 0)

            # Adjust ATR multipliers based on win rate performance
            if win_rate < 0.6:
                # Poor win rate - widen stops for safety
                self.adaptive_params['atr_multiplier_normal_vol'] = min(
                    self.adaptive_params['atr_multiplier_normal_vol'] + 0.5, 4.0
                )
                logger.info(f"🔧 ML ADJUSTMENT: Widening ATR multiplier due to low win rate ({win_rate:.1%})")
            elif win_rate > 0.75:
                # High win rate - can tighten stops slightly
                self.adaptive_params['atr_multiplier_normal_vol'] = max(
                    self.adaptive_params['atr_multiplier_normal_vol'] - 0.2, 2.0
                )
                logger.info(f"🔧 ML ADJUSTMENT: Tightening ATR multiplier due to high win rate ({win_rate:.1%})")

            # Adjust confidence threshold based on correlation with performance
            if confidence_correlation < 0.2:
                # Weak correlation - increase confidence requirement
                self.adaptive_params['min_confidence'] = min(
                    self.adaptive_params['min_confidence'] + 0.05, 0.95
                )
                logger.info(f"🔧 ML ADJUSTMENT: Raising confidence threshold due to weak correlation ({confidence_correlation:.2f})")
            elif confidence_correlation > 0.6:
                # Strong correlation - can slightly lower threshold
                self.adaptive_params['min_confidence'] = max(
                    self.adaptive_params['min_confidence'] - 0.02, 0.75
                )
                logger.info(f"🔧 ML ADJUSTMENT: Lowering confidence threshold due to strong correlation ({confidence_correlation:.2f})")

            # Adjust profit protection based on profit factor
            if profit_factor < 1.2:
                # Poor profit factor - tighten protection
                self.adaptive_params['profit_protection_percentage'] = max(
                    self.adaptive_params['profit_protection_percentage'] - 0.05, 0.15
                )
                logger.info(f"🔧 ML ADJUSTMENT: Tightening profit protection due to low profit factor ({profit_factor:.2f})")
            elif profit_factor > 2.0:
                # Excellent profit factor - can relax protection slightly
                self.adaptive_params['profit_protection_percentage'] = min(
                    self.adaptive_params['profit_protection_percentage'] + 0.02, 0.25
                )
                logger.info(f"🔧 ML ADJUSTMENT: Relaxing profit protection due to high profit factor ({profit_factor:.2f})")

            # Reset daily data
            self.daily_trade_data.clear()

            logger.info("✅ DAILY ML ANALYSIS COMPLETE")

        except Exception as e:
            logger.error(f"Error in daily ML analysis: {e}")
            
    async def perform_weekend_ml_analysis(self):
        """Perform comprehensive weekend ML analysis."""
        try:
            logger.info("🧠 STARTING WEEKEND ML ANALYSIS...")
            
            # Prepare weekly data
            weekly_data = {
                'week_period': f"{datetime.now().date() - timedelta(days=7)} to {datetime.now().date()}",
                'weekly_trades': self.weekly_trade_data,
                'daily_analyses': [self.last_daily_analysis] if self.last_daily_analysis else [],
                'adaptive_parameter_evolution': self.adaptive_params.copy()
            }
            
            # Perform comprehensive weekly analysis
            analysis_result = self.ml_engine.perform_weekly_analysis(weekly_data)
            self.last_weekly_analysis = analysis_result
            self.ml_analyses_performed += 1
            
            # Apply major strategy optimizations
            optimizations = analysis_result.get('next_week_strategy', {})
            optimized_params = optimizations.get('optimized_parameters', {})
            
            for param, value in optimized_params.items():
                if param in self.adaptive_params:
                    old_value = self.adaptive_params[param]
                    self.adaptive_params[param] = value
                    
                    logger.info(f"🚀 WEEKLY OPTIMIZATION:")
                    logger.info(f"   {param}: {old_value} → {value}")
                    
            # Reset weekly data
            self.weekly_trade_data = []
            
            logger.info("✅ WEEKEND ML ANALYSIS COMPLETE")
            
        except Exception as e:
            logger.error(f"Error in weekend ML analysis: {e}")

    async def generate_attribution_report(self, start_date: Optional[str] = None,
                                        end_date: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive trade attribution report."""
        try:
            # Skip if attribution analyzer not available
            if not ATTRIBUTION_AVAILABLE or self.attribution_analyzer is None:
                logger.debug("Attribution analyzer not available - skipping report generation")
                return {'error': 'Attribution analyzer not available'}

            logger.info("📊 GENERATING TRADE ATTRIBUTION REPORT...")

            # Generate attribution analysis
            report = self.attribution_analyzer.generate_attribution_report(start_date, end_date)

            # Add additional insights
            if self.attribution_analyzer.trade_history:
                report['insights'] = {
                    'top_strategies': self.attribution_analyzer.get_top_performing_strategies(5),
                    'worst_strategies': self.attribution_analyzer.get_worst_performing_strategies(5),
                    'seasonal_performance': self.attribution_analyzer.analyze_seasonal_performance(),
                    'confidence_intervals': self.attribution_analyzer.calculate_attribution_confidence()
                }

                # Add strategy recommendations
                top_strategies = report['insights']['top_strategies']
                if top_strategies:
                    best_strategy = top_strategies[0][0]
                    report['recommendations'] = {
                        'best_performing_strategy': best_strategy,
                        'suggested_focus': f"Increase allocation to {best_strategy} strategy",
                        'risk_adjusted_insights': "Consider strategies with Sharpe ratios > 1.0"
                    }

            # Save report to file
            report_file = logs_dir / f"attribution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"✅ ATTRIBUTION REPORT GENERATED: {report_file}")
            return report

        except Exception as e:
            logger.error(f"Error generating attribution report: {e}")
            return {'error': str(e)}
            
    async def get_rl_parameter_adjustments(self) -> Dict:
        """Get parameter adjustments from trained RL model."""
        try:
            # Skip if ML engine not available
            if not ML_ENGINE_AVAILABLE or self.ml_engine is None:
                logger.debug("ML engine not available - skipping RL adjustments")
                return {}

            # Get current market state for RL
            current_state = await self._get_current_market_state()
            if current_state is None:
                return {}

            # Get RL action
            rl_action = self.ml_engine.get_rl_action(current_state)
            if rl_action is None:
                return {}

            # Convert RL action to parameter adjustments
            adjustments = {
                'min_confidence': self.adaptive_params['min_confidence'] + rl_action[0] * 0.05,  # ±5% adjustment
                'min_rr_ratio': self.adaptive_params['min_rr_ratio'] + rl_action[1] * 0.5,      # ±0.5 adjustment
                'atr_multiplier_normal_vol': self.adaptive_params['atr_multiplier_normal_vol'] + rl_action[2] * 0.2  # ±0.2 adjustment
            }

            # Clamp adjustments to reasonable ranges
            adjustments['min_confidence'] = max(0.75, min(0.95, adjustments['min_confidence']))
            adjustments['min_rr_ratio'] = max(2.5, min(4.5, adjustments['min_rr_ratio']))
            adjustments['atr_multiplier_normal_vol'] = max(2.0, min(4.0, adjustments['atr_multiplier_normal_vol']))

            logger.info("🎯 RL Parameter Adjustments Applied:")
            for param, value in adjustments.items():
                logger.info(f"   {param}: {self.adaptive_params[param]:.3f} → {value:.3f}")

            return adjustments

        except Exception as e:
            logger.error(f"Error getting RL adjustments: {e}")
            return {}

    async def _get_current_market_state(self) -> Optional[np.ndarray]:
        """Get current market state for RL model."""
        try:
            # Get average market conditions across all pairs
            pairs = settings.get_currency_pairs()
            rsi_values = []
            volatility_values = []
            correlation_values = []
            sentiment_values = []

            for pair in pairs[:3]:  # Use top 3 pairs for state calculation
                try:
                    df = await self.data_manager.get_candles(pair, "M15", 20)
                    if df is not None and len(df) >= 10:
                        # Calculate RSI (simplified)
                        rsi = 50  # Default
                        if 'close' in df.columns:
                            price_changes = df['close'].pct_change()
                            gains = price_changes.where(price_changes > 0, 0).rolling(14).mean()
                            losses = (-price_changes.where(price_changes < 0, 0)).rolling(14).mean()
                            if len(gains) > 0 and len(losses) > 0 and losses.iloc[-1] != 0:
                                rs = gains.iloc[-1] / losses.iloc[-1]
                                rsi = 100 - (100 / (1 + rs))
                        rsi_values.append(rsi / 100)  # Normalize to 0-1

                        # Calculate volatility
                        if len(df) >= 5:
                            vol = df['close'].pct_change().std()
                            volatility_values.append(min(vol * 100, 0.02))  # Cap at 2%

                except Exception as e:
                    logger.debug(f"Error getting state for {pair}: {e}")
                    continue

            # Calculate average correlation (simplified)
            if len(pairs) >= 2:
                try:
                    await self.correlation_analyzer.update_correlation_matrix(pairs)
                    corr_matrix = self.correlation_analyzer.correlation_matrix
                    avg_corr = corr_matrix.mean().mean() if hasattr(corr_matrix, 'mean') else 0.0
                    correlation_values.append(avg_corr)
                except:
                    correlation_values.append(0.0)

            # Default sentiment (can be enhanced later)
            sentiment_values.append(0.0)

            # Aggregate to single state vector
            if rsi_values and volatility_values:
                state = np.array([
                    np.mean(rsi_values),           # Average RSI (0-1)
                    np.mean(volatility_values),     # Average volatility
                    self.adaptive_params['min_confidence'] / 0.95,  # Normalized confidence
                    np.mean(correlation_values) if correlation_values else 0.0,  # Average correlation
                    np.mean(sentiment_values)       # Average sentiment
                ])

                # Normalize to [-2, 2] range as expected by RL model
                state = np.clip(state * 4 - 2, -2, 2)
                return state

            return None

        except Exception as e:
            logger.error(f"Error calculating market state: {e}")
            return None

    async def adaptive_trading_scan(self):
        """Enhanced trading scan with adaptive parameters and real-time monitoring."""
        scan_start_time = time.time()
        try:
            logger.info(f"🤖 ADAPTIVE SCAN #{self.scan_count + 1}")

            # Perform health check only if metrics_collector is available
            if self.metrics_collector is not None:
                health_status = await self.metrics_collector.perform_health_check()
                if health_status.get('overall_health', 100) < 50:
                    logger.warning(f"⚠️ SYSTEM HEALTH WARNING: {health_status.get('overall_health', 0)}%")
                    for alert in health_status.get('alerts', []):
                        logger.warning(f"   {alert['type']}: {alert['message']}")
            else:
                logger.info("⚠️ Metrics collector not available - skipping health check")

            # Get RL-based parameter adjustments
            rl_adjustments = await self.get_rl_parameter_adjustments()
            if rl_adjustments:
                # Apply RL adjustments to adaptive parameters
                for param, value in rl_adjustments.items():
                    self.adaptive_params[param] = value

            # Real-time reversal monitoring
            await self.real_time_reversal_monitoring()

            # Adaptive protection
            await self.adaptive_percentage_protection()

            # PARALLEL PROCESSING - Process multiple currency pairs concurrently
            pairs = settings.get_currency_pairs()
            logger.info(f"🔄 Processing {len(pairs)} currency pairs with parallel processing")

            # Create parallel tasks for data fetching
            async def fetch_pair_data(pair: str) -> Dict:
                """Fetch all required data for a currency pair in parallel."""
                try:
                    # Parallel data fetching
                    df_15m_task = asyncio.create_task(self.data_manager.get_candles(pair, "M15", 100))
                    df_1h_task = asyncio.create_task(self.data_manager.get_candles(pair, "H1", 100))
                    df_h4_task = asyncio.create_task(self.data_manager.get_candles(pair, "H4", 50))
                    spread_task = asyncio.create_task(self.broker_manager.get_spread_pips(pair))
                    async def empty_sentiment():
                        return {}
                    sentiment_task = self.sentiment_aggregator.get_overall_sentiment(pair.replace('_', '')) if SENTIMENT_AVAILABLE and self.sentiment_aggregator else empty_sentiment()

                    # Execute all tasks concurrently
                    df_15m, df_1h, df_h4, spread_pips, sentiment_data = await asyncio.gather(
                        df_15m_task, df_1h_task, df_h4_task, spread_task, sentiment_task,
                        return_exceptions=True
                    )

                    # Handle exceptions
                    if isinstance(df_15m, Exception) or df_15m is None:
                        logger.warning(f"Failed to fetch 15M data for {pair}")
                        return None
                    if isinstance(df_1h, Exception) or df_1h is None:
                        logger.warning(f"Failed to fetch 1H data for {pair}")
                        return None

                    return {
                        'pair': pair,
                        'df_15m': df_15m,
                        'df_1h': df_1h,
                        'df_h4': df_h4 if not isinstance(df_h4, Exception) else None,
                        'spread_pips': spread_pips if not isinstance(spread_pips, Exception) else None,
                        'sentiment_data': sentiment_data if not isinstance(sentiment_data, Exception) else {}
                    }

                except Exception as e:
                    logger.error(f"Error fetching data for {pair}: {e}")
                    return None

            # Fetch data for all pairs in parallel
            data_tasks = [fetch_pair_data(pair) for pair in pairs]
            pair_data_results = await asyncio.gather(*data_tasks, return_exceptions=True)

            # Filter successful results
            valid_pair_data = [data for data in pair_data_results if data is not None and not isinstance(data, Exception)]

            logger.info(f"✅ Successfully fetched data for {len(valid_pair_data)}/{len(pairs)} pairs")

            # Get shared data once (not per pair)
            positions = await self.broker_manager.get_positions()
            account_info = await self.broker_manager.get_account_info()
            mpt_weights = self.correlation_analyzer.get_mpt_weights(None)
            portfolio_metrics = self.correlation_analyzer.calculate_portfolio_metrics(positions)
            await self.correlation_analyzer.update_correlation_matrix(pairs)

            # Update monitoring metrics only if metrics_collector is available
            if self.metrics_collector is not None:
                self.metrics_collector.update_account_metrics(account_info)
                self.metrics_collector.update_position_metrics(positions, sum(abs(pos.get('profit', 0)) for pos in positions if pos.get('profit', 0) < 0))

            # Process each pair with parallel signal analysis
            async def process_pair_trading_logic(pair_data: Dict) -> None:
                """Process trading logic for a single pair."""
                try:
                    pair = pair_data['pair']
                    df_15m = pair_data['df_15m']
                    df_1h = pair_data['df_1h']
                    df_h4 = pair_data['df_h4']
                    spread_data = pair_data['spread_pips']
                    spread_pips = spread_data.get('spread_pips') if spread_data and isinstance(spread_data, dict) else None
                    sentiment_data = pair_data['sentiment_data']

                    self.signals_analyzed += 1

                    # Validate data quality
                    if len(df_15m) < 50:
                        logger.debug(f"Insufficient data for {pair}, skipping")
                        return

                    # Parallel signal analysis components
                    signal_task = self.adaptive_signal_analysis(pair, df_15m, df_1h, df_h4)
                    news_check_task = self.check_news_events(pair)

                    signal, news_ok = await asyncio.gather(signal_task, news_check_task, return_exceptions=True)

                    if isinstance(signal, Exception) or signal is None:
                        logger.debug(f"   {pair}: No valid signal generated")
                        return

                    # News filter
                    if isinstance(news_ok, Exception) or not news_ok:
                        logger.info(f"   {pair}: Trade skipped - High-impact news event detected")
                        self.signals_rejected += 1
                        return

                    # Pre-trade validation
                    open_positions = len(positions)
                    balance = account_info['balance']
                    total_risk = sum(abs(pos.get('profit', 0) / balance) for pos in positions if pos.get('profit', 0) < 0)

                    # Risk checks
                    if total_risk > 0.05:  # 5% total risk limit
                        logger.warning(f"   {pair}: Total open risk {total_risk:.2%} exceeds 5%; skipping new trades")
                        return

                    # Currency exposure check
                    exposure = self.correlation_analyzer.get_currency_exposure(positions)
                    base, quote = pair[:3], pair[3:]

                    if abs(exposure.get(base, 0)) > 2.0 or abs(exposure.get(quote, 0)) > 2.0:
                        logger.warning(f"   {pair}: Excessive exposure to {base}/{quote} - skipping trade")
                        self.signals_rejected += 1
                        return

                    # Initialize volume scaling
                    volume_scale = 1.0

                    # Sentiment-based volume scaling
                    sentiment_score = sentiment_data.get('overall_sentiment', 0.0)
                    sentiment_confidence = sentiment_data.get('overall_confidence', 0.0)

                    if sentiment_confidence > 0.3:
                        if sentiment_score < -0.3:  # Very negative sentiment
                            logger.warning(f"   {pair}: Very negative sentiment ({sentiment_score:.2f}) - reducing volume")
                            volume_scale *= 0.3
                        elif sentiment_score < -0.1:  # Moderately negative
                            logger.warning(f"   {pair}: Negative sentiment ({sentiment_score:.2f}) - reducing volume")
                            volume_scale *= 0.6
                        elif sentiment_score > 0.3:  # Positive sentiment
                            logger.info(f"   {pair}: Positive sentiment ({sentiment_score:.2f}) - maintaining volume")
                        elif sentiment_score > 0.1:  # Moderately positive
                            logger.info(f"   {pair}: Moderately positive sentiment ({sentiment_score:.2f})")
                            volume_scale *= 1.1

                    logger.info(f"   {pair}: Sentiment {sentiment_score:.2f} (confidence: {sentiment_confidence:.2f})")

                    # MPT portfolio optimization
                    mpt_weight = mpt_weights.get(pair, 0.1)
                    diversification_score = portfolio_metrics.get('metrics', {}).get('diversification_score', 0.5)
                    mpt_volume_scale = mpt_weight * (1.0 + diversification_score)
                    volume_scale *= min(mpt_volume_scale, 1.5)

                    logger.info(f"   {pair}: MPT weight {mpt_weight:.3f}, diversification {diversification_score:.2f}")

                    # Correlation analysis
                    hedge_info = self.correlation_analyzer.should_hedge_position(positions, pair, signal['direction'].value)

                    if hedge_info['should_hedge']:
                        logger.warning(f"   {pair}: High correlation detected with existing positions")
                        volume_scale *= 0.4
                    elif hedge_info.get('correlation_risk', 0) > 0.6:
                        logger.info(f"   {pair}: Moderate correlation risk, reducing volume")
                        volume_scale *= 0.6

                    # Trading gates
                    max_positions_soft = 8
                    max_spread_soft = 20
                    gate_reasons = []

                    if spread_pips is not None and spread_pips > max_spread_soft:
                        gate_reasons.append(f"spread {spread_pips}p > {max_spread_soft}p")
                    elif spread_pips is not None and spread_pips > 15:
                        volume_scale *= 0.7

                    if open_positions >= max_positions_soft:
                        gate_reasons.append(f"positions {open_positions}/{max_positions_soft}")
                    elif open_positions >= 5:
                        volume_scale *= 0.8

                    if gate_reasons:
                        logger.info(f"   {pair}: Trade skipped by gate - {', '.join(gate_reasons)}")
                        return

                    # Execute trade
                    logger.info(f"🚀 EXECUTING ADAPTIVE TRADE: {pair}")

                    # Risk-based position sizing
                    sl_pips = abs(signal['entry_price'] - signal['stop_loss']) * 10000
                    pip_value = 10  # $10/pip for 1 lot on majors
                    risk_amount = balance * 0.01  # 1% risk per trade
                    volume = round((risk_amount / (sl_pips * pip_value)), 2)

                    # Confidence and volume adjustments
                    confidence = signal.get('confidence', 0)
                    if confidence >= 0.9:
                        volume = min(volume * 1.2, volume)
                    elif confidence < 0.8:
                        volume *= 0.8

                    volume *= volume_scale
                    volume = max(0.01, min(volume, 1.0))

                    # Final risk check
                    current_positions = await self.broker_manager.get_positions()
                    total_risk = sum([abs(pos.get('profit', 0)) for pos in current_positions if pos.get('profit', 0) < 0])
                    if total_risk > balance * 0.05:
                        logger.info(f"   {pair}: Total portfolio risk exceeded, reducing volume")
                        volume *= 0.5

                    if volume < 0.01:
                        logger.info(f"   {pair}: Volume below broker minimum after risk calculation, skipping")
                        return

                    order_result = await self.broker_manager.place_order(
                        symbol=pair,
                        order_type=signal['direction'].value,
                        volume=round(volume, 2),
                        sl=signal['stop_loss'],
                        tp=signal['take_profit']
                    )

                    if order_result and order_result.get('ticket'):
                        self.trades_executed += 1

                        # Record trade metrics only if metrics_collector is available
                        if self.metrics_collector is not None:
                            self.metrics_collector.record_trade(signal, round(volume, 2), sentiment_data)

                        # Record trade for attribution analysis
                        trade_data = {
                            'timestamp': datetime.now().isoformat(),
                            'ticket': order_result['ticket'],
                            'symbol': pair,
                            'direction': signal['direction'].value,
                            'entry_price': signal['entry_price'],
                            'stop_loss': signal['stop_loss'],
                            'take_profit': signal['take_profit'],
                            'volume': round(volume, 2),
                            'confidence': confidence,
                            'adaptive_params_used': self.adaptive_params.copy(),
                            'spread_pips': spread_pips,
                            'open_positions': open_positions,
                            'exit_reason': 'entry',  # Will be updated when trade closes
                            'profit': 0,  # Will be updated when trade closes
                            'hold_duration': 0,  # Will be updated when trade closes
                            'market_conditions': {
                                'volatility_15m': df_15m['close'].pct_change().std() if len(df_15m) > 1 else 0,
                                'volatility_1h': df_1h['close'].pct_change().std() if len(df_1h) > 1 else 0,
                                'price_at_entry': df_15m['close'].iloc[-1] if len(df_15m) > 0 else 0
                            },
                            'sentiment_data': sentiment_data,
                            'correlation_data': {},
                            'volatility_level': self.get_volatility_level(
                                df_15m['close'].pct_change().std() if len(df_15m) > 1 else 0, pair
                            ),
                            'session': 'normal'  # Could be enhanced with session detection
                        }

                        # Add trade to attribution analyzer if available
                        if self.attribution_analyzer is not None:
                            self.attribution_analyzer.add_trade(trade_data)

                        self.daily_trade_data.append(trade_data)
                        self.weekly_trade_data.append(trade_data)

                        logger.info(f"✅ ADAPTIVE TRADE EXECUTED: {pair} #{order_result['ticket']}")

                except Exception as e:
                    logger.error(f"Error processing pair {pair_data.get('pair', 'unknown')}: {e}")

            # Process all pairs in parallel
            processing_tasks = [process_pair_trading_logic(pair_data) for pair_data in valid_pair_data]
            await asyncio.gather(*processing_tasks, return_exceptions=True)

            logger.info(f"🔄 Parallel processing completed for {len(valid_pair_data)} pairs")

            # Record scan performance only if metrics_collector is available
            scan_duration = time.time() - scan_start_time
            if self.metrics_collector is not None:
                self.metrics_collector.record_scan_performance(
                    scan_duration=scan_duration,
                    pairs_processed=len(valid_pair_data),
                    signals_generated=self.signals_analyzed - self.signals_rejected,
                    trades_executed=self.trades_executed
                )

            self.scan_count += 1

        except Exception as e:
            logger.error(f"Error in adaptive trading scan: {e}")
            if self.metrics_collector:
                self.metrics_collector.increment_error_count("scan_error")
            
    async def run_adaptive_intelligent_bot(self):
        """Main adaptive intelligent bot loop."""
        logger.info("🧠 STARTING ADAPTIVE INTELLIGENT TRADING BOT")
        logger.info("="*80)
        logger.info("🚀 ADAPTIVE FEATURES:")
        logger.info("   - Real-time trend reversal detection")
        logger.info("   - ML-driven daily analysis (US-Japan gap)")
        logger.info("   - Comprehensive weekend learning")
        logger.info("   - Adaptive parameter optimization")
        logger.info("   - Intelligent scheduling")
        logger.info("   - Enhanced profit protection")
        logger.info("="*80)
        print("[RUN] Main bot loop starting...")
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        while self.running:
            print(f"[RUN] Bot loop iteration {self.scan_count + 1}...")
            try:
                await self.adaptive_trading_scan()
                print(f"[RUN] Completed trading scan #{self.scan_count + 1}")
                self.scan_count += 1
            except Exception as e:
                print(f"[ERROR] Exception in main loop: {e}")
            await asyncio.sleep(1)  # Prevent tight loop
        
        try:
            # Initialize core components (always available)
            await self.data_manager.initialize()
            await self.broker_manager.initialize()

            # Initialize advanced components lazily (only if available)
            if METRICS_AVAILABLE and self.metrics_collector:
                await self.metrics_collector.start()
                logger.info("✅ Metrics collector initialized")
            else:
                logger.info("⚠️ Metrics collector not available - monitoring disabled")

            if SCHEDULER_AVAILABLE and self.scheduler:
                self.scheduler.register_mode_change_callback(self.mode_change_handler)
                scheduler_task = asyncio.create_task(self.scheduler.monitor_mode_changes())
                logger.info("✅ Intelligent scheduler initialized")
            else:
                logger.info("⚠️ Intelligent scheduler not available - basic mode switching disabled")
                scheduler_task = None

            # Log component availability
            logger.info("🔧 COMPONENT STATUS:")
            logger.info(f"   Reversal Detector: {'✅' if REVERSAL_DETECTOR_AVAILABLE and self.reversal_detector else '❌'}")
            logger.info(f"   ML Engine: {'✅' if ML_ENGINE_AVAILABLE and self.ml_engine else '❌'}")
            logger.info(f"   Trade Analyzer: {'✅' if TRADE_ANALYZER_AVAILABLE and self.trade_analyzer else '❌'}")
            logger.info(f"   Intelligent Scheduler: {'✅' if SCHEDULER_AVAILABLE and self.scheduler else '❌'}")
            logger.info(f"   Sentiment Aggregator: {'✅' if SENTIMENT_AVAILABLE and self.sentiment_aggregator else '❌'}")
            logger.info(f"   Metrics Collector: {'✅' if METRICS_AVAILABLE and self.metrics_collector else '❌'}")
            logger.info(f"   Attribution Analyzer: {'✅' if ATTRIBUTION_AVAILABLE and self.attribution_analyzer else '❌'}")

            logger.info("✅ Adaptive intelligent bot initialized with Phase 1 improvements")
            
            # Main adaptive loop
            while self.running:
                try:
                    # Write heartbeat
                    await self.write_heartbeat()
                    
                    # Check current mode
                    if await self.scheduler.should_execute_trades(self.broker_manager):
                        await self.adaptive_trading_scan()
                        
                        # Status update every 3 scans
                        if self.scan_count % 3 == 0:
                            account_info = await self.broker_manager.get_account_info()
                            positions = await self.broker_manager.get_positions()
                            
                            logger.info(f"🤖 ADAPTIVE BOT STATUS:")
                            logger.info(f"   Mode: {self.current_mode}")
                            logger.info(f"   Scans: {self.scan_count}")
                            logger.info(f"   Trades: {self.trades_executed}")
                            logger.info(f"   Reversals Detected: {self.reversals_detected}")
                            logger.info(f"   ML Analyses: {self.ml_analyses_performed}")
                            logger.info(f"   Balance: ${account_info['balance']:,.2f}")
                            logger.info(f"   Positions: {len(positions)}")
                            logger.info(f"   Adaptive Confidence: {self.adaptive_params['min_confidence']:.1%}")
                            
                        # Wait 2 minutes
                        for i in range(12):
                            if not self.running:
                                break
                            await asyncio.sleep(10)
                            if i % 3 == 0:
                                await self.write_heartbeat()
                except Exception as e:
                    logger.error(f"Error in main loop iteration: {e}")
                    await asyncio.sleep(60)  # Wait before retrying
                    
        except Exception as e:
            logger.error(f"Critical error in bot execution: {e}")
            traceback.print_exc()
        finally:
            logger.info("🛑 Adaptive intelligent bot shutting down...")
            await self.write_heartbeat()

if __name__ == "__main__":
    bot = AdaptiveIntelligentBot()
    asyncio.run(bot.run_adaptive_intelligent_bot())
