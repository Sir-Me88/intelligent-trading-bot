#!/usr/bin/env python3
"""Core Trading Bot - Full functionality without problematic ML dependencies."""

import os
import sys
import asyncio
import logging
import signal
import traceback
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import json

# Add src to path
sys.path.append('src')

from src.config.settings import settings
from src.data.market_data import MarketDataManager
from src.trading.broker_interface import BrokerManager
from src.analysis.technical import TechnicalAnalyzer, SignalDirection
from src.analysis.correlation import CorrelationAnalyzer
from src.scheduling.intelligent_scheduler import IntelligentTradingScheduler
from src.monitoring.metrics import MetricsCollector
from src.utils.pip_calculator import PipCalculator

# Setup logging
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / "core_trading_bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class CoreTradingBot:
    """Core trading bot with all essential features - no problematic ML dependencies."""
    
    def __init__(self):
        logger.info("🚀 INITIALIZING CORE TRADING BOT")
        
        # Core components
        self.data_manager = MarketDataManager()
        self.broker_manager = BrokerManager()
        self.technical_analyzer = TechnicalAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer(self.data_manager)
        self.scheduler = IntelligentTradingScheduler()
        self.metrics_collector = MetricsCollector()
        
        # Trading parameters - BALANCED FOR QUALITY TRADING
        self.trading_params = {
            'min_confidence': 0.78,  # Set to 78% as suggested by analysis
            'min_rr_ratio': 1.5,     # Set to 1.5 as suggested by analysis
            'profit_protection_percentage': 0.25,
            'max_volatility': 0.002,  # Back to conservative volatility limit
            'minimum_profit_to_protect': 20.0,
            'atr_multiplier_low_vol': 2.0,
            'atr_multiplier_normal_vol': 2.5,
            'atr_multiplier_high_vol': 3.0,
            'volatility_threshold_low': 0.001,
            'volatility_threshold_high': 0.003
        }
        
        # Trading state
        self.running = True
        self.current_mode = 'TRADING'
        self.position_trackers = {}
        
        # Performance tracking
        self.heartbeat_file = logs_dir / "core_bot_heartbeat.json"
        self.scan_count = 0
        self.trades_executed = 0
        self.signals_analyzed = 0
        self.signals_rejected = 0
        self.positions_closed_profit = 0
        self.positions_closed_loss = 0
        
        # Currency pairs to trade
        self.currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
        
    def signal_handler(self, signum: int, frame: Optional[Any]) -> None:
        """Handle system signals gracefully."""
        logger.info(f"🛑 SIGNAL RECEIVED: {signum} - Shutting down gracefully...")
        self.running = False
        
    async def write_heartbeat(self):
        """Write heartbeat with current status."""
        try:
            heartbeat_data = {
                'timestamp': datetime.now().isoformat(),
                'mode': self.current_mode,
                'scan_count': self.scan_count,
                'trades_executed': self.trades_executed,
                'signals_analyzed': self.signals_analyzed,
                'signals_rejected': self.signals_rejected,
                'positions_closed_profit': self.positions_closed_profit,
                'positions_closed_loss': self.positions_closed_loss,
                'trading_parameters': self.trading_params,
                'running': self.running,
                'active_pairs': self.currency_pairs
            }
            
            with open(self.heartbeat_file, 'w') as f:
                json.dump(heartbeat_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error writing heartbeat: {e}")
            
    async def initialize_systems(self) -> bool:
        """Initialize all trading systems."""
        try:
            logger.info("🔧 INITIALIZING TRADING SYSTEMS")
            
            # Initialize broker connection
            logger.info("🔌 Connecting to MT5...")
            broker_connected = await self.broker_manager.initialize()
            
            if not broker_connected:
                logger.error("❌ Failed to connect to MT5 broker")
                return False
            
            logger.info("✅ MT5 broker connected successfully")
            
            # Get account info
            account_info = await self.broker_manager.get_account_info()
            if account_info:
                logger.info(f"💰 Account Balance: ${account_info.get('balance', 0):,.2f}")
                logger.info(f"📈 Account Equity: ${account_info.get('equity', 0):,.2f}")
            
            # Initialize scheduler
            logger.info("📅 Initializing intelligent scheduler...")
            try:
                await self.scheduler.initialize()
                logger.info("✅ Scheduler initialized")
            except Exception as e:
                logger.warning(f"⚠️ Scheduler initialization failed: {e}")
                logger.info("📅 Continuing without advanced scheduling")
            
            logger.info("🎯 ALL SYSTEMS INITIALIZED - READY FOR TRADING!")
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            return False
    
    async def analyze_market_and_trade(self, pair: str) -> None:
        """Analyze market for a specific pair and execute trades if conditions are met."""
        try:
            logger.debug(f"📊 Analyzing {pair}...")

            # Parallel data fetch
            df_15m, df_1h = await asyncio.gather(
                self.data_manager.get_candles(pair, "M15", 100),
                self.data_manager.get_candles(pair, "H1", 50)
            )

            if df_15m is None or df_1h is None or len(df_15m) < 20 or len(df_1h) < 20:
                logger.debug(f"⚠️ Insufficient data for {pair}")
                return

            # Check news conflicts
            news_check = await self.scheduler.check_news_conflicts(pair)
            if news_check['should_skip']:
                logger.debug(f"📰 {pair}: News conflict - {news_check['reason']}")
                return

            # Generate trading signal
            signal = self.technical_analyzer.generate_signal(df_15m, df_1h)
            self.signals_analyzed += 1

            if signal['direction'] == SignalDirection.NONE:
                logger.debug(f"📊 {pair}: No signal")
                return

            # Check signal confidence
            confidence = signal.get('confidence', 0)
            if confidence < self.trading_params['min_confidence']:
                self.signals_rejected += 1
                logger.debug(f"📊 {pair}: Signal rejected - Low confidence ({confidence:.1%})")
                return

            # Check risk/reward ratio
            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', 0)
            take_profit = signal.get('take_profit', 0)

            if entry_price > 0 and stop_loss > 0 and take_profit > 0:
                if signal['direction'] == SignalDirection.BUY:
                    risk = abs(entry_price - stop_loss)
                    reward = abs(take_profit - entry_price)
                else:
                    risk = abs(stop_loss - entry_price)
                    reward = abs(entry_price - take_profit)

                rr_ratio = reward / risk if risk > 0 else 0

                if rr_ratio < self.trading_params['min_rr_ratio']:
                    self.signals_rejected += 1
                    logger.debug(f"📊 {pair}: Signal rejected - Poor R/R ratio ({rr_ratio:.1f})")
                    return

            # Validate spread
            spread_valid = await self.broker_manager.validate_spread(pair)
            if not spread_valid:
                self.signals_rejected += 1
                logger.debug(f"📊 {pair}: Signal rejected - High spread")
                return

            # Check correlation hedging
            positions = await self.broker_manager.get_positions()
            hedge_check = self.correlation_analyzer.should_hedge_position(positions, {'pair': pair, 'direction': signal['direction']})
            if hedge_check['should_hedge']:
                logger.debug(f"🔗 {pair}: Correlation hedge recommended - {hedge_check['correlation_risk']:.2f} risk")
                # For now, skip if high correlation risk
                if hedge_check['correlation_risk'] > 0.7:
                    self.signals_rejected += 1
                    logger.debug(f"📊 {pair}: Signal rejected - High correlation risk")
                    return

            # Log high-quality signal
            logger.info(f"🎯 HIGH-QUALITY SIGNAL: {pair}")
            logger.info(f"   Direction: {signal['direction']}")
            logger.info(f"   Confidence: {confidence:.1%}")
            logger.info(f"   Entry: {entry_price:.5f}")
            logger.info(f"   Stop Loss: {stop_loss:.5f}")
            logger.info(f"   Take Profit: {take_profit:.5f}")
            logger.info(f"   R/R Ratio: {rr_ratio:.1f}")

            # Execute trade (in demo mode, this is safe)
            await self.execute_trade(pair, signal)

        except Exception as e:
            logger.error(f"Error analyzing {pair}: {e}")
    
    async def execute_trade(self, pair: str, signal: Dict) -> None:
        """Execute a trade based on the signal."""
        try:
            # Get account info for position sizing
            account_info = await self.broker_manager.get_account_info()
            if not account_info:
                logger.error("❌ Cannot get account info for trade execution")
                return

            account_equity = account_info.get('equity', 0)
            if account_equity <= 0:
                logger.error("❌ Invalid account equity for trade execution")
                return

            # Use settings risk per trade
            risk_amount = account_equity * settings.trading.risk_per_trade

            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', 0)

            if entry_price > 0 and stop_loss > 0:
                # Use PipCalculator for precise position sizing
                position_size = PipCalculator.calculate_position_size(
                    pair, risk_amount, entry_price, stop_loss
                )

                # Validate position size
                validation = PipCalculator.validate_position_size(
                    pair, position_size, entry_price, stop_loss,
                    settings.trading.max_total_risk
                )

                if not validation['valid']:
                    logger.warning(f"⚠️ {pair}: Position size validation failed - {validation}")
                    return

                risk_pips = validation['risk_pips']
                trade_risk = validation['risk_amount']

            else:
                position_size = 0.01
                risk_pips = 0
                trade_risk = 0

            # Check total risk cap
            current_positions = await self.broker_manager.get_positions()
            total_current_risk = sum(
                self._calculate_position_risk(pos, account_equity)
                for pos in current_positions
            )

            if total_current_risk + trade_risk > account_equity * settings.trading.max_total_risk:
                logger.warning(f"⚠️ {pair}: Trade would exceed total risk limit, skipping")
                return

            # Place the order
            order_type = 'BUY' if signal['direction'] == SignalDirection.BUY else 'SELL'

            logger.info(f"🚀 EXECUTING TRADE: {order_type} {pair}")
            logger.info(f"   Position Size: {position_size:.2f} lots")
            logger.info(f"   Risk Amount: ${risk_amount:.2f}")
            logger.info(f"   Risk Pips: {risk_pips:.1f}")
            logger.info(f"   Total Risk: ${(total_current_risk + trade_risk):.2f} / ${(account_equity * settings.trading.max_total_risk):.2f}")

            result = await self.broker_manager.place_order(
                symbol=pair,
                order_type=order_type,
                volume=position_size,
                sl=stop_loss,
                tp=signal.get('take_profit', 0)
            )

            if result.get('status') == 'SUCCESS':
                self.trades_executed += 1
                ticket = result.get('ticket')

                logger.info(f"✅ TRADE EXECUTED SUCCESSFULLY!")
                logger.info(f"   Ticket: {ticket}")
                logger.info(f"   Pair: {pair}")
                logger.info(f"   Type: {order_type}")
                logger.info(f"   Size: {position_size:.2f} lots")

                # Track position
                self.position_trackers[ticket] = {
                    'symbol': pair,
                    'type': order_type,
                    'entry_time': datetime.now().isoformat(),
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': signal.get('take_profit', 0),
                    'volume': position_size,
                    'risk_amount': trade_risk
                }

            else:
                logger.error(f"❌ TRADE EXECUTION FAILED: {result.get('comment', 'Unknown error')}")

        except Exception as e:
            logger.error(f"Error executing trade for {pair}: {e}")

    def _get_pip_value(self, pair: str) -> float:
        """Get pip value for pair (0.01 for JPY, 0.0001 for others)."""
        return 0.01 if 'JPY' in pair else 0.0001

    def _get_tick_value(self, pair: str) -> float:
        """Get tick value for pair (1000 for JPY, 100000 for others)."""
        return 1000 if 'JPY' in pair else 100000

    def _calculate_position_risk(self, position: Dict, account_equity: float) -> float:
        """Calculate risk amount for an open position using PipCalculator."""
        try:
            symbol = position.get('symbol', '')
            volume = position.get('volume', 0)
            entry_price = position.get('price_open', 0)
            sl = position.get('sl', 0)

            if volume > 0 and entry_price > 0 and sl > 0 and symbol:
                # Use PipCalculator for consistent calculations
                pip_value = PipCalculator.get_pip_value(symbol, volume, entry_price)
                risk_pips = abs(entry_price - sl) / PipCalculator.get_pip_size(symbol)
                return volume * risk_pips * pip_value / volume  # pip_value is already per lot
            return 0
        except Exception:
            return 0
    
    async def monitor_positions(self) -> None:
        """Monitor open positions for profit protection."""
        try:
            positions = await self.broker_manager.get_positions()
            
            for position in positions:
                ticket = position.get('ticket')
                profit = position.get('profit', 0)
                symbol = position.get('symbol', '')
                
                # Initialize position tracker if not exists
                if ticket not in self.position_trackers:
                    self.position_trackers[ticket] = {
                        'symbol': symbol,
                        'peak_profit': max(0, profit),
                        'entry_time': datetime.now().isoformat()
                    }
                
                tracker = self.position_trackers[ticket]
                
                # Update peak profit
                if profit > tracker.get('peak_profit', 0):
                    tracker['peak_profit'] = profit
                
                peak_profit = tracker.get('peak_profit', 0)
                
                # Profit protection logic
                if peak_profit >= self.trading_params['minimum_profit_to_protect']:
                    drawdown_amount = peak_profit - profit
                    drawdown_percentage = (drawdown_amount / peak_profit) if peak_profit > 0 else 0
                    
                    if drawdown_percentage >= self.trading_params['profit_protection_percentage']:
                        logger.info(f"🛡️ PROFIT PROTECTION TRIGGERED: {symbol}")
                        logger.info(f"   Peak: ${peak_profit:.2f}, Current: ${profit:.2f}")
                        logger.info(f"   Drawdown: {drawdown_percentage:.1%}")
                        
                        # Close position
                        close_result = await self.broker_manager.close_position(ticket)
                        if close_result:
                            self.positions_closed_profit += 1
                            logger.info(f"✅ Position closed with profit protection")
                            del self.position_trackers[ticket]
                
                # Stop loss protection
                elif profit <= -self.trading_params['minimum_profit_to_protect']:
                    logger.info(f"🔴 STOP LOSS PROTECTION: {symbol} at ${profit:.2f}")
                    close_result = await self.broker_manager.close_position(ticket)
                    if close_result:
                        self.positions_closed_loss += 1
                        logger.info(f"✅ Position closed with stop loss")
                        del self.position_trackers[ticket]
                        
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
    
    async def trading_loop(self) -> None:
        """Main trading loop."""
        logger.info("🔄 STARTING MAIN TRADING LOOP")
        
        while self.running:
            try:
                loop_start = time.time()
                self.scan_count += 1
                
                logger.info(f"🔄 TRADING CYCLE {self.scan_count}")
                
                # Monitor existing positions
                await self.monitor_positions()
                
                # Analyze markets for new opportunities (parallel)
                if self.running:
                    tasks = [self.analyze_market_and_trade(pair) for pair in self.currency_pairs]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            logger.error(f"Error analyzing {self.currency_pairs[i]}: {result}")
                
                # Write heartbeat
                await self.write_heartbeat()
                
                # Log cycle summary
                loop_time = time.time() - loop_start
                logger.info(f"📊 Cycle {self.scan_count} completed in {loop_time:.1f}s")
                logger.info(f"   Signals analyzed: {self.signals_analyzed}")
                logger.info(f"   Trades executed: {self.trades_executed}")
                logger.info(f"   Active positions: {len(self.position_trackers)}")
                
                # Wait before next cycle (30 seconds)
                await asyncio.sleep(30)
                
            except KeyboardInterrupt:
                logger.info("🛑 Keyboard interrupt received")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
    async def run(self) -> None:
        """Main run method."""
        try:
            # Set up signal handlers
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            logger.info("🚀 CORE TRADING BOT STARTING UP")
            logger.info("="*60)
            
            # Initialize systems
            if not await self.initialize_systems():
                logger.error("❌ System initialization failed - exiting")
                return
            
            logger.info("🎯 TRADING BOT IS NOW LIVE!")
            logger.info("💡 Press Ctrl+C to stop the bot")
            logger.info("="*60)
            
            # Start trading loop
            await self.trading_loop()
            
        except Exception as e:
            logger.error(f"❌ Critical error in main run: {e}")
            traceback.print_exc()
        finally:
            logger.info("🛑 TRADING BOT SHUTTING DOWN")
            await self._graceful_shutdown()
            await self.write_heartbeat()

    async def _graceful_shutdown(self) -> None:
        """Perform graceful shutdown with position cleanup."""
        try:
            logger.info("🔄 Performing graceful shutdown...")

            # Close all open positions
            positions = await self.broker_manager.get_positions()
            if positions:
                logger.info(f"📊 Closing {len(positions)} open positions...")
                close_tasks = []
                for position in positions:
                    ticket = position.get('ticket')
                    if ticket:
                        close_tasks.append(self.broker_manager.close_position(ticket))

                if close_tasks:
                    close_results = await asyncio.gather(*close_tasks, return_exceptions=True)
                    successful_closes = sum(1 for r in close_results if not isinstance(r, Exception) and r.get('status') == 'SUCCESS')
                    logger.info(f"✅ Closed {successful_closes}/{len(close_tasks)} positions")

            # Shutdown broker connection
            if hasattr(self.broker_manager, 'shutdown'):
                await self.broker_manager.shutdown()
                logger.info("✅ Broker connection closed")

            logger.info("🎯 Graceful shutdown completed")

        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")

async def main():
    """Main entry point."""
    bot = CoreTradingBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
