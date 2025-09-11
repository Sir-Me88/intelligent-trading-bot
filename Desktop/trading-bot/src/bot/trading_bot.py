"""Main trading bot orchestrator."""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import signal
import sys

from ..config.settings import settings
from ..data.market_data import MarketDataManager
from ..analysis.technical import TechnicalAnalyzer, SignalDirection
from ..analysis.correlation import CorrelationAnalyzer
from ..news.economic_calendar import EconomicCalendarFilter
from ..news.sentiment import SentimentAggregator
from ..trading.position_manager import PositionManager
from ..trading.broker_interface import BrokerManager
from ..monitoring.metrics import MetricsCollector
from ..monitoring.logger import setup_logging

logger = logging.getLogger(__name__)


class TradingBot:
    """Main trading bot class."""
    
    def __init__(self):
        self.running = False
        self.data_manager = MarketDataManager()
        self.technical_analyzer = TechnicalAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer(self.data_manager)
        self.economic_filter = EconomicCalendarFilter()
        self.sentiment_aggregator = SentimentAggregator()
        self.position_manager = PositionManager(self.data_manager)
        self.broker_manager = BrokerManager()
        self.metrics_collector = MetricsCollector()
        
        # Trading state
        self.last_scan_time = None
        self.scan_interval = 300  # 5 minutes
        self.daily_trade_count = 0
        self.last_trade_date = datetime.now().date()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing trading bot...")
        
        try:
            # Initialize data providers
            await self.data_manager.initialize()
            
            # Initialize broker connections
            await self.broker_manager.initialize()
            
            # Skip economic calendar for now (ForexFactory blocking)
            # await self.economic_filter.update_events()
            
            # Update correlation matrix
            currency_pairs = settings.get_currency_pairs()
            await self.correlation_analyzer.update_correlation_matrix(currency_pairs)
            
            # Start metrics collection
            await self.metrics_collector.start()
            
            logger.info("Trading bot initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize trading bot: {e}")
            raise
    
    async def run_single_cycle(self):
        """Run a single trading cycle - useful for autonomous operation."""
        loop_start = datetime.now()

        try:
            # Update account info
            account_info = await self.broker_manager.get_account_info()
            self.metrics_collector.update_account_metrics(account_info)

            # Update positions
            await self.position_manager.update_positions()

            # Scan for new signals
            await self._scan_for_signals(account_info)

            # Update correlation matrix periodically
            if self.correlation_analyzer.is_correlation_matrix_stale():
                await self.correlation_analyzer.update_correlation_matrix(settings.currency_pairs)

            # Update economic calendar periodically
            if self.economic_filter.is_cache_stale():
                await self.economic_filter.update_events()

            # Record loop metrics
            loop_duration = (datetime.now() - loop_start).total_seconds()
            self.metrics_collector.record_loop_duration(loop_duration)

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            self.metrics_collector.increment_error_count()
            raise

    async def run(self):
        """Main trading loop."""
        self.running = True
        logger.info("Starting trading bot main loop...")

        try:
            while self.running:
                await self.run_single_cycle()

                # Sleep until next scan
                await asyncio.sleep(self.scan_interval)

        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}")
            raise
        
        finally:
            await self._cleanup()
    
    async def _scan_for_signals(self, account_info: Dict):
        """Scan all currency pairs for trading signals."""
        logger.debug("Scanning for trading signals...")
        
        open_positions = self.position_manager.get_open_positions()
        
        for pair in settings.get_currency_pairs():
            try:
                # Skip economic calendar check for now (ForexFactory blocking)
                # economic_check = self.economic_filter.should_skip_pair(
                #     pair, 
                #     settings.news.impact_threshold,
                #     settings.news.time_buffer_hours
                # )
                # 
                # if economic_check['should_skip']:
                #     logger.debug(f"Skipping {pair}: {economic_check['reason']}")
                #     continue
                
                # Get market data
                df_15m = await self.data_manager.get_candles(pair, "M15", 100)
                df_1h = await self.data_manager.get_candles(pair, "H1", 100)
                
                if len(df_15m) < 50 or len(df_1h) < 50:
                    logger.warning(f"Insufficient data for {pair}")
                    continue
                
                # Generate technical signal
                signal = self.technical_analyzer.generate_signal(df_15m, df_1h)
                
                if signal['direction'] == SignalDirection.NONE:
                    continue
                
                logger.info(f"Signal detected for {pair}: {signal['direction'].value}")
                
                # Add pair to signal
                signal['pair'] = pair
                signal['timestamp'] = datetime.now()
                
                # Get sentiment analysis
                sentiment_data = await self.sentiment_aggregator.get_overall_sentiment(pair)
                
                # Apply sentiment filter
                sentiment_recommendation = sentiment_data['recommendation']
                if sentiment_recommendation['action'] == 'ignore':
                    logger.debug(f"Skipping {pair}: {sentiment_recommendation['reason']}")
                    continue
                
                # Check for hedging opportunities
                hedge_analysis = self.correlation_analyzer.should_hedge_position(
                    [p.__dict__ for p in open_positions],
                    signal,
                    settings.trading.correlation_threshold
                )
                
                # Validate trade with risk management
                validation = await self.position_manager.risk_manager.validate_trade(
                    signal, account_info, open_positions
                )
                
                if not validation['approved']:
                    logger.debug(f"Trade rejected for {pair}: {validation['reason']}")
                    continue
                
                # Apply sentiment position sizing
                position_size = validation['position_size']
                if sentiment_recommendation['action'] == 'reduce_position_size':
                    position_size *= sentiment_recommendation['factor']
                    logger.info(f"Reducing position size for {pair} due to sentiment")
                
                # Execute trade
                await self._execute_trade(signal, position_size, sentiment_data, hedge_analysis)
                
            except Exception as e:
                logger.error(f"Error scanning {pair}: {e}")
                continue
    
    async def _execute_trade(self, signal: Dict, position_size: float,
                           sentiment_data: Dict, hedge_analysis: Dict):
        """Execute a trading signal."""
        pair = signal['pair']

        # SAFETY CHECKS
        if settings.emergency_stop:
            logger.warning("Emergency stop activated - skipping trade execution")
            return

        if settings.demo_mode:
            logger.info(f"DEMO MODE: Would execute {signal['direction'].value} {position_size} {pair}")
            return

        # Check daily trade limit
        if hasattr(self, 'daily_trade_count') and self.daily_trade_count >= settings.max_daily_trades:
            logger.warning(f"Daily trade limit ({settings.max_daily_trades}) reached - skipping trade")
            return

        try:
            # Create order
            order = {
                'pair': pair,
                'direction': signal['direction'].value,
                'size': position_size,
                'entry_price': signal['entry_price'],
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit']
            }
            
            # Place order with broker
            result = await self.broker_manager.place_order(
                symbol=order['pair'],
                order_type=order['direction'],
                volume=order['size'],
                price=order['entry_price'],
                sl=order['stop_loss'],
                tp=order['take_profit']
            )

            if result.get('ticket'):
                # Create position in position manager
                position = self.position_manager.create_position(signal, position_size)

                # Update daily trade counter
                current_date = datetime.now().date()
                if current_date != self.last_trade_date:
                    self.daily_trade_count = 0
                    self.last_trade_date = current_date
                self.daily_trade_count += 1

                # Log trade
                logger.info(f"Trade executed: {signal['direction'].value} {position_size} {pair} @ {signal['entry_price']} (Daily trades: {self.daily_trade_count})")

                # Record metrics
                self.metrics_collector.record_trade(signal, position_size, sentiment_data)
                
                # Handle hedging if needed
                if hedge_analysis['should_hedge']:
                    await self._handle_hedging(hedge_analysis, signal)
                
            else:
                logger.error(f"Order failed for {pair}: {result.get('error', 'Unknown error')}")
                self.metrics_collector.increment_error_count()
                
        except Exception as e:
            logger.error(f"Error executing trade for {pair}: {e}")
            self.metrics_collector.increment_error_count()
    
    async def _handle_hedging(self, hedge_analysis: Dict, original_signal: Dict):
        """Handle hedging logic for correlated pairs."""
        logger.info(f"Implementing hedging strategy for {original_signal['pair']}")
        
        for hedge_pair_info in hedge_analysis['hedge_pairs']:
            try:
                hedge_pair = hedge_pair_info['pair']
                correlation = hedge_pair_info['correlation']
                hedge_ratio = hedge_pair_info['hedge_ratio']
                
                # Determine hedge direction
                if correlation > 0:
                    # Positive correlation - opposite direction
                    hedge_direction = 'sell' if original_signal['direction'].value == 'buy' else 'buy'
                else:
                    # Negative correlation - same direction
                    hedge_direction = original_signal['direction'].value
                
                # Calculate hedge size
                original_size = original_signal.get('size', 0)
                hedge_size = original_size * hedge_ratio
                
                # Get current price for hedge pair
                price_data = await self.data_manager.get_current_price(hedge_pair)
                hedge_price = price_data['ask'] if hedge_direction == 'buy' else price_data['bid']
                
                # Create hedge order
                hedge_order = {
                    'pair': hedge_pair,
                    'direction': hedge_direction,
                    'size': hedge_size,
                    'entry_price': hedge_price,
                    'stop_loss': None,  # Hedges typically don't have stops
                    'take_profit': None
                }
                
                # Place hedge order
                result = await self.broker_manager.place_order(hedge_order)
                
                if result['success']:
                    logger.info(f"Hedge executed: {hedge_direction} {hedge_size} {hedge_pair}")
                else:
                    logger.error(f"Hedge order failed for {hedge_pair}: {result.get('error')}")
                
            except Exception as e:
                logger.error(f"Error executing hedge: {e}")
    
    async def _cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up trading bot resources...")
        
        try:
            await self.data_manager.cleanup()
            await self.broker_manager.cleanup()
            await self.metrics_collector.stop()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


async def main():
    """Main entry point."""
    # Setup logging
    setup_logging()
    
    # Create and run bot
    bot = TradingBot()
    
    try:
        await bot.initialize()
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())










