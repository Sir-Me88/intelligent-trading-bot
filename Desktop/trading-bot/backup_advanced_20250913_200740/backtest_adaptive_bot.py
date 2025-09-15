#!/usr/bin/env python3
"""Backtesting script for the Adaptive Intelligent Trading Bot."""

import asyncio
import logging
import sys
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from src.config.settings import settings, app_settings
from src.data.backtest_data_manager import BacktestDataManager
from src.trading.backtest_broker import BacktestBrokerManager
from src.analysis.technical import TechnicalAnalyzer
from src.analysis.trend_reversal_detector import TrendReversalDetector
from src.ml.trading_ml_engine import TradingMLEngine
from src.ml.trade_analyzer import TradeAnalyzer
from src.analysis.correlation import CorrelationAnalyzer
from src.scheduling.intelligent_scheduler import IntelligentTradingScheduler

# Setup logging
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / "backtest_adaptive_bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class BacktestAdaptiveBot:
    """Backtesting version of the Adaptive Intelligent Bot."""

    def __init__(self, initial_balance: float = 10000.0,
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None):
        # Set backtest mode
        settings.backtest_mode = True

        # Initialize components
        self.data_manager = BacktestDataManager()
        self.broker_manager = BacktestBrokerManager(initial_balance)
        self.technical_analyzer = TechnicalAnalyzer()
        self.reversal_detector = TrendReversalDetector()
        self.ml_engine = TradingMLEngine()
        self.trade_analyzer = TradeAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer(self.data_manager)
        self.scheduler = IntelligentTradingScheduler()

        # Backtest parameters
        self.start_date = start_date or (datetime.now(timezone.utc) - timedelta(days=30))
        self.end_date = end_date or datetime.now(timezone.utc)
        self.scan_interval = 60  # seconds between scans
        self.current_time = self.start_date

        # Statistics
        self.total_scans = 0
        self.signals_analyzed = 0
        self.trades_executed = 0

        logger.info("🧪 BACKTEST ADAPTIVE BOT INITIALIZED")
        logger.info(f"   Period: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"   Initial Balance: ${initial_balance:.2f}")

    async def initialize(self) -> bool:
        """Initialize backtest components."""
        try:
            logger.info("🔧 Initializing backtest components...")

            # Initialize data manager
            if not await self.data_manager.initialize():
                logger.error("Failed to initialize data manager")
                return False

            # Load historical data
            symbols = app_settings.get_currency_pairs()
            # Load both M1 and M15 data for backtesting
            if not await self.data_manager.load_historical_data(symbols, "M1", self.start_date, self.end_date):
                logger.error("Failed to load M1 historical data")
                return False
            if not await self.data_manager.load_historical_data(symbols, "M15", self.start_date, self.end_date):
                logger.error("Failed to load M15 historical data")
                return False

            # Initialize broker
            if not await self.broker_manager.initialize():
                logger.error("Failed to initialize broker")
                return False

            # Initialize scheduler (handle network failures gracefully)
            try:
                await self.scheduler.initialize()
                logger.info("✅ Scheduler initialized successfully")
            except Exception as e:
                logger.warning(f"Scheduler initialization failed (continuing without news): {e}")
                # Create a mock scheduler that always allows trading
                async def mock_should_execute_trades():
                    return True
                self.scheduler.should_execute_trades = mock_should_execute_trades

            logger.info("✅ Backtest initialization complete")
            return True

        except Exception as e:
            logger.error(f"Error initializing backtest: {e}")
            return False

    async def run_backtest(self) -> Dict:
        """Run the backtest simulation."""
        try:
            logger.info("🚀 STARTING BACKTEST SIMULATION")
            logger.info("="*60)

            # Reset to start
            self.data_manager.reset_to_start()
            self.current_time = self.start_date

            # Main backtest loop
            while True:
                # Update current time
                self.current_time = self.data_manager.get_current_time()
                if self.current_time is None:
                    logger.info("Reached end of historical data")
                    break

                # Update broker with current prices
                current_prices = self.data_manager.get_current_prices()
                self.broker_manager.update_positions(current_prices)

                # Override broker's price getter
                self.broker_manager._get_current_price = lambda symbol: current_prices.get(symbol, 1.0)

                # Run trading scan
                await self.run_trading_scan()

                # Advance time
                if not self.data_manager.advance_time():
                    break

                # Small delay to prevent overwhelming logs
                await asyncio.sleep(0.01)

            # Generate final report
            return await self.generate_backtest_report()

        except Exception as e:
            logger.error(f"Error during backtest: {e}")
            return {'error': str(e)}

    async def run_trading_scan(self):
        """Run a single trading scan."""
        try:
            self.total_scans += 1

            # Check if we should execute trades (backtest mode always returns True)
            if not await self.scheduler.should_execute_trades():
                return

            logger.info(f"🤖 BACKTEST SCAN #{self.total_scans} at {self.current_time}")

            # Analyze each currency pair
            for pair in app_settings.get_currency_pairs():
                try:
                    self.signals_analyzed += 1

                    # Get market data
                    df_15m = await self.data_manager.get_candles(pair, "M15", 100)
                    df_1h = await self.data_manager.get_candles(pair, "H1", 100)
                    df_h4 = await self.data_manager.get_candles(pair, "H4", 50)

                    if df_15m is None or df_1h is None or len(df_15m) < 50:
                        continue

                    # Generate signal
                    signal = await self.generate_adaptive_signal(pair, df_15m, df_1h, df_h4)

                    if signal:
                        # Execute trade
                        await self.execute_backtest_trade(pair, signal)

                except Exception as e:
                    logger.error(f"Error processing {pair}: {e}")

        except Exception as e:
            logger.error(f"Error in trading scan: {e}")

    async def generate_adaptive_signal(self, pair: str, df_15m, df_1h, df_h4):
        """Generate adaptive trading signal."""
        try:
            # Generate base signal
            signal = self.technical_analyzer.generate_signal(df_15m, df_1h)

            if signal['direction'].value == 'NONE':
                return None

            # Apply confidence threshold
            confidence = signal.get('confidence', 0)
            if confidence < 0.75:  # LOWERED from 0.85 to allow more signals
                return None

            # Apply R/R ratio
            if signal['direction'].value == 'BUY':
                risk = abs(signal['entry_price'] - signal['stop_loss'])
                reward = abs(signal['take_profit'] - signal['entry_price'])
            else:
                risk = abs(signal['stop_loss'] - signal['entry_price'])
                reward = abs(signal['entry_price'] - signal['take_profit'])

            rr_ratio = reward / risk if risk > 0 else 0
            if rr_ratio < app_settings.adaptive_params['min_rr_ratio']:
                return None

            logger.info(f"   {pair}: SIGNAL ACCEPTED - {signal['direction'].value} (Conf: {confidence:.1%}, R/R: {rr_ratio:.2f})")
            return signal

        except Exception as e:
            logger.error(f"Error generating signal for {pair}: {e}")
            return None

    async def execute_backtest_trade(self, pair: str, signal):
        """Execute trade in backtest environment."""
        try:
            # Calculate position size (1% risk per trade)
            account_info = await self.broker_manager.get_account_info()
            balance = account_info['balance']

            if signal['direction'].value == 'BUY':
                risk = abs(signal['entry_price'] - signal['stop_loss'])
            else:
                risk = abs(signal['stop_loss'] - signal['entry_price'])

            # Risk 1% of balance
            risk_amount = balance * 0.01
            pip_value = 10  # $10 per pip for standard lot
            volume = round((risk_amount / (risk * 10000 * pip_value)), 2)
            volume = max(0.01, min(volume, 1.0))  # Min 0.01, Max 1.0 lot

            # Place order
            order_result = await self.broker_manager.place_order(
                symbol=pair,
                order_type=signal['direction'].value,
                volume=volume,
                sl=signal['stop_loss'],
                tp=signal['take_profit']
            )

            if order_result['status'] == 'SUCCESS':
                self.trades_executed += 1
                logger.info(f"   ✅ BACKTEST TRADE EXECUTED: {pair} {signal['direction'].value} {volume} lots")

        except Exception as e:
            logger.error(f"Error executing backtest trade for {pair}: {e}")

    async def generate_backtest_report(self) -> Dict:
        """Generate comprehensive backtest report."""
        try:
            logger.info("📊 GENERATING BACKTEST REPORT")
            logger.info("="*60)

            # Get broker statistics
            broker_stats = self.broker_manager.get_backtest_stats()

            # Get data summary
            data_summary = self.data_manager.get_data_summary()

            # Calculate additional metrics
            total_days = (self.end_date - self.start_date).days
            avg_trades_per_day = broker_stats['total_trades'] / total_days if total_days > 0 else 0

            report = {
                'backtest_period': {
                    'start_date': self.start_date.isoformat(),
                    'end_date': self.end_date.isoformat(),
                    'total_days': total_days
                },
                'performance_metrics': {
                    'initial_balance': self.broker_manager.initial_balance,
                    'final_balance': broker_stats['final_balance'],
                    'net_profit': broker_stats['net_profit'],
                    'return_percentage': broker_stats['return_pct'],
                    'max_drawdown': broker_stats['max_drawdown']
                },
                'trading_statistics': {
                    'total_trades': broker_stats['total_trades'],
                    'winning_trades': broker_stats['winning_trades'],
                    'losing_trades': broker_stats['losing_trades'],
                    'win_rate': broker_stats['win_rate'],
                    'avg_win': broker_stats['avg_win'],
                    'avg_loss': broker_stats['avg_loss'],
                    'profit_factor': broker_stats['profit_factor'],
                    'avg_trades_per_day': avg_trades_per_day
                },
                'backtest_info': {
                    'total_scans': self.total_scans,
                    'signals_analyzed': self.signals_analyzed,
                    'trades_executed': self.trades_executed
                },
                'data_summary': data_summary,
                'trades_history': self.broker_manager.trades_history
            }

            # Print summary to console
            self.print_report_summary(report)

            # Save detailed report
            await self.save_report(report)

            return report

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {'error': str(e)}

    def print_report_summary(self, report: Dict):
        """Print backtest summary to console."""
        perf = report['performance_metrics']
        stats = report['trading_statistics']

        print("\n" + "="*60)
        print("🧪 BACKTEST RESULTS SUMMARY")
        print("="*60)
        print(".2f")
        print(".2f")
        print(".1f")
        print(".2f")
        print(".1f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print("="*60)

    async def save_report(self, report: Dict):
        """Save detailed backtest report to file."""
        try:
            reports_dir = Path("backtest_reports")
            reports_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_report_{timestamp}.json"

            with open(reports_dir / filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"📄 Detailed report saved to: {reports_dir / filename}")

        except Exception as e:
            logger.error(f"Error saving report: {e}")

async def main():
    """Main backtest function."""
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='Backtest Adaptive Intelligent Trading Bot')
        parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')
        parser.add_argument('--days', type=int, default=30, help='Backtest period in days')
        parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')

        args = parser.parse_args()

        # Calculate dates
        if args.start_date:
            start_date = datetime.fromisoformat(args.start_date).replace(tzinfo=timezone.utc)
        else:
            start_date = datetime.now(timezone.utc) - timedelta(days=args.days)

        end_date = datetime.now(timezone.utc)

        # Initialize and run backtest
        bot = BacktestAdaptiveBot(
            initial_balance=args.balance,
            start_date=start_date,
            end_date=end_date
        )

        if not await bot.initialize():
            logger.error("Failed to initialize backtest")
            return

        # Run backtest
        report = await bot.run_backtest()

        if 'error' in report:
            logger.error(f"Backtest failed: {report['error']}")
        else:
            logger.info("✅ Backtest completed successfully")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
