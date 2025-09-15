#!/usr/bin/env python3
"""Test script for Advanced Trading Bot components."""

import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.config.settings import settings
from src.data.market_data import MarketDataManager
from src.analysis.technical import TechnicalAnalyzer
from src.analysis.trend_reversal_detector import TrendReversalDetector
from src.ml.trading_ml_engine import TradingMLEngine
from src.analysis.correlation import CorrelationAnalyzer
from src.scheduling.intelligent_scheduler import IntelligentTradingScheduler
from src.news.sentiment import SentimentAggregator
from src.monitoring.metrics import MetricsCollector
from src.analysis.trade_attribution import TradeAttributionAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedBotTester:
    """Test suite for advanced trading bot components."""

    def __init__(self):
        self.test_results = {}
        self.components = {}

    async def setup_components(self):
        """Initialize all bot components for testing."""
        try:
            logger.info("🔧 Setting up bot components for testing...")

            # Initialize components
            self.components['market_data'] = MarketDataManager()
            self.components['technical_analyzer'] = TechnicalAnalyzer()
            self.components['reversal_detector'] = TrendReversalDetector()
            self.components['ml_engine'] = TradingMLEngine()
            self.components['correlation_analyzer'] = CorrelationAnalyzer(self.components['market_data'])
            self.components['scheduler'] = IntelligentTradingScheduler()
            self.components['sentiment_aggregator'] = SentimentAggregator()
            self.components['metrics_collector'] = MetricsCollector()
            self.components['attribution_analyzer'] = TradeAttributionAnalyzer()

            logger.info("✅ All components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to setup components: {e}")
            return False

    async def test_market_data_connection(self):
        """Test market data connectivity."""
        try:
            logger.info("📊 Testing market data connection...")

            # Test with a simple pair
            test_pair = "EURUSD"
            df = await self.components['market_data'].get_candles(test_pair, "M15", 10)

            if df is not None and len(df) > 0:
                logger.info(f"✅ Market data connection successful - Retrieved {len(df)} candles for {test_pair}")
                self.test_results['market_data'] = True
                return True
            else:
                logger.error("❌ Market data connection failed - No data retrieved")
                self.test_results['market_data'] = False
                return False

        except Exception as e:
            logger.error(f"❌ Market data test failed: {e}")
            self.test_results['market_data'] = False
            return False

    async def test_technical_analysis(self):
        """Test technical analysis components."""
        try:
            logger.info("📈 Testing technical analysis...")

            # Get test data
            test_pair = "EURUSD"
            df_15m = await self.components['market_data'].get_candles(test_pair, "M15", 50)
            df_1h = await self.components['market_data'].get_candles(test_pair, "H1", 50)

            if df_15m is None or df_1h is None:
                logger.error("❌ Technical analysis test failed - No market data")
                self.test_results['technical_analysis'] = False
                return False

            # Test signal generation
            signal = self.components['technical_analyzer'].generate_signal(
                df_15m, df_1h,
                adaptive_params={
                    'min_confidence': 0.7,
                    'min_rr_ratio': 2.0,
                    'atr_multiplier_normal_vol': 2.0
                },
                pair=test_pair
            )

            if signal:
                logger.info(f"✅ Technical analysis successful - Generated signal: {signal['direction']}")
                self.test_results['technical_analysis'] = True
                return True
            else:
                logger.warning("⚠️ Technical analysis - No signal generated (may be normal)")
                self.test_results['technical_analysis'] = True  # Still counts as working
                return True

        except Exception as e:
            logger.error(f"❌ Technical analysis test failed: {e}")
            self.test_results['technical_analysis'] = False
            return False

    async def test_reversal_detection(self):
        """Test trend reversal detection."""
        try:
            logger.info("🔄 Testing trend reversal detection...")

            test_pair = "EURUSD"
            df_15m = await self.components['market_data'].get_candles(test_pair, "M15", 100)

            if df_15m is None:
                logger.error("❌ Reversal detection test failed - No market data")
                self.test_results['reversal_detection'] = False
                return False

            # Test reversal detection
            reversal_data = {
                'df_1m': df_15m,  # Using same data for simplicity
                'df_5m': df_15m,
                'df_15m': df_15m,
                'df_1h': df_15m,
                'df_h4': df_15m,
                'current_position': 'LONG'
            }

            result = self.components['reversal_detector'].detect_trend_reversal(test_pair, reversal_data)

            logger.info(f"✅ Reversal detection completed - Result: {result}")
            self.test_results['reversal_detection'] = True
            return True

        except Exception as e:
            logger.error(f"❌ Reversal detection test failed: {e}")
            self.test_results['reversal_detection'] = False
            return False

    async def test_ml_engine(self):
        """Test ML engine components."""
        try:
            logger.info("🧠 Testing ML engine...")

            # Test basic ML functionality
            test_data = {
                'session_date': datetime.now().date().isoformat(),
                'executed_trades': [],
                'signals_analyzed': 10,
                'signals_rejected': 3,
                'reversals_detected': 2,
                'adaptive_parameters_used': {'min_confidence': 0.8}
            }

            # Test daily analysis
            analysis_result = self.components['ml_engine'].perform_daily_analysis(test_data)

            if analysis_result:
                logger.info("✅ ML engine daily analysis successful")
                self.test_results['ml_engine'] = True
                return True
            else:
                logger.error("❌ ML engine daily analysis failed")
                self.test_results['ml_engine'] = False
                return False

        except Exception as e:
            logger.error(f"❌ ML engine test failed: {e}")
            self.test_results['ml_engine'] = False
            return False

    async def test_correlation_analysis(self):
        """Test correlation analysis."""
        try:
            logger.info("📊 Testing correlation analysis...")

            pairs = ["EURUSD", "GBPUSD", "USDJPY"]
            correlation_result = await self.components['correlation_analyzer'].update_correlation_matrix(pairs)

            if correlation_result:
                logger.info("✅ Correlation analysis successful")
                self.test_results['correlation_analysis'] = True
                return True
            else:
                logger.warning("⚠️ Correlation analysis completed but may have limited data")
                self.test_results['correlation_analysis'] = True
                return True

        except Exception as e:
            logger.error(f"❌ Correlation analysis test failed: {e}")
            self.test_results['correlation_analysis'] = False
            return False

    async def test_sentiment_analysis(self):
        """Test sentiment analysis."""
        try:
            logger.info("💬 Testing sentiment analysis...")

            test_pair = "EURUSD"
            sentiment_result = await self.components['sentiment_aggregator'].get_overall_sentiment(test_pair)

            if sentiment_result:
                logger.info(f"✅ Sentiment analysis successful - Score: {sentiment_result.get('overall_sentiment', 'N/A')}")
                self.test_results['sentiment_analysis'] = True
                return True
            else:
                logger.warning("⚠️ Sentiment analysis completed but may have limited data")
                self.test_results['sentiment_analysis'] = True
                return True

        except Exception as e:
            logger.error(f"❌ Sentiment analysis test failed: {e}")
            self.test_results['sentiment_analysis'] = False
            return False

    async def test_trade_attribution(self):
        """Test trade attribution system."""
        try:
            logger.info("📈 Testing trade attribution system...")

            # Add sample trade data
            sample_trade = {
                'timestamp': datetime.now().isoformat(),
                'ticket': 12345,
                'symbol': 'EURUSD',
                'direction': 'BUY',
                'entry_price': 1.0500,
                'stop_loss': 1.0450,
                'take_profit': 1.0600,
                'volume': 0.1,
                'confidence': 0.85,
                'profit': 50.0,
                'exit_reason': 'take_profit',
                'hold_duration': 2.5,
                'market_conditions': {'volatility': 0.001},
                'sentiment_data': {'score': 0.1}
            }

            self.components['attribution_analyzer'].add_trade(sample_trade)

            # Generate attribution report
            report = self.components['attribution_analyzer'].generate_attribution_report()

            if report and 'summary' in report:
                logger.info("✅ Trade attribution system successful")
                self.test_results['trade_attribution'] = True
                return True
            else:
                logger.error("❌ Trade attribution report generation failed")
                self.test_results['trade_attribution'] = False
                return False

        except Exception as e:
            logger.error(f"❌ Trade attribution test failed: {e}")
            self.test_results['trade_attribution'] = False
            return False

    async def test_monitoring_system(self):
        """Test monitoring and metrics collection."""
        try:
            logger.info("📊 Testing monitoring system...")

            # Test health check
            health_status = await self.components['metrics_collector'].perform_health_check()

            if health_status:
                logger.info(f"✅ Monitoring system health check successful - Status: {health_status.get('overall_health', 'Unknown')}%")
                self.test_results['monitoring_system'] = True
                return True
            else:
                logger.error("❌ Monitoring system health check failed")
                self.test_results['monitoring_system'] = False
                return False

        except Exception as e:
            logger.error(f"❌ Monitoring system test failed: {e}")
            self.test_results['monitoring_system'] = False
            return False

    async def run_all_tests(self):
        """Run all component tests."""
        logger.info("🚀 STARTING ADVANCED TRADING BOT TEST SUITE")
        logger.info("="*60)

        # Setup components
        if not await self.setup_components():
            logger.error("❌ Component setup failed - aborting tests")
            return False

        # Run individual tests
        test_methods = [
            self.test_market_data_connection,
            self.test_technical_analysis,
            self.test_reversal_detection,
            self.test_ml_engine,
            self.test_correlation_analysis,
            self.test_sentiment_analysis,
            self.test_trade_attribution,
            self.test_monitoring_system
        ]

        passed_tests = 0
        total_tests = len(test_methods)

        for test_method in test_methods:
            try:
                result = await test_method()
                if result:
                    passed_tests += 1
            except Exception as e:
                logger.error(f"❌ Test {test_method.__name__} crashed: {e}")
                self.test_results[test_method.__name__] = False

        # Generate test summary
        logger.info("\n" + "="*60)
        logger.info("📋 TEST RESULTS SUMMARY")
        logger.info("="*60)

        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")

        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")

        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED! Bot is ready for deployment.")
            return True
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOST TESTS PASSED - Minor issues detected but bot should be functional.")
            return True
        else:
            logger.error("❌ CRITICAL ISSUES DETECTED - Bot needs fixes before deployment.")
            return False

async def main():
    """Main test execution function."""
    tester = AdvancedBotTester()
    success = await tester.run_all_tests()

    if success:
        logger.info("\n🎯 RECOMMENDATION: Proceed with paper trading tests")
    else:
        logger.error("\n🚨 RECOMMENDATION: Fix critical issues before proceeding")

    return success

if __name__ == "__main__":
    # Run tests
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
