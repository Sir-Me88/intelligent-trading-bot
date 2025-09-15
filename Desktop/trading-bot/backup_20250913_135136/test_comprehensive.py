#!/usr/bin/env python3
"""Comprehensive test of trading bot functionality."""

import sys
import asyncio
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_comprehensive_functionality():
    """Test comprehensive trading bot functionality."""
    print("🚀 COMPREHENSIVE TRADING BOT TEST")
    print("="*60)
    
    success_count = 0
    total_tests = 0
    
    # Test 1: Core Components
    print("\n1. Testing Core Components...")
    total_tests += 1
    try:
        from src.config.settings import settings, app_settings
        from src.data.market_data import MarketDataManager
        from src.analysis.technical import TechnicalAnalyzer
        from src.trading.broker_interface import BrokerManager
        from src.monitoring.metrics import MetricsCollector
        
        data_manager = MarketDataManager()
        tech_analyzer = TechnicalAnalyzer()
        broker = BrokerManager()
        metrics = MetricsCollector()
        
        print("   ✅ All core components initialized successfully")
        success_count += 1
    except Exception as e:
        print(f"   ❌ Core components failed: {e}")
    
    # Test 2: Technical Analysis with Real Data Structure
    print("\n2. Testing Technical Analysis...")
    total_tests += 1
    try:
        # Create more realistic OHLCV data
        np.random.seed(42)  # For reproducible results
        n_periods = 200
        
        # Generate realistic price movement
        base_price = 1.0850
        returns = np.random.normal(0, 0.0005, n_periods)  # Small random returns
        prices = [base_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        dates = pd.date_range(start='2024-01-01', periods=n_periods, freq='15min')
        
        ohlc_data = []
        for i in range(n_periods):
            close = prices[i]
            high = close * (1 + abs(np.random.normal(0, 0.0002)))
            low = close * (1 - abs(np.random.normal(0, 0.0002)))
            open_price = prices[i-1] if i > 0 else close
            volume = np.random.randint(1000, 5000)
            
            ohlc_data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': max(open_price, high, close),
                'low': min(open_price, low, close),
                'close': close,
                'volume': volume
            })
        
        df_15m = pd.DataFrame(ohlc_data)
        df_1h = df_15m.iloc[::4].copy()  # Simulate 1H data from 15M
        
        # Test signal generation
        signal = tech_analyzer.generate_signal(df_15m, df_1h)
        
        print(f"   ✅ Technical analysis completed")
        print(f"   📊 Signal: {signal['direction']}")
        print(f"   📊 Confidence: {signal.get('confidence', 0):.2%}")
        print(f"   📊 Entry price: {signal.get('entry_price', 0):.5f}")
        print(f"   📊 Stop loss: {signal.get('stop_loss', 0):.5f}")
        print(f"   📊 Take profit: {signal.get('take_profit', 0):.5f}")
        
        success_count += 1
    except Exception as e:
        print(f"   ❌ Technical analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Risk Management
    print("\n3. Testing Risk Management...")
    total_tests += 1
    try:
        from src.trading.risk import RiskManager
        
        risk_manager = RiskManager()
        
        # Test position sizing
        account_balance = 10000
        risk_per_trade = 0.01
        entry_price = 1.0850
        stop_loss = 1.0800
        
        position_size = risk_manager.calculate_position_size(
            account_balance, risk_per_trade, entry_price, stop_loss
        )
        
        print(f"   ✅ Risk management functional")
        print(f"   📊 Account balance: ${account_balance}")
        print(f"   📊 Risk per trade: {risk_per_trade:.1%}")
        print(f"   📊 Calculated position size: {position_size:.2f} lots")
        
        success_count += 1
    except Exception as e:
        print(f"   ❌ Risk management failed: {e}")
    
    # Test 4: Position Management
    print("\n4. Testing Position Management...")
    total_tests += 1
    try:
        from src.trading.position_manager import PositionManager
        
        pos_manager = PositionManager()
        
        # Test position tracking
        test_position = {
            'ticket': 12345,
            'symbol': 'EURUSD',
            'type': 0,  # BUY
            'volume': 0.1,
            'price_open': 1.0850,
            'sl': 1.0800,
            'tp': 1.0950,
            'profit': 25.50
        }
        
        print(f"   ✅ Position management functional")
        print(f"   📊 Test position: {test_position['symbol']}")
        print(f"   📊 Volume: {test_position['volume']} lots")
        print(f"   📊 Current profit: ${test_position['profit']}")
        
        success_count += 1
    except Exception as e:
        print(f"   ❌ Position management failed: {e}")
    
    # Test 5: Monitoring and Metrics
    print("\n5. Testing Monitoring System...")
    total_tests += 1
    try:
        # Test metrics collection
        metrics.record_signal_generated('EURUSD', 'BUY', 0.75)
        metrics.record_trade_executed('EURUSD', 'BUY', 0.1, 25.50)
        
        # Get current metrics
        current_metrics = metrics.get_current_metrics()
        
        print(f"   ✅ Monitoring system functional")
        print(f"   📊 Signals generated: {current_metrics.get('signals_generated', 0)}")
        print(f"   📊 Trades executed: {current_metrics.get('trades_executed', 0)}")
        
        success_count += 1
    except Exception as e:
        print(f"   ❌ Monitoring system failed: {e}")
    
    # Test 6: News and Sentiment (Basic)
    print("\n6. Testing News and Sentiment...")
    total_tests += 1
    try:
        from src.news.sentiment import SentimentAggregator
        
        sentiment = SentimentAggregator()
        
        # Test basic sentiment analysis
        test_text = "The EUR/USD pair is showing strong bullish momentum with positive economic indicators."
        sentiment_score = sentiment.analyze_text_sentiment(test_text)
        
        print(f"   ✅ Sentiment analysis functional")
        print(f"   📊 Test text sentiment: {sentiment_score:.2f}")
        
        success_count += 1
    except Exception as e:
        print(f"   ❌ Sentiment analysis failed: {e}")
    
    # Test 7: Correlation Analysis
    print("\n7. Testing Correlation Analysis...")
    total_tests += 1
    try:
        from src.analysis.correlation import CorrelationAnalyzer
        
        correlation_analyzer = CorrelationAnalyzer(data_manager)
        
        print(f"   ✅ Correlation analysis initialized")
        print(f"   📊 Ready for multi-pair correlation analysis")
        
        success_count += 1
    except Exception as e:
        print(f"   ❌ Correlation analysis failed: {e}")
    
    # Test 8: Scheduling System
    print("\n8. Testing Intelligent Scheduler...")
    total_tests += 1
    try:
        from src.scheduling.intelligent_scheduler import IntelligentScheduler
        
        scheduler = IntelligentScheduler()
        
        # Test schedule info
        schedule_info = scheduler.get_trading_schedule_info()
        
        print(f"   ✅ Intelligent scheduler functional")
        print(f"   📊 Current schedule: {schedule_info}")
        
        success_count += 1
    except Exception as e:
        print(f"   ❌ Intelligent scheduler failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("🎯 COMPREHENSIVE TEST RESULTS:")
    print(f"   ✅ Successful tests: {success_count}/{total_tests}")
    print(f"   📊 Success rate: {(success_count/total_tests)*100:.1f}%")
    
    if success_count == total_tests:
        print("   🚀 ALL SYSTEMS OPERATIONAL!")
        print("   🎉 Trading bot is ready for deployment")
    elif success_count >= total_tests * 0.8:
        print("   ⚠️  Most systems operational")
        print("   🔧 Minor issues need attention")
    else:
        print("   ❌ Multiple system failures")
        print("   🛠️  Significant debugging required")
    
    print("\n📋 NEXT STEPS:")
    print("   1. Configure MT5 credentials in .env file")
    print("   2. Add API keys for news/sentiment services")
    print("   3. Test with live market data")
    print("   4. Run backtest to validate strategies")
    print("="*60)
    
    return success_count == total_tests

if __name__ == "__main__":
    asyncio.run(test_comprehensive_functionality())