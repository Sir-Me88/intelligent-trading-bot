#!/usr/bin/env python3
"""Basic functionality test for the trading bot."""

import sys
import asyncio
import logging
from datetime import datetime

# Add src to path
sys.path.append('src')

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_basic_components():
    """Test basic components without ML dependencies."""
    print("🔍 TESTING BASIC TRADING BOT COMPONENTS")
    print("="*50)
    
    # Test 1: Configuration
    print("\n1. Testing Configuration...")
    try:
        from src.config.settings import settings, app_settings
        print("   ✅ Settings loaded successfully")
        print(f"   📊 Trading risk per trade: {settings.trading.risk_per_trade}")
        print(f"   📊 Max open positions: {settings.trading.max_open_positions}")
    except Exception as e:
        print(f"   ❌ Settings failed: {e}")
        return False
    
    # Test 2: Market Data Manager
    print("\n2. Testing Market Data Manager...")
    try:
        from src.data.market_data import MarketDataManager
        data_manager = MarketDataManager()
        print("   ✅ Market Data Manager initialized")
    except Exception as e:
        print(f"   ❌ Market Data Manager failed: {e}")
        return False
    
    # Test 3: Technical Analyzer
    print("\n3. Testing Technical Analyzer...")
    try:
        from src.analysis.technical import TechnicalAnalyzer
        tech_analyzer = TechnicalAnalyzer()
        print("   ✅ Technical Analyzer initialized")
    except Exception as e:
        print(f"   ❌ Technical Analyzer failed: {e}")
        return False
    
    # Test 4: Broker Interface
    print("\n4. Testing Broker Interface...")
    try:
        from src.trading.broker_interface import BrokerManager
        broker = BrokerManager()
        print("   ✅ Broker Manager initialized")
    except Exception as e:
        print(f"   ❌ Broker Manager failed: {e}")
        return False
    
    # Test 5: Monitoring
    print("\n5. Testing Monitoring...")
    try:
        from src.monitoring.metrics import MetricsCollector
        metrics = MetricsCollector()
        print("   ✅ Metrics Collector initialized")
    except Exception as e:
        print(f"   ❌ Metrics Collector failed: {e}")
        return False
    
    # Test 6: Basic Data Processing (without external APIs)
    print("\n6. Testing Basic Data Processing...")
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(1.0800, 1.0900, 100),
            'high': np.random.uniform(1.0810, 1.0910, 100),
            'low': np.random.uniform(1.0790, 1.0890, 100),
            'close': np.random.uniform(1.0800, 1.0900, 100),
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Test technical analysis on sample data
        signal = tech_analyzer.generate_signal(sample_data, sample_data)
        print(f"   ✅ Sample signal generated: {signal['direction']}")
        print(f"   📊 Signal confidence: {signal.get('confidence', 0):.2%}")
        
    except Exception as e:
        print(f"   ❌ Data processing failed: {e}")
        return False
    
    print("\n" + "="*50)
    print("🎯 BASIC FUNCTIONALITY TEST RESULTS:")
    print("   ✅ Core components working")
    print("   ✅ Technical analysis functional")
    print("   ✅ Data processing operational")
    print("   ⚠️  ML components need API keys for full functionality")
    print("   ⚠️  Broker connection needs MT5 credentials")
    print("="*50)
    
    return True

if __name__ == "__main__":
    asyncio.run(test_basic_components())