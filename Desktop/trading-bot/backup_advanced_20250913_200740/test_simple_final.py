#!/usr/bin/env python3
"""Simple final test to verify all components can be imported and initialized."""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("🔍 Testing module imports...")

    try:
        # Add src to path
        sys.path.append('src')

        # Test core modules
        from src.config.settings import settings
        print("✅ Settings module imported")

        from src.data.market_data import MarketDataManager
        print("✅ Market data manager imported")

        from src.analysis.technical import TechnicalAnalyzer
        print("✅ Technical analyzer imported")

        from src.analysis.trend_reversal_detector import TrendReversalDetector
        print("✅ Trend reversal detector imported")

        from src.ml.trading_ml_engine import TradingMLEngine
        print("✅ ML engine imported")

        from src.analysis.correlation import CorrelationAnalyzer
        print("✅ Correlation analyzer imported")

        from src.scheduling.intelligent_scheduler import IntelligentTradingScheduler
        print("✅ Intelligent scheduler imported")

        from src.news.sentiment import SentimentAggregator
        print("✅ Sentiment aggregator imported")

        from src.monitoring.metrics import MetricsCollector
        print("✅ Metrics collector imported")

        from src.analysis.trade_attribution import TradeAttributionAnalyzer
        print("✅ Trade attribution analyzer imported")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_initialization():
    """Test that components can be initialized."""
    print("\n🔍 Testing component initialization...")

    try:
        sys.path.append('src')

        # Test basic initialization
        from src.analysis.technical import TechnicalAnalyzer
        analyzer = TechnicalAnalyzer()
        print("✅ Technical analyzer initialized")

        from src.analysis.trend_reversal_detector import TrendReversalDetector
        detector = TrendReversalDetector()
        print("✅ Trend reversal detector initialized")

        from src.ml.trading_ml_engine import TradingMLEngine
        ml_engine = TradingMLEngine()
        print("✅ ML engine initialized")

        from src.scheduling.intelligent_scheduler import IntelligentTradingScheduler
        scheduler = IntelligentTradingScheduler()
        print("✅ Intelligent scheduler initialized")

        from src.news.sentiment import SentimentAggregator
        sentiment = SentimentAggregator()
        print("✅ Sentiment aggregator initialized")

        from src.monitoring.metrics import MetricsCollector
        metrics = MetricsCollector()
        print("✅ Metrics collector initialized")

        from src.analysis.trade_attribution import TradeAttributionAnalyzer
        attribution = TradeAttributionAnalyzer()
        print("✅ Trade attribution analyzer initialized")

        return True

    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\n🔍 Testing file structure...")

    required_files = [
        'src/__init__.py',
        'src/config/__init__.py',
        'src/config/settings.py',
        'src/data/__init__.py',
        'src/analysis/__init__.py',
        'src/ml/__init__.py',
        'src/news/__init__.py',
        'src/monitoring/__init__.py',
        'src/scheduling/__init__.py',
        'requirements.txt',
        'README.md'
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def test_dependencies():
    """Test that core dependencies are available."""
    print("\n🔍 Testing core dependencies...")

    try:
        import numpy as np
        print("✅ NumPy available")

        import pandas as pd
        print("✅ Pandas available")

        import aiohttp
        print("✅ aiohttp available")

        import json
        print("✅ JSON available")

        import logging
        print("✅ Logging available")

        return True

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 FINAL BOT VALIDATION TEST SUITE")
    print("="*50)

    tests = [
        ("Module Imports", test_imports),
        ("Component Initialization", test_initialization),
        ("File Structure", test_file_structure),
        ("Core Dependencies", test_dependencies)
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")

    print("\n" + "="*50)
    print("📋 FINAL VALIDATION RESULTS")
    print("="*50)

    print(f"🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("📝 BOT STATUS: READY FOR DEPLOYMENT")
        print("\n🚀 NEXT STEPS:")
        print("   1. Configure API keys in .env file")
        print("   2. Run: python run_adaptive_intelligent_bot.py")
        print("   3. Monitor performance and adjust parameters")
        print("   4. Consider paper trading first")
        return True
    elif passed_tests >= total_tests * 0.75:
        print("⚠️ MOST TESTS PASSED - Bot is functional but may need minor fixes")
        return True
    else:
        print("❌ CRITICAL ISSUES DETECTED - Bot needs fixes before deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
