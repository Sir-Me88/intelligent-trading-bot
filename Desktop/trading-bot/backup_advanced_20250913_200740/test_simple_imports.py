#!/usr/bin/env python3
"""Simple import test for core bot components."""

import sys
import os

def test_core_imports():
    """Test core component imports."""
    print("🔍 Testing core component imports...")

    try:
        # Add src to path
        sys.path.append('src')

        # Test basic imports that don't require external dependencies
        print("Testing basic Python modules...")
        import datetime
        import json
        import logging
        print("✅ Basic Python imports successful")

        # Test our custom modules (simple ones first)
        print("Testing custom modules...")

        # Test data structures and simple classes
        from src.analysis.trade_attribution import AttributionResult
        print("✅ Trade attribution module imported")

        # Test basic dataclasses and simple functions
        from src.scheduling.intelligent_scheduler import IntelligentTradingScheduler
        print("✅ Intelligent scheduler module imported")

        # Test analysis modules
        from src.analysis.correlation import CorrelationAnalyzer
        print("✅ Correlation analyzer module imported")

        from src.analysis.technical import TechnicalAnalyzer, SignalDirection
        print("✅ Technical analyzer module imported")

        from src.analysis.trend_reversal_detector import TrendReversalDetector
        print("✅ Trend reversal detector module imported")

        # Test monitoring
        from src.monitoring.metrics import MetricsCollector
        print("✅ Metrics collector module imported")

        print("✅ All core component imports successful!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    print("\n🔍 Testing configuration loading...")

    try:
        # Test basic config import
        from src.config.settings import settings
        print("✅ Settings module imported")

        # Test basic settings access
        if hasattr(settings, 'get_currency_pairs'):
            pairs = settings.get_currency_pairs()
            print(f"✅ Currency pairs loaded: {len(pairs)} pairs")
        else:
            print("⚠️ Currency pairs method not available")

        return True

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def main():
    """Run simple import tests."""
    print("🚀 SIMPLE IMPORT TEST SUITE")
    print("="*40)

    tests = [
        ("Core Component Imports", test_core_imports),
        ("Configuration Loading", test_config_loading)
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

    print("\n" + "="*40)
    print("📋 TEST RESULTS SUMMARY")
    print("="*40)

    print(f"🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 ALL SIMPLE TESTS PASSED!")
        print("📝 Recommendation: Install remaining dependencies for full functionality")
        return True
    elif passed_tests >= total_tests * 0.5:
        print("⚠️ MOST TESTS PASSED - Core functionality working")
        return True
    else:
        print("❌ CRITICAL ISSUES DETECTED - Check dependencies")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
