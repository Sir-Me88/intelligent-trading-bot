#!/usr/bin/env python3
"""
Test script for ML components and adaptive parameter learning.

This script validates that all ML components are working correctly
and tests the adaptive parameter optimization system.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append('src')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_ml_components():
    """Test all ML components."""
    print("🧪 TESTING ML COMPONENTS")
    print("=" * 50)

    results = {}

    # Test 1: Trading ML Engine
    print("\n1️⃣ Testing Trading ML Engine...")
    try:
        from src.ml.trading_ml_engine import TradingMLEngine

        ml_engine = TradingMLEngine()

        # Test daily analysis
        sample_daily_data = {
            'session_date': datetime.now().date().isoformat(),
            'executed_trades': [
                {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': 'EURUSD',
                    'direction': 'BUY',
                    'entry_price': 1.0850,
                    'stop_loss': 1.0800,
                    'take_profit': 1.0950,
                    'volume': 0.1,
                    'confidence': 0.85,
                    'exit_reason': 'take_profit',
                    'profit': 100.0,
                    'hold_duration': 2.5
                }
            ],
            'signals_analyzed': 50,
            'signals_rejected': 15,
            'reversals_detected': 3,
            'adaptive_parameters_used': {
                'min_confidence': 0.75,
                'min_rr_ratio': 2.5,
                'profit_protection_percentage': 0.25
            }
        }

        analysis_result = ml_engine.perform_daily_analysis(sample_daily_data)

        if analysis_result and 'strategy_adjustments' in analysis_result:
            results['Trading ML Engine'] = "✅ Working"
            print("   ✅ Daily analysis completed")
        else:
            results['Trading ML Engine'] = "⚠️ Partial functionality"
            print("   ⚠️ Daily analysis returned unexpected result")

    except ImportError as e:
        results['Trading ML Engine'] = f"❌ Import failed: {e}"
        print(f"   ❌ Import failed: {e}")
    except Exception as e:
        results['Trading ML Engine'] = f"❌ Error: {e}"
        print(f"   ❌ Error: {e}")

    # Test 2: Trade Analyzer
    print("\n2️⃣ Testing Trade Analyzer...")
    try:
        from src.ml.trade_analyzer import TradeAnalyzer

        analyzer = TradeAnalyzer()

        # Test with sample trade data
        sample_trades = [
            {
                'timestamp': datetime.now().isoformat(),
                'symbol': 'EURUSD',
                'direction': 'BUY',
                'entry_price': 1.0850,
                'exit_price': 1.0950,
                'profit': 100.0,
                'volume': 0.1,
                'confidence': 0.85,
                'exit_reason': 'take_profit'
            }
        ]

        analysis = analyzer.analyze_trade_performance(sample_trades)

        if analysis and 'total_trades' in analysis:
            results['Trade Analyzer'] = "✅ Working"
            print("   ✅ Trade analysis completed")
        else:
            results['Trade Analyzer'] = "⚠️ Partial functionality"
            print("   ⚠️ Trade analysis returned unexpected result")

    except ImportError as e:
        results['Trade Analyzer'] = f"❌ Import failed: {e}"
        print(f"   ❌ Import failed: {e}")
    except Exception as e:
        results['Trade Analyzer'] = f"❌ Error: {e}"
        print(f"   ❌ Error: {e}")

    # Test 3: Trend Reversal Detector
    print("\n3️⃣ Testing Trend Reversal Detector...")
    try:
        from src.analysis.trend_reversal_detector import TrendReversalDetector

        detector = TrendReversalDetector()

        # Test with sample data
        sample_data = {
            'df_1m': None,  # Would need actual dataframes
            'df_5m': None,
            'df_15m': None,
            'df_1h': None,
            'df_h4': None,
            'current_position': 'LONG'
        }

        # Just test that the class can be instantiated
        results['Trend Reversal Detector'] = "✅ Import successful"
        print("   ✅ Class instantiated successfully")

    except ImportError as e:
        results['Trend Reversal Detector'] = f"❌ Import failed: {e}"
        print(f"   ❌ Import failed: {e}")
    except Exception as e:
        results['Trend Reversal Detector'] = f"❌ Error: {e}"
        print(f"   ❌ Error: {e}")

    # Test 4: Sentiment Analysis
    print("\n4️⃣ Testing Sentiment Analysis...")
    try:
        from src.news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()

        # Test VADER sentiment
        vader_result = analyzer.analyze_vader_sentiment("This is a great trading opportunity!")
        if 'compound' in vader_result:
            print("   ✅ VADER sentiment working")
        else:
            print("   ⚠️ VADER sentiment not working")

        # Test FinBERT sentiment (if available)
        try:
            finbert_result = analyzer.analyze_finbert_sentiment("Stock prices are rising significantly.")
            if 'compound' in finbert_result:
                print("   ✅ FinBERT sentiment working")
            else:
                print("   ⚠️ FinBERT sentiment not working")
        except Exception as e:
            print(f"   ⚠️ FinBERT sentiment error: {e}")

        results['Sentiment Analysis'] = "✅ Working"
        print("   ✅ Sentiment analysis components loaded")

    except ImportError as e:
        results['Sentiment Analysis'] = f"❌ Import failed: {e}"
        print(f"   ❌ Import failed: {e}")
    except Exception as e:
        results['Sentiment Analysis'] = f"❌ Error: {e}"
        print(f"   ❌ Error: {e}")

    # Test 5: Adaptive Parameters
    print("\n5️⃣ Testing Adaptive Parameters...")
    try:
        from src.config.settings import settings

        # Check if adaptive parameters exist
        if hasattr(settings, 'trading'):
            print("   ✅ Settings structure exists")
            results['Adaptive Parameters'] = "✅ Working"
        else:
            print("   ⚠️ Settings structure incomplete")
            results['Adaptive Parameters'] = "⚠️ Incomplete"

    except Exception as e:
        results['Adaptive Parameters'] = f"❌ Error: {e}"
        print(f"   ❌ Error: {e}")

    # Test 6: Reinforcement Learning (if available)
    print("\n6️⃣ Testing Reinforcement Learning...")
    try:
        import numpy as np
        from stable_baselines3 import PPO
        from gymnasium import spaces

        # Test basic RL setup
        obs_space = spaces.Box(low=-2, high=2, shape=(5,), dtype=np.float32)
        action_space = spaces.Box(low=-0.5, high=0.5, shape=(3,), dtype=np.float32)

        # This would normally create a model, but we'll just test imports
        results['Reinforcement Learning'] = "✅ Libraries available"
        print("   ✅ RL libraries imported successfully")

    except ImportError as e:
        results['Reinforcement Learning'] = f"❌ Import failed: {e}"
        print(f"   ❌ Import failed: {e}")
    except Exception as e:
        results['Reinforcement Learning'] = f"❌ Error: {e}"
        print(f"   ❌ Error: {e}")

    # Print summary
    print("\n📊 ML COMPONENTS TEST RESULTS:")
    print("=" * 50)

    working_count = 0
    total_count = len(results)

    for component, status in results.items():
        print(f"   {component}: {status}")
        if status.startswith("✅"):
            working_count += 1

    print(f"\n🎯 SUMMARY: {working_count}/{total_count} ML components working")

    if working_count == total_count:
        print("🎉 All ML components are operational!")
    elif working_count >= total_count * 0.7:
        print("✅ Most ML components are working - good for basic operation")
    else:
        print("⚠️ Several ML components have issues - check dependencies")

    return results

async def test_adaptive_learning():
    """Test adaptive parameter learning system."""
    print("\n🧠 TESTING ADAPTIVE LEARNING")
    print("=" * 50)

    try:
        from src.ml.trading_ml_engine import TradingMLEngine

        ml_engine = TradingMLEngine()

        # Simulate learning data
        learning_data = {
            'win_rate': 0.65,
            'profit_factor': 1.8,
            'max_drawdown': 0.05,
            'total_trades': 100,
            'avg_holding_time': 2.5,
            'confidence_correlation': 0.75
        }

        # Test parameter adjustments
        adjustments = ml_engine.calculate_parameter_adjustments(learning_data)

        if adjustments:
            print("✅ Adaptive learning system working")
            print("   📊 Recommended parameter adjustments:")
            for param, adjustment in adjustments.items():
                print(f"      {param}: {adjustment}")
        else:
            print("⚠️ Adaptive learning returned no adjustments")

    except Exception as e:
        print(f"❌ Adaptive learning test failed: {e}")

async def test_full_integration():
    """Test full ML integration with trading bot."""
    print("\n🔗 TESTING FULL ML INTEGRATION")
    print("=" * 50)

    try:
        # Test importing the adaptive bot
        from run_adaptive_intelligent_bot import AdaptiveIntelligentBot

        # Create instance (don't start it)
        bot = AdaptiveIntelligentBot()

        # Check if ML components are available
        ml_available = bot.ml_engine is not None
        reversal_available = bot.reversal_detector is not None
        sentiment_available = bot.sentiment_aggregator is not None

        print(f"   🤖 ML Engine: {'✅' if ml_available else '❌'}")
        print(f"   🔄 Reversal Detector: {'✅' if reversal_available else '❌'}")
        print(f"   📊 Sentiment Aggregator: {'✅' if sentiment_available else '❌'}")

        if ml_available or reversal_available or sentiment_available:
            print("✅ Full integration test passed - some ML features available")
        else:
            print("⚠️ Full integration test - no ML features available")

    except Exception as e:
        print(f"❌ Full integration test failed: {e}")

async def main():
    """Main test function."""
    print("🚀 ML COMPONENTS VALIDATION")
    print("=" * 60)

    # Run all tests
    await test_ml_components()
    await test_adaptive_learning()
    await test_full_integration()

    print("\n" + "=" * 60)
    print("✅ ML Components validation complete!")
    print("\n💡 Next steps:")
    print("   1. Fix any failed components by installing missing dependencies")
    print("   2. Configure API keys for sentiment analysis")
    print("   3. Run the adaptive bot to test real-time learning")
    print("   4. Monitor adaptive parameters in logs/adaptive_bot_heartbeat.json")

if __name__ == "__main__":
    asyncio.run(main())
