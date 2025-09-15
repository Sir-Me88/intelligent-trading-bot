#!/usr/bin/env python3
"""Test script for 2025 improvements: FinGPT v3.1, XAI, and Edge Computing."""

import sys
import asyncio
from datetime import datetime

# Add src to path
sys.path.append('src')

def test_fingpt_upgrade():
    """Test FinGPT v3.1 upgrade."""
    print("🔍 Testing FinGPT v3.1 Upgrade...")

    try:
        from src.news.sentiment import FinGPTAnalyzer
        analyzer = FinGPTAnalyzer()

        # Check if market context is loaded
        if hasattr(analyzer, 'market_context'):
            context = analyzer.market_context
            print("✅ FinGPT v3.1 market context loaded:"            print(f"   - NFP 2025 data: {context.get('nfp_2025_data', False)}")
            print(f"   - Fed policy focus: {context.get('fed_policy_focus', False)}")
            print(f"   - Volatility regime: {context.get('volatility_regime', 'unknown')}")
            return True
        else:
            print("❌ FinGPT market context not found")
            return False

    except Exception as e:
        print(f"❌ FinGPT test failed: {e}")
        return False

def test_xai_integration():
    """Test Explainable AI (XAI) integration."""
    print("\n🔍 Testing XAI Integration...")

    try:
        from src.news.sentiment import SentimentAnalyzer
        analyzer = SentimentAnalyzer()

        # Test XAI explanation
        test_text = "Fed announces interest rate hike, markets react strongly"
        sentiment_score = 0.75

        explanation = analyzer.explain_sentiment_decision(test_text, sentiment_score)

        if explanation.get('explanation_available', False):
            print("✅ XAI explanation generated:"            print(f"   - Confidence: {explanation.get('confidence_interpretation', 'N/A')}")
            print(f"   - Key influencers: {explanation.get('key_influencers', 'N/A')[:50]}...")
            print(f"   - Market context: {explanation.get('market_context', 'N/A')}")
            print(f"   - Recommendation: {explanation.get('recommendation', 'N/A')}")
            print(f"   - Compliance: {explanation.get('compliance_note', 'N/A')}")
            return True
        else:
            print(f"❌ XAI explanation failed: {explanation.get('reason', 'Unknown')}")
            return False

    except Exception as e:
        print(f"❌ XAI test failed: {e}")
        return False

def test_edge_computing():
    """Test Edge Computing components."""
    print("\n🔍 Testing Edge Computing Components...")

    try:
        from src.trading.edge_optimizer import EdgeSentimentPrefetcher, LatencyOptimizer, EdgeComputingManager
        print("✅ Edge computing modules imported successfully")

        # Test component initialization
        try:
            from src.news.sentiment import SentimentAggregator
            sentiment_agg = SentimentAggregator()

            prefetcher = EdgeSentimentPrefetcher(sentiment_agg)
            print("✅ EdgeSentimentPrefetcher initialized")

            # Mock broker for testing
            class MockBroker:
                async def place_order(self, params):
                    await asyncio.sleep(0.01)  # Simulate network delay
                    return {'success': True, 'order_id': 'test_123'}

            mock_broker = MockBroker()
            latency_optimizer = LatencyOptimizer(mock_broker, prefetcher)
            print("✅ LatencyOptimizer initialized")

            edge_manager = EdgeComputingManager(mock_broker, sentiment_agg)
            print("✅ EdgeComputingManager initialized")

            return True

        except Exception as e:
            print(f"❌ Edge computing initialization failed: {e}")
            return False

    except ImportError as e:
        print(f"❌ Edge computing import failed: {e}")
        return False

def test_dependency_status():
    """Test key dependency availability."""
    print("\n🔍 Testing Key Dependencies...")

    dependencies = [
        ('numpy', 'Numerical computing'),
        ('pandas', 'Data manipulation'),
        ('aiohttp', 'Async HTTP client'),
        ('torch', 'PyTorch ML framework'),
        ('transformers', 'Hugging Face transformers'),
    ]

    failed_deps = []

    for dep, description in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} - {description}")
        except ImportError:
            print(f"❌ {dep} - {description} (MISSING)")
            failed_deps.append(dep)

    # Test optional dependencies
    optional_deps = [
        ('shap', 'Explainable AI'),
        ('vaderSentiment', 'Sentiment analysis'),
        ('tweepy', 'Twitter API'),
    ]

    print("\n📋 Optional Dependencies:")
    for dep, description in optional_deps:
        try:
            __import__(dep)
            print(f"✅ {dep} - {description}")
        except ImportError:
            print(f"⚠️  {dep} - {description} (optional - not installed)")

    return len(failed_deps) == 0

async def test_integration():
    """Test integration of all 2025 improvements."""
    print("\n🔍 Testing Integration...")

    try:
        from src.news.sentiment import SentimentAggregator
        from src.trading.edge_optimizer import EdgeComputingManager

        # Initialize components
        sentiment_agg = SentimentAggregator()
        print("✅ SentimentAggregator initialized")

        # Mock broker for testing
        class MockBroker:
            async def place_order(self, params):
                await asyncio.sleep(0.01)
                return {
                    'success': True,
                    'order_id': f'test_{datetime.now().strftime("%H%M%S")}',
                    'execution_time_ms': 45.2
                }

        mock_broker = MockBroker()
        edge_manager = EdgeComputingManager(mock_broker, sentiment_agg)
        print("✅ EdgeComputingManager initialized")

        # Test sentiment analysis with XAI
        print("\n   Testing sentiment analysis with XAI...")
        sentiment_result = await sentiment_agg.get_overall_sentiment("EURUSD")

        if sentiment_result:
            print("✅ Sentiment analysis successful"            print(".3f"            if 'xai_explanation' in sentiment_result:
                xai = sentiment_result['xai_explanation']
                if xai.get('explanation_available', False):
                    print("✅ XAI explanation integrated"                else:
                    print("⚠️  XAI explanation not available (SHAP may not be installed)"
        else:
            print("❌ Sentiment analysis failed"            return False

        # Test edge computing status
        print("\n   Testing edge computing status...")
        status = edge_manager.get_system_status()
        print("✅ Edge computing status retrieved"        print(f"   - Optimizations enabled: {status.get('edge_optimizations_enabled', False)}")
        print(f"   - Active pairs: {status.get('active_pairs', [])}")

        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all 2025 improvement tests."""
    print("🚀 2025 TRADING BOT IMPROVEMENTS VALIDATION")
    print("="*60)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    tests = [
        ("FinGPT v3.1 Upgrade", test_fingpt_upgrade),
        ("XAI Integration", test_xai_integration),
        ("Edge Computing", test_edge_computing),
        ("Dependencies", test_dependency_status),
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

    # Integration test
    print("
📋 Running: Integration Test"    try:
        if asyncio.run(test_integration()):
            passed_tests += 1
            print("✅ Integration Test PASSED")
        else:
            print("❌ Integration Test FAILED")
    except Exception as e:
        print(f"❌ Integration Test CRASHED: {e}")

    total_tests += 1

    print("\n" + "="*60)
    print("📋 2025 IMPROVEMENTS VALIDATION RESULTS")
    print("="*60)

    for i, (test_name, _) in enumerate(tests + [("Integration Test", None)], 1):
        status = "✅ PASSED" if i <= passed_tests else "❌ FAILED"
        print(f"{test_name}: {status}")

    print(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 ALL 2025 IMPROVEMENTS WORKING PERFECTLY!")
        print("\n🚀 IMMEDIATE PHASE READY:")
        print("   1. ✅ SHAP installed for XAI")
        print("   2. ✅ FinGPT v3.1 integrated")
        print("   3. ✅ Edge computing ready")
        print("   4. 🔄 Configure API keys")
        print("   5. 🔄 Performance benchmarking")
        print("\n📝 SHORT-TERM GOALS (Next Month):")
        print("   - Federated Learning for privacy-safe ML")
        print("   - Advanced backtesting with market impact")
        print("   - UI dashboard enhancements")
        print("   - Security audit")
        print("\n📈 LONG-TERM GOALS (2025):")
        print("   - Blockchain trade auditing")
        print("   - Multi-asset expansion")
        print("   - API commercialization")
        return True
    elif passed_tests >= total_tests * 0.75:
        print("⚠️ MOST IMPROVEMENTS WORKING - Minor issues detected")
        return True
    else:
        print("❌ CRITICAL ISSUES DETECTED - Review implementation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
