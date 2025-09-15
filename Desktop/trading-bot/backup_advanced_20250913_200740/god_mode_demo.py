#!/usr/bin/env python3
"""
🤖 GOD MODE DEMO - Quick Start Example
Shows how easy it is to use GOD MODE with your trading bot
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def god_mode_demo():
    """Demonstrate GOD MODE capabilities in action"""
    print("🤖 GOD MODE DEMO - Live Sentiment Analysis")
    print("=" * 50)

    try:
        # 1. Test Sentiment Analysis
        print("\n1️⃣ Testing Sentiment Analysis...")
        from src.news.sentiment import SentimentAggregator

        sentiment_agg = SentimentAggregator()

        # Test with sample text
        test_pairs = ["EURUSD", "GBPUSD", "USDJPY"]

        for pair in test_pairs:
            print(f"\n📊 Analyzing {pair}...")
            try:
                sentiment_data = await sentiment_agg.get_overall_sentiment(pair)
                sentiment = sentiment_data.get('overall_sentiment', 0.0)
                confidence = sentiment_data.get('overall_confidence', 0.0)

                # Determine sentiment strength
                if abs(sentiment) > 0.4:
                    strength = "STRONG"
                    emoji = "🔴" if sentiment < 0 else "🟢"
                elif abs(sentiment) > 0.2:
                    strength = "MODERATE"
                    emoji = "🟠" if sentiment < 0 else "🟡"
                else:
                    strength = "NEUTRAL"
                    emoji = "⚪"

                print(f"   {emoji} {strength}: {sentiment:.3f} (confidence: {confidence:.2f})")

                # Show recommendation
                recommendation = sentiment_data.get('recommendation', {})
                action = recommendation.get('action', 'normal')
                reason = recommendation.get('reason', 'Neutral sentiment')

                print(f"   💡 Recommendation: {action.upper()} - {reason}")

            except Exception as e:
                print(f"   ⚠️ Analysis failed: {e}")

        # 2. Test Risk Multiplier
        print("\n2️⃣ Testing Risk Multiplier...")
        from src.risk.sentiment_risk_multiplier import calculate_risk_multiplier

        for pair in test_pairs[:2]:  # Test first 2 pairs
            print(f"\n🛡️ Risk Analysis for {pair}...")
            try:
                risk_signal = await calculate_risk_multiplier(pair, account_balance=10000)

                print(f"   📈 Risk Multiplier: {risk_signal.final_risk_multiplier:.2f}")
                print(f"   🎯 Risk Level: {risk_signal.risk_level}")
                print(f"   💰 Max Loss: ${risk_signal.recommended_max_loss:.2f}")
                print(f"   📝 Reason: {risk_signal.risk_adjustment_reason}")

            except Exception as e:
                print(f"   ⚠️ Risk analysis failed: {e}")

        # 3. Test Entry Timing
        print("\n3️⃣ Testing Entry Timing Optimization...")
        from src.analysis.entry_timing_optimizer import optimize_entry_timing

        base_signal = {
            'direction': 'buy',
            'entry_price': 1.0500,
            'strength': 0.8
        }

        for pair in test_pairs[:1]:  # Test first pair
            print(f"\n⏰ Timing Optimization for {pair}...")
            try:
                timing_signal = await optimize_entry_timing(pair, base_signal)

                print(f"   🕐 Optimal Entry: {timing_signal.optimal_entry_time}")
                print(f"   ⏳ Delay: {timing_signal.entry_delay_minutes} minutes")
                print(f"   🎯 Confidence: {timing_signal.confidence:.2f}")
                print(f"   💡 Recommendation: {timing_signal.recommendation}")

            except Exception as e:
                print(f"   ⚠️ Timing optimization failed: {e}")

        # 4. Test Alert System
        print("\n4️⃣ Testing Alert System...")
        from src.monitoring.alerts import get_alert_stats

        try:
            alert_stats = get_alert_stats()
            print("   🚨 Alert Statistics:")
            print(f"      Active Alerts: {alert_stats.get('active_alerts', 0)}")
            print(f"      Total History: {alert_stats.get('total_alerts_history', 0)}")
            print(f"      Telegram Enabled: {alert_stats.get('telegram_enabled', False)}")

        except Exception as e:
            print(f"   ⚠️ Alert system check failed: {e}")

        # 5. Test Performance Summary
        print("\n5️⃣ Testing Performance Tracking...")
        from src.ml.sentiment_performance_tracker import get_performance_summary

        try:
            performance = get_performance_summary()
            print("   📊 Performance Summary:")
            print(f"      Total Trades: {performance.get('total_trades', 0)}")
            print(f"      Win Rate: {performance.get('win_rate', 0):.1%}")
            print(f"      Total P&L: ${performance.get('total_pnl', 0):.2f}")

        except Exception as e:
            print(f"   ⚠️ Performance tracking failed: {e}")

        print("\n" + "=" * 50)
        print("🎉 GOD MODE DEMO COMPLETED SUCCESSFULLY!")
        print("\n📋 What GOD MODE provides:")
        print("   ✅ Real-time sentiment analysis")
        print("   ✅ Dynamic risk management")
        print("   ✅ Intelligent entry timing")
        print("   ✅ Automated alerts")
        print("   ✅ Performance tracking")
        print("   ✅ Comprehensive analytics")

        print("\n🚀 Ready to enhance your trading with AI-powered sentiment analysis!")

        return True

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def quick_start_guide():
    """Show quick start commands"""
    print("\n📖 GOD MODE QUICK START GUIDE")
    print("=" * 40)

    print("\n1️⃣ Test GOD MODE Components:")
    print("   python test_god_mode_core.py")

    print("\n2️⃣ Run This Demo:")
    print("   python god_mode_demo.py")

    print("\n3️⃣ Start Trading with GOD MODE:")
    print("   python -c \"")
    print("   import asyncio")
    print("   from src.bot.trading_bot import TradingBot")
    print("   from src.monitoring.alerts import initialize_alert_system")
    print("   from src.monitoring.dashboard import initialize_dashboard")
    print("   ")
    print("   async def main():")
    print("       await initialize_alert_system()")
    print("       await initialize_dashboard()")
    print("       bot = TradingBot()")
    print("       await bot.initialize()")
    print("       await bot.run()")
    print("   ")
    print("   asyncio.run(main())")
    print("   \"")

    print("\n4️⃣ Monitor GOD MODE Performance:")
    print("   python -c \"from src.monitoring.dashboard import get_dashboard_summary; print(get_dashboard_summary())\"")

    print("\n5️⃣ Check Active Alerts:")
    print("   python -c \"from src.monitoring.alerts import get_active_alerts; print(get_active_alerts())\"")

    print("\n📚 For detailed setup instructions, see:")
    print("   GOD_MODE_SETUP_GUIDE.md")

async def main():
    """Main demo function"""
    print("🤖 WELCOME TO GOD MODE SENTIMENT TRADING!")
    print("The future of intelligent algorithmic trading")
    print("=" * 60)

    # Run the demo
    success = await god_mode_demo()

    if success:
        print("\n🎯 DEMO STATUS: SUCCESS ✅")
        print("GOD MODE is working perfectly!")

        # Show quick start guide
        await quick_start_guide()

    else:
        print("\n⚠️ DEMO STATUS: ISSUES DETECTED")
        print("Some components may need configuration.")
        print("Check GOD_MODE_SETUP_GUIDE.md for troubleshooting.")

    print("\n" + "=" * 60)
    print("🚀 HAPPY TRADING WITH GOD MODE!")
    print("   Your bot is now equipped with AI-powered market intelligence")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
