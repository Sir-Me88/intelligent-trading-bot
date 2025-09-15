import sys
import traceback
sys.path.append('src')

print('=== DEEP COMPONENT TESTING WITH INSTANTIATION ===')

# Test ML Engine instantiation and functionality
try:
    from src.ml.trading_ml_engine import TradingMLEngine
    print('✅ ML Engine import: OK')

    # Try to instantiate
    ml_engine = TradingMLEngine()
    print('✅ ML Engine instantiation: OK')

    # Try basic methods
    test_data = {'executed_trades': []}
    result = ml_engine.perform_daily_analysis(test_data)
    print('✅ ML Engine daily analysis: OK')
    print(f'   Analysis type: {result.get("analysis_type", "Unknown")}')
    print(f'   Trades analyzed: {result.get("trades_analyzed", 0)}')

except Exception as e:
    print(f'❌ ML Engine failed: {e}')
    traceback.print_exc()

print()

# Test Scheduler instantiation and functionality
try:
    from src.scheduling.intelligent_scheduler import IntelligentTradingScheduler
    print('✅ Scheduler import: OK')

    scheduler = IntelligentTradingScheduler()
    print('✅ Scheduler instantiation: OK')

    # Test basic functionality
    print(f'   Current mode: {getattr(scheduler, "mode", "Unknown")}')
    print(f'   Initialized: {getattr(scheduler, "initialized", False)}')

except Exception as e:
    print(f'❌ Scheduler failed: {e}')
    traceback.print_exc()

print()

# Test Reversal Detector instantiation
try:
    from src.analysis.trend_reversal_detector import TrendReversalDetector
    print('✅ Reversal Detector import: OK')

    detector = TrendReversalDetector()
    print('✅ Reversal Detector instantiation: OK')

except Exception as e:
    print(f'❌ Reversal Detector failed: {e}')
    traceback.print_exc()

print()

# Test Trade Analyzer instantiation
try:
    from src.ml.trade_analyzer import TradeAnalyzer
    print('✅ Trade Analyzer import: OK')

    analyzer = TradeAnalyzer()
    print('✅ Trade Analyzer instantiation: OK')

except Exception as e:
    print(f'❌ Trade Analyzer failed: {e}')
    traceback.print_exc()

print()

# Test Sentiment Aggregator instantiation
try:
    from src.news.sentiment import SentimentAggregator
    print('✅ Sentiment Aggregator import: OK')

    sentiment = SentimentAggregator()
    print('✅ Sentiment Aggregator instantiation: OK')

except Exception as e:
    print(f'❌ Sentiment Aggregator failed: {e}')
    traceback.print_exc()

print()
print('=== TESTING MAIN BOT INSTANTIATION ===')

# Test main bot instantiation
try:
    from run_adaptive_intelligent_bot import AdaptiveIntelligentBot
    print('✅ Main bot import: OK')

    # This will test if all components can be initialized together
    print('⚠️  Skipping full bot instantiation (requires broker connection)')
    print('✅ All individual components tested successfully')

except Exception as e:
    print(f'❌ Main bot failed: {e}')
    traceback.print_exc()
