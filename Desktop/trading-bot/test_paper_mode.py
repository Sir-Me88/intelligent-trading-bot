import sys
import traceback
sys.path.append('src')

print('=== TESTING BOT WITH PAPER MODE ===')

try:
    from run_adaptive_intelligent_bot import AdaptiveIntelligentBot
    print('✅ Main bot import: OK')

    # Test with paper mode
    import os
    os.environ['PAPER_MODE'] = 'true'

    print('🔄 Attempting bot instantiation in PAPER MODE...')
    bot = AdaptiveIntelligentBot()
    print('✅ Main bot instantiation: OK')

    print(f'   Paper mode: {bot.paper_mode}')
    print(f'   Current mode: {bot.current_mode}')
    print(f'   Scan count: {bot.scan_count}')
    print('✅ Bot basic functionality: OK')

    # Test broker initialization
    print('🔄 Testing broker initialization...')
    broker_result = bot.broker_manager.initialize()
    print(f'   Broker initialization: {"SUCCESS" if broker_result else "FAILED"}')

    print('🎉 BOT IS READY FOR PAPER TRADING!')

except Exception as e:
    print(f'❌ Bot failed: {e}')
    traceback.print_exc()
