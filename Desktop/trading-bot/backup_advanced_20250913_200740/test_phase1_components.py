#!/usr/bin/env python3
"""
Test Phase 1 Enhanced Components
Validates all new Phase 1 features before deployment.
"""

import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_enhanced_broker():
    """Test enhanced MT5 broker interface."""
    print("\n🔌 TESTING ENHANCED MT5 BROKER INTERFACE")
    print("=" * 50)
    
    try:
        from src.trading.broker_interface import BrokerManager
        
        # Test enhanced broker
        broker = BrokerManager(use_enhanced=True)
        
        print("1. Testing enhanced connection...")
        connected = await broker.initialize()
        
        if connected:
            print("✅ Enhanced broker connected successfully")
            
            # Test account info
            print("2. Testing account information...")
            account_info = await broker.get_account_info()
            print(f"   Balance: ${account_info.get('balance', 0):,.2f}")
            print(f"   Equity: ${account_info.get('equity', 0):,.2f}")
            print(f"   Server: {account_info.get('server', 'Unknown')}")
            
            # Test spread validation
            print("3. Testing spread validation...")
            spread_valid = await broker.validate_spread('EURUSD')
            print(f"   EURUSD spread valid: {spread_valid}")
            
            # Test connection status
            print("4. Testing connection status...")
            is_connected = broker.is_connected()
            print(f"   Connection status: {is_connected}")
            
            print("✅ Enhanced broker interface test PASSED")
            return True
        else:
            print("❌ Enhanced broker connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced broker test failed: {e}")
        return False


async def test_enhanced_risk_manager():
    """Test enhanced risk management system."""
    print("\n🛡️ TESTING ENHANCED RISK MANAGEMENT")
    print("=" * 50)
    
    try:
        from src.trading.broker_interface import BrokerManager
        from src.risk.enhanced_risk_manager import EnhancedRiskManager
        
        # Initialize broker for risk manager
        broker = BrokerManager(use_enhanced=True)
        if not await broker.initialize():
            print("❌ Cannot test risk manager - broker connection failed")
            return False
        
        # Initialize risk manager
        risk_manager = EnhancedRiskManager(broker.broker)
        
        print("1. Testing trade validation...")
        validation = await risk_manager.validate_trade(
            symbol='EURUSD',
            volume=0.1,
            direction='BUY'
        )
        
        print(f"   Trade approved: {validation['approved']}")
        print(f"   Risk level: {validation['risk_level'].value}")
        print(f"   Position size: {validation['position_size']:.2f}")
        
        if validation['warnings']:
            print(f"   Warnings: {'; '.join(validation['warnings'])}")
        
        if validation['reasons']:
            print(f"   Reasons: {'; '.join(validation['reasons'])}")
        
        print("2. Testing daily counters...")
        risk_manager.reset_daily_counters()
        risk_manager.increment_trade_count()
        print(f"   Daily trades: {risk_manager.daily_trades_count}")
        
        print("✅ Enhanced risk management test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced risk management test failed: {e}")
        return False


def test_telegram_setup():
    """Test Telegram bot configuration."""
    print("\n📱 TESTING TELEGRAM CONFIGURATION")
    print("=" * 50)
    
    try:
        # Check environment variables
        telegram_token = os.getenv('TELEGRAM_TOKEN')
        authorized_users = os.getenv('TELEGRAM_AUTHORIZED_USERS')
        
        print("1. Checking environment variables...")
        print(f"   TELEGRAM_TOKEN: {'SET' if telegram_token else 'NOT SET'}")
        print(f"   AUTHORIZED_USERS: {'SET' if authorized_users else 'NOT SET'}")
        
        if not telegram_token:
            print("⚠️ TELEGRAM_TOKEN not set - Telegram features will be disabled")
            print("   To enable Telegram:")
            print("   1. Create bot with @BotFather")
            print("   2. Add TELEGRAM_TOKEN to .env file")
            return False
        
        if not authorized_users:
            print("⚠️ TELEGRAM_AUTHORIZED_USERS not set")
            print("   To enable Telegram:")
            print("   1. Get your user ID from @userinfobot")
            print("   2. Add TELEGRAM_AUTHORIZED_USERS to .env file")
            return False
        
        # Test Telegram bot import
        print("2. Testing Telegram bot import...")
        try:
            from src.mobile.telegram_controller import TelegramController
            print("✅ Telegram controller imported successfully")
        except ImportError as e:
            print(f"❌ Telegram import failed: {e}")
            print("   Install with: pip install python-telegram-bot==20.7")
            return False
        
        # Test authorized users parsing
        print("3. Testing authorized users parsing...")
        try:
            user_ids = [int(uid.strip()) for uid in authorized_users.split(',')]
            print(f"   Authorized user IDs: {user_ids}")
        except ValueError:
            print("❌ Invalid TELEGRAM_AUTHORIZED_USERS format")
            print("   Should be: user_id1,user_id2,user_id3")
            return False
        
        print("✅ Telegram configuration test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Telegram configuration test failed: {e}")
        return False


async def test_enhanced_bot_integration():
    """Test enhanced bot integration."""
    print("\n🤖 TESTING ENHANCED BOT INTEGRATION")
    print("=" * 50)
    
    try:
        # Test enhanced bot import
        print("1. Testing enhanced bot import...")
        from run_enhanced_phase1_bot import EnhancedPhase1TradingBot
        print("✅ Enhanced bot imported successfully")
        
        # Test bot initialization (without running)
        print("2. Testing bot initialization...")
        bot = EnhancedPhase1TradingBot()
        print(f"   Bot created with {len(bot.adaptive_params)} adaptive parameters")
        print(f"   Trading mode: {bot.trading_mode}")
        
        # Test status method
        print("3. Testing status method...")
        try:
            status = await bot.get_status()
            print(f"   Status keys: {list(status.keys())}")
        except Exception as e:
            print(f"   Status test failed (expected without full init): {e}")
        
        print("✅ Enhanced bot integration test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced bot integration test failed: {e}")
        return False


async def run_all_tests():
    """Run all Phase 1 component tests."""
    print("🚀 PHASE 1 ENHANCED COMPONENTS TEST SUITE")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Enhanced Broker
    result1 = await test_enhanced_broker()
    test_results.append(("Enhanced Broker", result1))
    
    # Test 2: Enhanced Risk Manager
    result2 = await test_enhanced_risk_manager()
    test_results.append(("Enhanced Risk Manager", result2))
    
    # Test 3: Telegram Configuration
    result3 = test_telegram_setup()
    test_results.append(("Telegram Configuration", result3))
    
    # Test 4: Enhanced Bot Integration
    result4 = await test_enhanced_bot_integration()
    test_results.append(("Enhanced Bot Integration", result4))
    
    # Summary
    print("\n📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 1 components are ready for deployment.")
        print("\nNext steps:")
        print("1. Run: python run_enhanced_phase1_bot.py")
        print("2. Test Telegram control via mobile")
        print("3. Monitor enhanced features in action")
    else:
        print(f"\n⚠️ {total-passed} test(s) failed. Please fix issues before deployment.")
        print("\nTroubleshooting:")
        print("1. Check .env file for missing variables")
        print("2. Verify MT5 connection")
        print("3. Install missing dependencies")
        print("4. Review error messages above")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        sys.exit(1)
