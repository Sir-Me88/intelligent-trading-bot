#!/usr/bin/env python3
"""Verify that all optimizations are properly applied to the main bot files."""

import sys
from pathlib import Path

def verify_main_bot_configuration():
    """Verify the main bot has all our optimizations."""
    print("🔍 VERIFYING MAIN BOT CONFIGURATION")
    print("="*50)
    
    # Check main bot file
    bot_file = Path('run_core_trading_bot.py')
    if not bot_file.exists():
        print("❌ Main bot file not found!")
        return False
    
    with open(bot_file, 'r', encoding='utf-8') as f:
        bot_content = f.read()
    
    print("📊 CHECKING OPTIMIZED PARAMETERS:")
    
    # Check trading parameters
    if "'min_confidence': 0.78" in bot_content:
        print("   ✅ min_confidence: 0.78 (optimized)")
    else:
        print("   ❌ min_confidence not set to 0.78")
    
    if "'min_rr_ratio': 1.5" in bot_content:
        print("   ✅ min_rr_ratio: 1.5 (optimized)")
    else:
        print("   ❌ min_rr_ratio not set to 1.5")
    
    if "'max_volatility': 0.002" in bot_content:
        print("   ✅ max_volatility: 0.002 (conservative)")
    else:
        print("   ❌ max_volatility not properly set")
    
    # Check broker interface
    broker_file = Path('src/trading/broker_interface.py')
    if broker_file.exists():
        with open(broker_file, 'r', encoding='utf-8') as f:
            broker_content = f.read()
        
        if "max_spread_pips = 20" in broker_content:
            print("   ✅ max_spread_pips: 20 (balanced)")
        else:
            print("   ❌ max_spread_pips not set to 20")
    
    # Check .env file
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            env_content = f.read()
        
        print("\n📊 CHECKING API CONFIGURATION:")
        
        api_keys = [
            ('MT5_LOGIN', 'MT5 Connection'),
            ('FMP_API_KEY', 'Financial Modeling Prep'),
            ('NEWS_API_KEY', 'NewsAPI'),
            ('ALPHA_VANTAGE_KEY', 'Alpha Vantage'),
            ('TWELVE_DATA_KEY', 'Twelve Data'),
            ('TWITTER_BEARER_TOKEN', 'Twitter API'),
            ('TELEGRAM_BOT_TOKEN', 'Telegram Bot'),
            ('TELEGRAM_CHAT_ID', 'Telegram Chat')
        ]
        
        configured_apis = 0
        for key, name in api_keys:
            if f"{key}=" in env_content:
                # Check if it has a real value (not placeholder)
                for line in env_content.split('\n'):
                    if line.startswith(f'{key}='):
                        value = line.split('=', 1)[1].strip()
                        if value and 'your_' not in value.lower():
                            print(f"   ✅ {name}")
                            configured_apis += 1
                            break
                        else:
                            print(f"   ❌ {name} (placeholder value)")
                            break
            else:
                print(f"   ❌ {name} (not found)")
        
        print(f"\n📊 API Summary: {configured_apis}/{len(api_keys)} configured")
    
    print(f"\n🎯 MAIN BOT STATUS:")
    if "'min_confidence': 0.78" in bot_content and "'min_rr_ratio': 1.5" in bot_content:
        print("   ✅ Main bot has optimized parameters")
        return True
    else:
        print("   ❌ Main bot missing optimizations")
        return False

def identify_files_for_cleanup():
    """Identify which files can be safely removed."""
    print(f"\n🧹 FILE CLEANUP ANALYSIS")
    print("="*35)
    
    # Essential files (KEEP)
    essential_files = {
        # Core bot files
        'run_core_trading_bot.py': 'Main trading bot',
        'start_bot.py': 'Bot launcher',
        'requirements.txt': 'Dependencies',
        '.env': 'Configuration',
        '.env.example': 'Config template',
        '.gitignore': 'Git ignore rules',
        
        # Management tools
        'monitor_bot.py': 'Bot monitoring',
        'restart_bot.py': 'Bot restart',
        'stop_bot.py': 'Bot stop',
        
        # Documentation
        'README.md': 'Main documentation',
        'QUICK_START_GUIDE.md': 'User guide',
        
        # Source code
        'src/': 'Core source code directory'
    }
    
    # Testing/debugging files (CAN REMOVE)
    cleanup_files = {
        'test_basic_functionality.py': 'Basic testing',
        'test_comprehensive.py': 'Comprehensive testing', 
        'test_working_systems.py': 'System testing',
        'run_simple_live_test.py': 'Live testing',
        'analyze_signal_rejection.py': 'Signal analysis',
        'investigate_signal_quality.py': 'Quality investigation',
        'quick_signal_test.py': 'Quick testing',
        'setup_mt5_connection.py': 'MT5 setup (one-time use)',
        'setup_demo_account.py': 'Demo setup (one-time use)',
        'setup_api_configuration.py': 'API setup (one-time use)',
        'setup_forex_apis.py': 'Forex API setup (one-time use)',
        'setup_remaining_apis.py': 'Additional API setup (one-time use)',
        'fix_telegram_setup.py': 'Telegram troubleshooting',
        'verify_main_bot_config.py': 'This verification script',
        'OVERNIGHT_CHECKLIST.md': 'Temporary checklist',
        'TODAY_SUMMARY.md': 'Session summary'
    }
    
    # Check which files actually exist
    print("📁 ESSENTIAL FILES (KEEP):")
    for file_path, description in essential_files.items():
        if Path(file_path).exists():
            print(f"   ✅ {file_path} - {description}")
        else:
            print(f"   ❌ {file_path} - {description} (MISSING!)")
    
    print(f"\n🗑️ FILES FOR CLEANUP (CAN REMOVE):")
    cleanup_count = 0
    for file_path, description in cleanup_files.items():
        if Path(file_path).exists():
            print(f"   🗑️ {file_path} - {description}")
            cleanup_count += 1
        else:
            print(f"   ➖ {file_path} - {description} (not found)")
    
    print(f"\n📊 CLEANUP SUMMARY:")
    print(f"   Files to remove: {cleanup_count}")
    print(f"   Estimated space saved: ~{cleanup_count * 50}KB")
    
    return list(cleanup_files.keys())

if __name__ == "__main__":
    print("🚀 TRADING BOT DEPLOYMENT PREPARATION")
    print("="*50)
    
    # Verify main configuration
    config_ok = verify_main_bot_configuration()
    
    # Identify cleanup files
    cleanup_list = identify_files_for_cleanup()
    
    print(f"\n🎯 DEPLOYMENT READINESS:")
    if config_ok:
        print("   ✅ Main bot properly configured")
        print("   ✅ Ready for cleanup and deployment")
    else:
        print("   ❌ Main bot needs configuration fixes")
        print("   ⚠️ Fix configuration before deployment")
    
    print(f"\n💡 NEXT STEPS:")
    print("   1. Review the cleanup list above")
    print("   2. Run cleanup script to remove test files")
    print("   3. Create deployment package")
    print("   4. Deploy to VPS")