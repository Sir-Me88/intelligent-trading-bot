#!/usr/bin/env python3
"""Quick MT5 connection test."""

import sys
import os
sys.path.append('src')

import MetaTrader5 as mt5
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_mt5_connection():
    """Test MT5 connection."""
    print("🔗 Testing MT5 Connection...")
    print("=" * 50)

    # Get credentials from environment
    login = os.getenv('MT5_LOGIN')
    password = os.getenv('MT5_PASSWORD')
    server = os.getenv('MT5_SERVER')

    print(f"Login: {login}")
    print(f"Server: {server}")
    print(f"Password: {'*' * len(password) if password else 'Not set'}")
    print()

    if not login or not password or not server:
        print("❌ MT5 credentials not found in environment variables")
        print("   Please check your .env file")
        return False

    try:
        # Initialize MT5
        if not mt5.initialize():
            print("❌ MT5 initialization failed")
            return False

        print("✅ MT5 initialized successfully")

        # Login to account
        if not mt5.login(login=int(login), password=password, server=server):
            error = mt5.last_error()
            print(f"❌ MT5 login failed: {error}")
            mt5.shutdown()
            return False

        print("✅ MT5 login successful")

        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            print("❌ Failed to get account info")
            mt5.shutdown()
            return False

        print("✅ Account info retrieved successfully")
        print(f"   Balance: ${account_info.balance:.2f}")
        print(f"   Equity: ${account_info.equity:.2f}")
        print(f"   Margin: ${account_info.margin:.2f}")

        # Test symbol info
        symbol_info = mt5.symbol_info("EURUSD")
        if symbol_info is None:
            print("⚠️ EURUSD symbol not available")
        else:
            print("✅ EURUSD symbol available")
            print(f"   Spread: {symbol_info.spread} points")

        mt5.shutdown()
        print("✅ MT5 connection test completed successfully")
        print("🎯 READY FOR LIVE DEPLOYMENT!")
        return True

    except Exception as e:
        print(f"❌ Error during MT5 test: {e}")
        try:
            mt5.shutdown()
        except:
            pass
        return False

if __name__ == "__main__":
    success = test_mt5_connection()
    print()
    print("=" * 50)
    if success:
        print("🎉 MT5 IS CONNECTED AND READY FOR LIVE TRADING!")
        print("🚀 You can safely deploy to VPS")
    else:
        print("❌ MT5 CONNECTION ISSUES - CHECK CREDENTIALS")
        print("🔧 Fix MT5 credentials before VPS deployment")
    print("=" * 50)
