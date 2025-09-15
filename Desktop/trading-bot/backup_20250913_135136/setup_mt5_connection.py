#!/usr/bin/env python3
"""Setup and test MT5 connection."""

import sys
import os
from pathlib import Path

sys.path.append('src')

def setup_mt5_credentials():
    """Guide user through MT5 setup."""
    print("🔧 MT5 CONNECTION SETUP")
    print("="*40)
    
    print("\n📋 You need the following MT5 credentials:")
    print("   1. MT5 Login (account number)")
    print("   2. MT5 Password") 
    print("   3. MT5 Server (e.g., 'MetaQuotes-Demo')")
    
    print("\n🔍 Current .env configuration:")
    
    # Read current .env
    env_path = Path('.env')
    if env_path.exists():
        with open(env_path, 'r') as f:
            lines = f.readlines()
        
        mt5_lines = [line for line in lines if line.startswith('MT5_')]
        for line in mt5_lines:
            if 'your_mt5' in line:
                print(f"   ❌ {line.strip()} (needs configuration)")
            else:
                print(f"   ✅ {line.strip()}")
    
    print("\n🛠️ To configure MT5:")
    print("   1. Open MetaTrader 5")
    print("   2. Go to File -> Open an Account")
    print("   3. Choose a broker (or use demo account)")
    print("   4. Note your login, password, and server")
    print("   5. Update the .env file with your credentials")
    
    print("\n📝 Example .env configuration:")
    print("   MT5_LOGIN=12345678")
    print("   MT5_PASSWORD=your_password")
    print("   MT5_SERVER=MetaQuotes-Demo")
    
    return True

async def test_mt5_connection():
    """Test MT5 connection with current credentials."""
    print("\n🔌 TESTING MT5 CONNECTION")
    print("-"*30)
    
    try:
        from src.trading.broker_interface import BrokerManager
        
        broker = BrokerManager()
        
        # Test connection
        print("1. Initializing MT5...")
        connection_result = await broker.connect()
        
        if connection_result:
            print("   ✅ MT5 connection successful!")
            
            # Test account info
            account_info = await broker.get_account_info()
            if account_info:
                print(f"   📊 Account: {account_info.get('login', 'N/A')}")
                print(f"   💰 Balance: ${account_info.get('balance', 0):.2f}")
                print(f"   📈 Equity: ${account_info.get('equity', 0):.2f}")
            
            # Test market data
            print("\n2. Testing market data...")
            symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
            
            for symbol in symbols:
                try:
                    tick = await broker.get_tick(symbol)
                    if tick:
                        print(f"   📊 {symbol}: {tick.get('bid', 0):.5f}/{tick.get('ask', 0):.5f}")
                    else:
                        print(f"   ❌ {symbol}: No data")
                except Exception as e:
                    print(f"   ❌ {symbol}: {e}")
            
            return True
        else:
            print("   ❌ MT5 connection failed")
            print("   🔧 Check your credentials in .env file")
            return False
            
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        return False

if __name__ == "__main__":
    import asyncio
    
    setup_mt5_credentials()
    
    print("\n" + "="*40)
    response = input("Do you want to test the connection now? (y/n): ")
    
    if response.lower() == 'y':
        asyncio.run(test_mt5_connection())
    else:
        print("💡 Run this script again after updating .env file")