#!/usr/bin/env python3
"""Setup MT5 demo account automatically."""

import sys
import os
from pathlib import Path

def setup_demo_account():
    """Guide through demo account setup."""
    print("🎯 MT5 DEMO ACCOUNT SETUP")
    print("="*40)
    
    print("\n📋 STEP 1: Download MetaTrader 5")
    print("   If you don't have MT5 installed:")
    print("   1. Go to: https://www.metatrader5.com/en/download")
    print("   2. Download and install MT5")
    print("   3. Launch the application")
    
    print("\n📋 STEP 2: Create Demo Account")
    print("   1. Open MetaTrader 5")
    print("   2. Click 'File' -> 'Open an Account'")
    print("   3. Select 'Open a demo account'")
    print("   4. Choose 'MetaQuotes Software Corp.' server")
    print("   5. Fill in your details (any name/email works)")
    print("   6. Choose account type: 'Forex' with $10,000 balance")
    print("   7. Click 'Next' and 'Finish'")
    
    print("\n📋 STEP 3: Get Your Credentials")
    print("   After account creation, note:")
    print("   - Login: (your account number)")
    print("   - Password: (password you set)")
    print("   - Server: (usually 'MetaQuotes-Demo')")
    
    print("\n📋 STEP 4: Update .env File")
    print("   Replace these lines in your .env file:")
    print("   MT5_LOGIN=your_demo_login_number")
    print("   MT5_PASSWORD=your_demo_password")
    print("   MT5_SERVER=MetaQuotes-Demo")
    
    return True

def update_env_file():
    """Interactive .env file update."""
    print("\n🔧 INTERACTIVE .ENV UPDATE")
    print("-"*30)
    
    # Get user input
    login = input("Enter your MT5 Login (account number): ").strip()
    password = input("Enter your MT5 Password: ").strip()
    server = input("Enter your MT5 Server (or press Enter for 'MetaQuotes-Demo'): ").strip()
    
    if not server:
        server = "MetaQuotes-Demo"
    
    if not login or not password:
        print("❌ Login and password are required!")
        return False
    
    # Read current .env file
    env_path = Path('.env')
    if not env_path.exists():
        print("❌ .env file not found!")
        return False
    
    # Update .env file
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    # Replace MT5 lines
    updated_lines = []
    for line in lines:
        if line.startswith('MT5_LOGIN='):
            updated_lines.append(f'MT5_LOGIN={login}\n')
        elif line.startswith('MT5_PASSWORD='):
            updated_lines.append(f'MT5_PASSWORD={password}\n')
        elif line.startswith('MT5_SERVER='):
            updated_lines.append(f'MT5_SERVER={server}\n')
        else:
            updated_lines.append(line)
    
    # Write updated file
    with open(env_path, 'w') as f:
        f.writelines(updated_lines)
    
    print("\n✅ .env file updated successfully!")
    print(f"   📊 Login: {login}")
    print(f"   📊 Server: {server}")
    print(f"   🔒 Password: {'*' * len(password)}")
    
    return True

async def test_connection():
    """Test the MT5 connection."""
    print("\n🔌 TESTING MT5 CONNECTION")
    print("-"*30)
    
    try:
        # Import after .env update
        sys.path.append('src')
        from src.trading.broker_interface import BrokerManager
        
        broker = BrokerManager()
        
        print("1. Connecting to MT5...")
        connection = await broker.initialize()
        
        if connection:
            print("   ✅ Connection successful!")
            
            # Get account info
            account_info = await broker.get_account_info()
            if account_info:
                print(f"   📊 Account: {account_info.get('login', 'N/A')}")
                print(f"   💰 Balance: ${account_info.get('balance', 0):,.2f}")
                print(f"   📈 Equity: ${account_info.get('equity', 0):,.2f}")
                print(f"   🏢 Company: {account_info.get('company', 'N/A')}")
            
            # Test market data
            print("\n2. Testing market data...")
            try:
                # Test if we can get symbol info (basic market data test)
                import MetaTrader5 as mt5
                symbol_info = mt5.symbol_info('EURUSD')
                if symbol_info:
                    print(f"   📊 EURUSD: {symbol_info.bid:.5f}/{symbol_info.ask:.5f}")
                    print(f"   📊 Spread: {symbol_info.spread} points")
                    print("   ✅ Market data working!")
                else:
                    print("   ⚠️  No market data (market might be closed)")
            except Exception as e:
                print(f"   ⚠️  Market data test: {e}")
            
            print("\n🎉 MT5 CONNECTION SUCCESSFUL!")
            print("   Your trading bot is ready for live market data!")
            return True
            
        else:
            print("   ❌ Connection failed!")
            print("   🔧 Please check your credentials and try again")
            return False
            
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        print("   💡 Make sure MT5 is installed and running")
        return False

if __name__ == "__main__":
    import asyncio
    
    setup_demo_account()
    
    print("\n" + "="*40)
    response = input("Do you want to update .env file now? (y/n): ")
    
    if response.lower() == 'y':
        if update_env_file():
            print("\n" + "="*40)
            test_response = input("Test the connection now? (y/n): ")
            
            if test_response.lower() == 'y':
                asyncio.run(test_connection())
        else:
            print("❌ Failed to update .env file")
    else:
        print("💡 Update .env file manually and run the test later")