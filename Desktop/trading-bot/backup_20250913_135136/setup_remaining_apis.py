#!/usr/bin/env python3
"""Setup remaining high-priority APIs quickly."""

import requests
from pathlib import Path

def setup_alpha_vantage():
    """Setup Alpha Vantage API."""
    print("📊 ALPHA VANTAGE SETUP")
    print("="*30)
    
    print("🎯 Benefits:")
    print("  • Additional forex data validation")
    print("  • Technical indicators")
    print("  • Market data backup")
    print("  • FREE 25 calls/day")
    
    print("\n📋 Quick Setup:")
    print("1. Go to: https://www.alphavantage.co/")
    print("2. Click 'Get your free API key today'")
    print("3. Fill out the simple form")
    print("4. Check your email for the API key")
    
    api_key = input("\n🔑 Enter your Alpha Vantage API Key (or press Enter to skip): ").strip()
    
    if api_key:
        # Test the API key
        print("🧪 Testing API key...")
        try:
            test_url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=USD&to_currency=EUR&apikey={api_key}"
            response = requests.get(test_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'Realtime Currency Exchange Rate' in data:
                    print("✅ Alpha Vantage API key is valid!")
                    return api_key
                elif 'Error Message' in data:
                    print(f"❌ API Error: {data['Error Message']}")
                    return api_key  # Still save it
                else:
                    print("⚠️ Unexpected response format")
                    return api_key
            else:
                print(f"❌ HTTP Error {response.status_code}")
                return api_key
                
        except Exception as e:
            print(f"⚠️ Could not test API key: {e}")
            return api_key
    
    return None

def setup_twelve_data():
    """Setup Twelve Data API."""
    print("\n📊 TWELVE DATA SETUP")
    print("="*25)
    
    print("🎯 Benefits:")
    print("  • Real-time forex data")
    print("  • Multiple timeframes")
    print("  • Reliable data source")
    print("  • FREE 800 calls/day")
    
    print("\n📋 Quick Setup:")
    print("1. Go to: https://twelvedata.com/")
    print("2. Click 'Get free API key'")
    print("3. Sign up with email")
    print("4. Verify email and get your key")
    
    api_key = input("\n🔑 Enter your Twelve Data API Key (or press Enter to skip): ").strip()
    
    if api_key:
        # Test the API key
        print("🧪 Testing API key...")
        try:
            test_url = f"https://api.twelvedata.com/quote?symbol=EURUSD&apikey={api_key}"
            response = requests.get(test_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'symbol' in data and data['symbol'] == 'EURUSD':
                    print("✅ Twelve Data API key is valid!")
                    return api_key
                elif 'message' in data:
                    print(f"❌ API Error: {data['message']}")
                    return api_key  # Still save it
                else:
                    print("⚠️ Unexpected response format")
                    return api_key
            else:
                print(f"❌ HTTP Error {response.status_code}")
                return api_key
                
        except Exception as e:
            print(f"⚠️ Could not test API key: {e}")
            return api_key
    
    return None

def update_env_file(api_updates):
    """Update .env file with new API keys."""
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    # Read current file
    with open(env_file, 'r') as f:
        lines = f.readlines()
    
    # Update or add lines
    updated_lines = []
    keys_updated = set()
    
    for line in lines:
        updated = False
        for key, value in api_updates.items():
            if line.startswith(f'{key}=') and value:
                updated_lines.append(f'{key}={value}\n')
                keys_updated.add(key)
                updated = True
                print(f"  ✅ Updated {key}")
                break
        
        if not updated:
            updated_lines.append(line)
    
    # Add new keys that weren't found
    for key, value in api_updates.items():
        if key not in keys_updated and value:
            updated_lines.append(f'{key}={value}\n')
            print(f"  ✅ Added {key}")
    
    # Write updated file
    with open(env_file, 'w') as f:
        f.writelines(updated_lines)
    
    return True

def main():
    """Main setup function."""
    print("🚀 SETUP REMAINING HIGH-PRIORITY APIS")
    print("="*50)
    
    print("📊 We'll configure the most impactful remaining APIs:")
    print("1. Alpha Vantage (Market data validation)")
    print("2. Twelve Data (Reliable data source)")
    
    api_updates = {}
    
    # Alpha Vantage
    alpha_key = setup_alpha_vantage()
    if alpha_key:
        api_updates['ALPHA_VANTAGE_KEY'] = alpha_key
    
    # Twelve Data
    twelve_key = setup_twelve_data()
    if twelve_key:
        api_updates['TWELVE_DATA_KEY'] = twelve_key
    
    # Update .env file
    if api_updates:
        print("\n💾 UPDATING .ENV FILE")
        print("="*25)
        
        if update_env_file(api_updates):
            print("\n🎉 API CONFIGURATION COMPLETE!")
            print("="*40)
            print("✅ New APIs configured:")
            for key in api_updates.keys():
                print(f"  • {key}")
            
            print("\n🚀 NEXT STEPS:")
            print("1. Restart trading bot: python restart_bot.py")
            print("2. Monitor enhanced performance: python monitor_bot.py")
            print("3. Your bot now has even better data sources!")
            
        else:
            print("❌ Failed to update .env file")
    else:
        print("\n💡 No new APIs configured.")
        print("Your bot is already well-configured with the essential APIs!")

if __name__ == "__main__":
    main()