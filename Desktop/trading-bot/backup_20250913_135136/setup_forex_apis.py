#!/usr/bin/env python3
"""Setup the top 3 forex trading APIs: Alpha Vantage, Twelve Data, and Twitter."""

import requests
from pathlib import Path

def setup_alpha_vantage():
    """Setup Alpha Vantage API."""
    print("📊 ALPHA VANTAGE SETUP")
    print("="*30)
    
    print("🎯 Benefits for Forex Trading:")
    print("  • Real-time forex rates validation")
    print("  • Technical indicators (RSI, MACD, etc.)")
    print("  • Market data backup source")
    print("  • Currency strength analysis")
    print("  • FREE 25 calls/day (perfect for validation)")
    
    print("\n📋 Quick Setup (2 minutes):")
    print("1. Go to: https://www.alphavantage.co/")
    print("2. Click 'Get your free API key today'")
    print("3. Fill out: Name, Email, Organization (can be 'Personal')")
    print("4. Check your email for the API key")
    
    api_key = input("\n🔑 Enter your Alpha Vantage API Key (or press Enter to skip): ").strip()
    
    if api_key:
        print("🧪 Testing API key with EURUSD data...")
        try:
            test_url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=EUR&to_currency=USD&apikey={api_key}"
            response = requests.get(test_url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if 'Realtime Currency Exchange Rate' in data:
                    rate_data = data['Realtime Currency Exchange Rate']
                    rate = rate_data.get('5. Exchange Rate', 'N/A')
                    print(f"✅ Alpha Vantage API working! EURUSD: {rate}")
                    return api_key
                elif 'Error Message' in data:
                    print(f"❌ API Error: {data['Error Message']}")
                    return api_key  # Still save it
                elif 'Note' in data:
                    print(f"⚠️ API Limit: {data['Note']}")
                    print("✅ API key is valid (hit rate limit)")
                    return api_key
                else:
                    print("⚠️ Unexpected response, but saving key")
                    return api_key
            else:
                print(f"❌ HTTP Error {response.status_code}")
                return api_key
                
        except Exception as e:
            print(f"⚠️ Could not test API key: {e}")
            print("✅ Saving key anyway")
            return api_key
    
    return None

def setup_twelve_data():
    """Setup Twelve Data API."""
    print("\n📊 TWELVE DATA SETUP")
    print("="*25)
    
    print("🎯 Benefits for Forex Trading:")
    print("  • Real-time forex quotes")
    print("  • Multiple timeframes (1min to 1month)")
    print("  • High-quality data source")
    print("  • Technical indicators")
    print("  • FREE 800 calls/day (excellent for active trading)")
    
    print("\n📋 Quick Setup (2 minutes):")
    print("1. Go to: https://twelvedata.com/")
    print("2. Click 'Get free API key'")
    print("3. Sign up with email")
    print("4. Verify email and copy your API key")
    
    api_key = input("\n🔑 Enter your Twelve Data API Key (or press Enter to skip): ").strip()
    
    if api_key:
        print("🧪 Testing API key with EURUSD quote...")
        try:
            test_url = f"https://api.twelvedata.com/quote?symbol=EURUSD&apikey={api_key}"
            response = requests.get(test_url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if 'symbol' in data and data['symbol'] == 'EURUSD':
                    price = data.get('close', 'N/A')
                    print(f"✅ Twelve Data API working! EURUSD: {price}")
                    return api_key
                elif 'message' in data:
                    print(f"❌ API Error: {data['message']}")
                    return api_key  # Still save it
                elif 'code' in data and data['code'] == 429:
                    print("⚠️ Rate limit reached, but API key is valid")
                    return api_key
                else:
                    print("⚠️ Unexpected response, but saving key")
                    return api_key
            else:
                print(f"❌ HTTP Error {response.status_code}")
                return api_key
                
        except Exception as e:
            print(f"⚠️ Could not test API key: {e}")
            print("✅ Saving key anyway")
            return api_key
    
    return None

def setup_twitter_api():
    """Setup Twitter API for market sentiment."""
    print("\n🐦 TWITTER API SETUP")
    print("="*25)
    
    print("🎯 Benefits for Forex Trading:")
    print("  • Real-time market sentiment")
    print("  • Breaking news detection")
    print("  • Central bank communications")
    print("  • Market buzz and trends")
    print("  • FREE 500k tweets/month")
    
    print("\n📋 Setup Process (5-10 minutes):")
    print("1. Go to: https://developer.twitter.com/")
    print("2. Click 'Sign up for free account'")
    print("3. Apply for developer access (usually instant approval)")
    print("4. Create a new app/project")
    print("5. Generate Bearer Token")
    print("6. Copy the Bearer Token")
    
    print("\n💡 Note: Twitter API setup is more involved but very powerful")
    print("   You can skip this for now and add it later if needed")
    
    bearer_token = input("\n🔑 Enter your Twitter Bearer Token (or press Enter to skip): ").strip()
    
    if bearer_token:
        print("🧪 Testing Twitter API...")
        try:
            headers = {'Authorization': f'Bearer {bearer_token}'}
            test_url = "https://api.twitter.com/2/tweets/search/recent?query=forex&max_results=10"
            response = requests.get(test_url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    tweet_count = len(data['data'])
                    print(f"✅ Twitter API working! Found {tweet_count} recent forex tweets")
                    return bearer_token
                else:
                    print("⚠️ Unexpected response format")
                    return bearer_token
            elif response.status_code == 401:
                print("❌ Invalid Bearer Token")
                return None
            elif response.status_code == 429:
                print("⚠️ Rate limit reached, but token is valid")
                return bearer_token
            else:
                print(f"❌ HTTP Error {response.status_code}")
                return bearer_token
                
        except Exception as e:
            print(f"⚠️ Could not test Twitter API: {e}")
            print("✅ Saving token anyway")
            return bearer_token
    
    return None

def update_env_file(api_updates):
    """Update .env file with new API keys."""
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    print("\n💾 UPDATING .ENV FILE")
    print("="*25)
    
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
    print("🚀 FOREX TRADING API SETUP")
    print("="*40)
    
    print("📊 Setting up the top 3 APIs for forex trading:")
    print("1. 📊 Alpha Vantage - Market data validation")
    print("2. 📊 Twelve Data - Real-time forex data")
    print("3. 🐦 Twitter API - Market sentiment")
    
    print("\n💡 These APIs will supercharge your trading bot with:")
    print("  • Enhanced market data accuracy")
    print("  • Real-time sentiment analysis")
    print("  • Better signal validation")
    print("  • Reduced false positives")
    
    api_updates = {}
    
    # Setup APIs
    alpha_key = setup_alpha_vantage()
    if alpha_key:
        api_updates['ALPHA_VANTAGE_KEY'] = alpha_key
    
    twelve_key = setup_twelve_data()
    if twelve_key:
        api_updates['TWELVE_DATA_KEY'] = twelve_key
    
    twitter_token = setup_twitter_api()
    if twitter_token:
        api_updates['TWITTER_BEARER_TOKEN'] = twitter_token
    
    # Update .env file
    if api_updates:
        if update_env_file(api_updates):
            print("\n🎉 FOREX API SETUP COMPLETE!")
            print("="*40)
            print("✅ Successfully configured:")
            for key in api_updates.keys():
                api_names = {
                    'ALPHA_VANTAGE_KEY': 'Alpha Vantage (Market Data)',
                    'TWELVE_DATA_KEY': 'Twelve Data (Real-time Forex)',
                    'TWITTER_BEARER_TOKEN': 'Twitter API (Market Sentiment)'
                }
                print(f"  • {api_names.get(key, key)}")
            
            print(f"\n📊 Total APIs configured: {len(api_updates)}/3")
            
            print("\n🚀 NEXT STEPS:")
            print("1. Restart your trading bot: python restart_bot.py")
            print("2. Monitor enhanced performance: python monitor_bot.py")
            print("3. Your bot now has premium market intelligence!")
            
            print("\n💡 WHAT'S NEW:")
            if 'ALPHA_VANTAGE_KEY' in api_updates:
                print("  • Market data validation and backup")
            if 'TWELVE_DATA_KEY' in api_updates:
                print("  • High-quality real-time forex data")
            if 'TWITTER_BEARER_TOKEN' in api_updates:
                print("  • Real-time market sentiment analysis")
            
        else:
            print("❌ Failed to update .env file")
    else:
        print("\n💡 No new APIs configured.")
        print("You can run this script again anytime to add more APIs!")

if __name__ == "__main__":
    main()