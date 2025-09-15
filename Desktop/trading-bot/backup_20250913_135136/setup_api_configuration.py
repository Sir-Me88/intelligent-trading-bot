#!/usr/bin/env python3
"""Complete API configuration setup for the trading bot."""

import os
import sys
from pathlib import Path
import requests
import json

def display_api_overview():
    """Display overview of available APIs and their benefits."""
    print("🔧 TRADING BOT API CONFIGURATION")
    print("="*60)
    
    print("\n📊 AVAILABLE API INTEGRATIONS:")
    print("="*40)
    
    apis = {
        "📰 News & Economic Data": {
            "FMP_API_KEY": {
                "name": "Financial Modeling Prep",
                "url": "https://financialmodelingprep.com/developer/docs",
                "benefits": ["Economic calendar", "Earnings data", "Market news"],
                "free_tier": "250 calls/day",
                "cost": "Free tier available"
            }
        },
        "📈 Market Sentiment": {
            "NEWS_API_KEY": {
                "name": "NewsAPI",
                "url": "https://newsapi.org/",
                "benefits": ["Real-time news", "Sentiment analysis", "Market events"],
                "free_tier": "1000 calls/day",
                "cost": "Free tier available"
            },
            "TWITTER_BEARER_TOKEN": {
                "name": "Twitter API v2",
                "url": "https://developer.twitter.com/",
                "benefits": ["Social sentiment", "Market buzz", "Breaking news"],
                "free_tier": "500k tweets/month",
                "cost": "Free tier available"
            }
        },
        "📊 Additional Market Data": {
            "ALPHA_VANTAGE_KEY": {
                "name": "Alpha Vantage",
                "url": "https://www.alphavantage.co/",
                "benefits": ["Stock data", "Forex data", "Technical indicators"],
                "free_tier": "25 calls/day",
                "cost": "Free tier available"
            },
            "TWELVE_DATA_KEY": {
                "name": "Twelve Data",
                "url": "https://twelvedata.com/",
                "benefits": ["Real-time data", "Historical data", "Multiple markets"],
                "free_tier": "800 calls/day",
                "cost": "Free tier available"
            }
        },
        "🚨 Notifications": {
            "TELEGRAM_BOT_TOKEN": {
                "name": "Telegram Bot",
                "url": "https://core.telegram.org/bots#botfather",
                "benefits": ["Trade alerts", "Performance updates", "Error notifications"],
                "free_tier": "Unlimited",
                "cost": "Free"
            }
        }
    }
    
    for category, api_list in apis.items():
        print(f"\n{category}")
        print("-" * 30)
        
        for api_key, details in api_list.items():
            print(f"  🔑 {details['name']}")
            print(f"     Benefits: {', '.join(details['benefits'])}")
            print(f"     Free Tier: {details['free_tier']}")
            print(f"     Setup: {details['url']}")
            print()

def check_current_api_status():
    """Check current API configuration status."""
    print("\n🔍 CURRENT API STATUS:")
    print("="*30)
    
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found")
        return {}
    
    # Read current .env
    current_apis = {}
    with open(env_file, 'r') as f:
        lines = f.readlines()
    
    api_keys = [
        'FMP_API_KEY', 'NEWS_API_KEY', 'TWITTER_BEARER_TOKEN',
        'ALPHA_VANTAGE_KEY', 'TWELVE_DATA_KEY', 'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CHAT_ID'
    ]
    
    for line in lines:
        for key in api_keys:
            if line.startswith(f'{key}='):
                value = line.split('=', 1)[1].strip()
                if value and value != f'your_{key.lower()}_here':
                    current_apis[key] = "✅ Configured"
                else:
                    current_apis[key] = "❌ Not configured"
    
    # Display status
    for key in api_keys:
        status = current_apis.get(key, "❌ Not found")
        print(f"  {key}: {status}")
    
    return current_apis

def setup_priority_apis():
    """Setup the most important APIs first."""
    print("\n🎯 PRIORITY API SETUP")
    print("="*30)
    
    print("Let's configure the most impactful APIs first:")
    print("1. 📰 Financial Modeling Prep (Economic Calendar)")
    print("2. 🚨 Telegram Bot (Trade Notifications)")
    print("3. 📈 NewsAPI (Market Sentiment)")
    
    return True

def setup_fmp_api():
    """Setup Financial Modeling Prep API."""
    print("\n📰 FINANCIAL MODELING PREP SETUP")
    print("="*40)
    
    print("🎯 Why FMP API?")
    print("  • Economic calendar events")
    print("  • Earnings announcements")
    print("  • Market-moving news")
    print("  • FREE 250 calls/day")
    
    print("\n📋 Setup Steps:")
    print("1. Go to: https://financialmodelingprep.com/developer/docs")
    print("2. Click 'Get API Key' (top right)")
    print("3. Sign up with email")
    print("4. Verify email and get your API key")
    print("5. Copy the API key")
    
    api_key = input("\n🔑 Enter your FMP API Key (or press Enter to skip): ").strip()
    
    if api_key:
        # Test the API key
        print("🧪 Testing API key...")
        try:
            test_url = f"https://financialmodelingprep.com/api/v3/economic_calendar?apikey={api_key}"
            response = requests.get(test_url, timeout=10)
            
            if response.status_code == 200:
                print("✅ API key is valid!")
                return api_key
            elif response.status_code == 401:
                print("❌ Invalid API key")
                return None
            else:
                print(f"⚠️ API returned status {response.status_code}")
                return api_key  # Might still work
                
        except Exception as e:
            print(f"⚠️ Could not test API key: {e}")
            return api_key  # Assume it's valid
    
    return None

def setup_telegram_bot():
    """Setup Telegram bot for notifications."""
    print("\n🚨 TELEGRAM BOT SETUP")
    print("="*30)
    
    print("🎯 Why Telegram Bot?")
    print("  • Instant trade notifications")
    print("  • Performance updates")
    print("  • Error alerts")
    print("  • 100% FREE")
    
    print("\n📋 Setup Steps:")
    print("1. Open Telegram app")
    print("2. Search for '@BotFather'")
    print("3. Send: /newbot")
    print("4. Choose a name: 'My Trading Bot'")
    print("5. Choose username: 'mytradingbot_[yourname]'")
    print("6. Copy the bot token")
    print("7. Send a message to your bot")
    print("8. Go to: https://api.telegram.org/bot[TOKEN]/getUpdates")
    print("9. Find your chat_id in the response")
    
    bot_token = input("\n🔑 Enter your Telegram Bot Token (or press Enter to skip): ").strip()
    chat_id = None
    
    if bot_token:
        chat_id = input("💬 Enter your Telegram Chat ID (or press Enter to skip): ").strip()
        
        if bot_token and chat_id:
            # Test the bot
            print("🧪 Testing Telegram bot...")
            try:
                test_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                test_data = {
                    'chat_id': chat_id,
                    'text': '🚀 Trading Bot API Test - Configuration Successful!'
                }
                response = requests.post(test_url, json=test_data, timeout=10)
                
                if response.status_code == 200:
                    print("✅ Telegram bot is working! Check your phone.")
                    return bot_token, chat_id
                else:
                    print("❌ Telegram bot test failed")
                    return bot_token, chat_id  # Still save it
                    
            except Exception as e:
                print(f"⚠️ Could not test Telegram bot: {e}")
                return bot_token, chat_id
    
    return bot_token, chat_id

def setup_news_api():
    """Setup NewsAPI for sentiment analysis."""
    print("\n📈 NEWSAPI SETUP")
    print("="*20)
    
    print("🎯 Why NewsAPI?")
    print("  • Real-time market news")
    print("  • Sentiment analysis")
    print("  • FREE 1000 calls/day")
    
    print("\n📋 Setup Steps:")
    print("1. Go to: https://newsapi.org/")
    print("2. Click 'Get API Key'")
    print("3. Sign up with email")
    print("4. Copy your API key")
    
    api_key = input("\n🔑 Enter your NewsAPI Key (or press Enter to skip): ").strip()
    
    if api_key:
        # Test the API key
        print("🧪 Testing API key...")
        try:
            test_url = f"https://newsapi.org/v2/everything?q=forex&apiKey={api_key}&pageSize=1"
            response = requests.get(test_url, timeout=10)
            
            if response.status_code == 200:
                print("✅ NewsAPI key is valid!")
                return api_key
            else:
                print("❌ NewsAPI key test failed")
                return api_key  # Still save it
                
        except Exception as e:
            print(f"⚠️ Could not test NewsAPI key: {e}")
            return api_key
    
    return None

def update_env_file(api_updates):
    """Update the .env file with new API keys."""
    print("\n💾 UPDATING .ENV FILE")
    print("="*25)
    
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    # Read current file
    with open(env_file, 'r') as f:
        lines = f.readlines()
    
    # Update lines
    updated_lines = []
    for line in lines:
        updated = False
        for key, value in api_updates.items():
            if line.startswith(f'{key}=') and value:
                updated_lines.append(f'{key}={value}\n')
                updated = True
                print(f"  ✅ Updated {key}")
                break
        
        if not updated:
            updated_lines.append(line)
    
    # Write updated file
    with open(env_file, 'w') as f:
        f.writelines(updated_lines)
    
    print("✅ .env file updated successfully!")
    return True

def main():
    """Main API configuration function."""
    display_api_overview()
    current_status = check_current_api_status()
    
    print("\n" + "="*60)
    response = input("Do you want to configure APIs now? (y/n): ")
    
    if response.lower() != 'y':
        print("💡 You can run this script anytime to configure APIs")
        return
    
    setup_priority_apis()
    
    # Collect API configurations
    api_updates = {}
    
    # FMP API
    fmp_key = setup_fmp_api()
    if fmp_key:
        api_updates['FMP_API_KEY'] = fmp_key
    
    # Telegram Bot
    telegram_token, telegram_chat = setup_telegram_bot()
    if telegram_token:
        api_updates['TELEGRAM_BOT_TOKEN'] = telegram_token
    if telegram_chat:
        api_updates['TELEGRAM_CHAT_ID'] = telegram_chat
    
    # NewsAPI
    news_key = setup_news_api()
    if news_key:
        api_updates['NEWS_API_KEY'] = news_key
    
    # Update .env file
    if api_updates:
        update_env_file(api_updates)
        
        print("\n🎉 API CONFIGURATION COMPLETE!")
        print("="*40)
        print("✅ Configured APIs:")
        for key in api_updates.keys():
            print(f"  • {key}")
        
        print("\n🚀 NEXT STEPS:")
        print("1. Restart your trading bot: python restart_bot.py")
        print("2. Monitor enhanced performance: python monitor_bot.py")
        print("3. Check logs for API integration: logs/core_trading_bot.log")
        
    else:
        print("\n💡 No APIs configured. You can run this script again anytime.")

if __name__ == "__main__":
    main()