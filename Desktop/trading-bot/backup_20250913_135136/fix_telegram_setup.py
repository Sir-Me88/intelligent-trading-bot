#!/usr/bin/env python3
"""Fix Telegram bot setup with detailed troubleshooting."""

import requests
import json
from pathlib import Path

def test_telegram_bot(token, chat_id):
    """Test Telegram bot with detailed error reporting."""
    print(f"🧪 TESTING TELEGRAM BOT")
    print(f"   Token: {token[:10]}...{token[-10:] if len(token) > 20 else token}")
    print(f"   Chat ID: {chat_id}")
    
    try:
        # Test 1: Check bot info
        print("\n1. Testing bot token...")
        bot_info_url = f"https://api.telegram.org/bot{token}/getMe"
        response = requests.get(bot_info_url, timeout=10)
        
        if response.status_code == 200:
            bot_data = response.json()
            if bot_data.get('ok'):
                bot_name = bot_data['result']['username']
                print(f"   ✅ Bot token valid: @{bot_name}")
            else:
                print(f"   ❌ Bot API error: {bot_data}")
                return False
        else:
            print(f"   ❌ HTTP Error {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        # Test 2: Send test message
        print("\n2. Testing message sending...")
        test_url = f"https://api.telegram.org/bot{token}/sendMessage"
        test_data = {
            'chat_id': chat_id,
            'text': '🚀 Trading Bot Test Message - Setup Successful!'
        }
        
        response = requests.post(test_url, json=test_data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('ok'):
                print("   ✅ Test message sent successfully!")
                print("   📱 Check your Telegram for the test message")
                return True
            else:
                print(f"   ❌ Message send failed: {result}")
                return False
        else:
            print(f"   ❌ HTTP Error {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def get_chat_id_helper(token):
    """Help user get their chat ID."""
    print("\n💬 GETTING YOUR CHAT ID")
    print("="*30)
    
    print("📋 Steps:")
    print("1. Open Telegram and find your bot")
    print("2. Send any message to your bot (like 'hello')")
    print("3. Press Enter here to check for messages...")
    
    input("Press Enter after sending a message to your bot...")
    
    try:
        updates_url = f"https://api.telegram.org/bot{token}/getUpdates"
        response = requests.get(updates_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('ok') and data.get('result'):
                print("\n📨 Found messages:")
                for update in data['result'][-3:]:  # Show last 3 messages
                    if 'message' in update:
                        chat_id = update['message']['chat']['id']
                        text = update['message'].get('text', 'No text')
                        print(f"   Chat ID: {chat_id}")
                        print(f"   Message: {text}")
                        print(f"   From: {update['message']['from'].get('first_name', 'Unknown')}")
                        print()
                
                # Get the most recent chat_id
                if data['result']:
                    latest_chat_id = data['result'][-1]['message']['chat']['id']
                    print(f"🎯 Your Chat ID is: {latest_chat_id}")
                    return str(latest_chat_id)
            else:
                print("❌ No messages found. Make sure you sent a message to your bot.")
                return None
        else:
            print(f"❌ Error getting updates: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def interactive_telegram_setup():
    """Interactive Telegram setup with troubleshooting."""
    print("🚨 TELEGRAM BOT SETUP - STEP BY STEP")
    print("="*50)
    
    print("\n📋 STEP 1: Create Bot")
    print("1. Open Telegram app")
    print("2. Search for '@BotFather'")
    print("3. Send: /newbot")
    print("4. Follow the prompts")
    print("5. Copy the bot token")
    
    token = input("\n🔑 Enter your bot token: ").strip()
    
    if not token:
        print("❌ No token provided")
        return None, None
    
    # Validate token format
    if ':' not in token or len(token) < 20:
        print("❌ Invalid token format. Should be like: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz")
        return None, None
    
    print("\n📋 STEP 2: Get Chat ID")
    chat_id = get_chat_id_helper(token)
    
    if not chat_id:
        chat_id = input("\n💬 Enter your chat ID manually (or press Enter to skip): ").strip()
    
    if token and chat_id:
        print("\n🧪 TESTING CONFIGURATION")
        if test_telegram_bot(token, chat_id):
            return token, chat_id
        else:
            print("❌ Test failed. Please check your configuration.")
            return None, None
    
    return None, None

def update_env_with_telegram(token, chat_id):
    """Update .env file with Telegram credentials."""
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    # Read current file
    with open(env_file, 'r') as f:
        lines = f.readlines()
    
    # Update lines
    updated_lines = []
    token_updated = False
    chat_updated = False
    
    for line in lines:
        if line.startswith('TELEGRAM_BOT_TOKEN='):
            updated_lines.append(f'TELEGRAM_BOT_TOKEN={token}\n')
            token_updated = True
        elif line.startswith('TELEGRAM_CHAT_ID='):
            updated_lines.append(f'TELEGRAM_CHAT_ID={chat_id}\n')
            chat_updated = True
        else:
            updated_lines.append(line)
    
    # Add lines if they don't exist
    if not token_updated:
        updated_lines.append(f'TELEGRAM_BOT_TOKEN={token}\n')
    if not chat_updated:
        updated_lines.append(f'TELEGRAM_CHAT_ID={chat_id}\n')
    
    # Write updated file
    with open(env_file, 'w') as f:
        f.writelines(updated_lines)
    
    print("✅ .env file updated with Telegram credentials")
    return True

def main():
    """Main function."""
    print("🔧 TELEGRAM BOT FIX TOOL")
    print("="*30)
    
    # Check current status
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
            
        if 'TELEGRAM_BOT_TOKEN=' in content and 'TELEGRAM_CHAT_ID=' in content:
            print("📊 Current Telegram configuration found in .env")
            
            # Extract current values
            for line in content.split('\n'):
                if line.startswith('TELEGRAM_BOT_TOKEN='):
                    current_token = line.split('=', 1)[1].strip()
                    if current_token and 'your_' not in current_token:
                        print(f"   Token: {current_token[:10]}...{current_token[-5:]}")
                elif line.startswith('TELEGRAM_CHAT_ID='):
                    current_chat = line.split('=', 1)[1].strip()
                    if current_chat and 'your_' not in current_chat:
                        print(f"   Chat ID: {current_chat}")
            
            test_existing = input("\nTest existing configuration? (y/n): ").strip().lower()
            if test_existing == 'y':
                # Try to test existing config
                try:
                    lines = content.split('\n')
                    token = None
                    chat_id = None
                    
                    for line in lines:
                        if line.startswith('TELEGRAM_BOT_TOKEN='):
                            token = line.split('=', 1)[1].strip()
                        elif line.startswith('TELEGRAM_CHAT_ID='):
                            chat_id = line.split('=', 1)[1].strip()
                    
                    if token and chat_id and 'your_' not in token and 'your_' not in chat_id:
                        if test_telegram_bot(token, chat_id):
                            print("🎉 Existing Telegram configuration works!")
                            return
                        else:
                            print("❌ Existing configuration failed")
                    else:
                        print("❌ Invalid existing configuration")
                except:
                    print("❌ Could not test existing configuration")
    
    # Setup new configuration
    print("\n🔧 Setting up new Telegram configuration...")
    token, chat_id = interactive_telegram_setup()
    
    if token and chat_id:
        if update_env_with_telegram(token, chat_id):
            print("\n🎉 TELEGRAM SETUP COMPLETE!")
            print("✅ Bot token and chat ID saved to .env")
            print("📱 You should have received a test message")
            print("\n🚀 Next: Restart your trading bot to enable notifications")
            print("   Command: python restart_bot.py")
        else:
            print("❌ Failed to update .env file")
    else:
        print("❌ Telegram setup incomplete")

if __name__ == "__main__":
    main()