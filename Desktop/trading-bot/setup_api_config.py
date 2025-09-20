#!/usr/bin/env python3
"""
API Configuration Setup Script for Trading Bot.

This script helps configure all API keys and credentials needed for the trading bot.
Run with --test flag to validate existing configuration.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv, set_key
import requests
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load existing environment variables
load_dotenv()

class APIConfigurator:
    """Handles API configuration and validation."""

    def __init__(self):
        self.env_file = Path('.env')
        self.config_status = {}

    def setup_mt5_credentials(self):
        """Setup MT5 broker credentials."""
        print("\n🔧 MT5 BROKER CONFIGURATION (REQUIRED)")
        print("=" * 50)

        current_login = os.getenv('MT5_LOGIN', '')
        current_password = os.getenv('MT5_PASSWORD', '')
        current_server = os.getenv('MT5_SERVER', '')

        if current_login and current_password and current_server:
            print("✅ MT5 credentials already configured")
            response = input("Update MT5 credentials? (y/N): ").lower().strip()
            if response != 'y':
                return True

        print("\n📝 Enter your MT5 credentials:")
        print("   (Get these from your MetaTrader 5 platform)")

        login = input(f"MT5 Login [{current_login}]: ").strip() or current_login
        password = input(f"MT5 Password [{'*' * len(current_password) if current_password else ''}]: ").strip() or current_password
        server = input(f"MT5 Server [{current_server}]: ").strip() or current_server

        if not login or not password or not server:
            print("❌ All MT5 credentials are required")
            return False

        # Save to .env file
        self._set_env_var('MT5_LOGIN', login)
        self._set_env_var('MT5_PASSWORD', password)
        self._set_env_var('MT5_SERVER', server)

        print("✅ MT5 credentials configured")
        return True

    def setup_fmp_api(self):
        """Setup Financial Modeling Prep API."""
        print("\n🔧 FINANCIAL MODELING PREP API (REQUIRED)")
        print("=" * 50)

        current_key = os.getenv('FMP_API_KEY', '')

        if current_key:
            print("✅ FMP API key already configured")
            response = input("Update FMP API key? (y/N): ").lower().strip()
            if response != 'y':
                return True

        print("\n📝 Get your FREE API key from:")
        print("   https://financialmodelingprep.com/")
        print("   1. Sign up for free account")
        print("   2. Get API key from dashboard")

        api_key = input(f"FMP API Key [{'*' * 8 if current_key else ''}]: ").strip() or current_key

        if not api_key:
            print("❌ FMP API key is required")
            return False

        # Test the API key
        if self._test_fmp_api(api_key):
            self._set_env_var('FMP_API_KEY', api_key)
            print("✅ FMP API configured and tested")
            return True
        else:
            print("❌ FMP API key is invalid")
            return False

    def setup_twelve_data_api(self):
        """Setup Twelve Data API (optional)."""
        print("\n🔧 TWELVE DATA API (OPTIONAL)")
        print("=" * 50)

        current_key = os.getenv('TWELVE_DATA_KEY', '')

        if current_key:
            print("✅ Twelve Data API key already configured")
            response = input("Update Twelve Data API key? (y/N): ").lower().strip()
            if response != 'y':
                return True

        print("\n📝 Get your FREE API key from:")
        print("   https://twelvedata.com/")
        print("   Free tier: 800 requests/day")

        response = input("Configure Twelve Data API? (y/N): ").lower().strip()
        if response != 'y':
            print("⏭️ Skipping Twelve Data API configuration")
            return True

        api_key = input("Twelve Data API Key: ").strip()

        if api_key and self._test_twelve_data_api(api_key):
            self._set_env_var('TWELVE_DATA_KEY', api_key)
            print("✅ Twelve Data API configured and tested")
            return True
        elif api_key:
            print("❌ Twelve Data API key is invalid")
            return False
        else:
            print("⏭️ Twelve Data API not configured")
            return True

    def setup_telegram_bot(self):
        """Setup Telegram bot for notifications."""
        print("\n🔧 TELEGRAM BOT (OPTIONAL)")
        print("=" * 50)

        current_token = os.getenv('TELEGRAM_TOKEN', '')
        current_users = os.getenv('TELEGRAM_AUTHORIZED_USERS', '')

        if current_token and current_users:
            print("✅ Telegram bot already configured")
            response = input("Update Telegram configuration? (y/N): ").lower().strip()
            if response != 'y':
                return True

        print("\n📝 Setup Telegram bot for notifications:")
        print("   1. Message @BotFather on Telegram")
        print("   2. Send /newbot and follow instructions")
        print("   3. Copy the bot token")
        print("   4. Get your user ID from @userinfobot")

        response = input("Configure Telegram bot? (y/N): ").lower().strip()
        if response != 'y':
            print("⏭️ Skipping Telegram bot configuration")
            return True

        token = input("Telegram Bot Token: ").strip()
        users = input("Authorized User IDs (comma-separated): ").strip()

        if token and users:
            # Basic validation
            if len(token.split(':')) == 2 and all(uid.strip().isdigit() for uid in users.split(',')):
                self._set_env_var('TELEGRAM_TOKEN', token)
                self._set_env_var('TELEGRAM_AUTHORIZED_USERS', users)
                print("✅ Telegram bot configured")
                return True
            else:
                print("❌ Invalid Telegram configuration")
                return False
        else:
            print("⏭️ Telegram bot not configured")
            return True

    def setup_eventregistry_api(self):
        """Setup EventRegistry API for news sentiment."""
        print("\n🔧 EVENTREGISTRY API (OPTIONAL)")
        print("=" * 50)

        current_key = os.getenv('EVENTREGISTRY_API_KEY', '')

        if current_key:
            print("✅ EventRegistry API key already configured")
            response = input("Update EventRegistry API key? (y/N): ").lower().strip()
            if response != 'y':
                return True

        print("\n📝 Get your FREE API key from:")
        print("   https://eventregistry.org/")
        print("   Free tier: 10,000 requests/month")

        response = input("Configure EventRegistry API? (y/N): ").lower().strip()
        if response != 'y':
            print("⏭️ Skipping EventRegistry API configuration")
            return True

        api_key = input("EventRegistry API Key: ").strip()

        if api_key:
            self._set_env_var('EVENTREGISTRY_API_KEY', api_key)
            print("✅ EventRegistry API configured")
            return True
        else:
            print("⏭️ EventRegistry API not configured")
            return True

    def setup_twitter_api(self):
        """Setup Twitter API for sentiment analysis."""
        print("\n🔧 TWITTER API (OPTIONAL)")
        print("=" * 50)

        current_token = os.getenv('TWITTER_BEARER_TOKEN', '')

        if current_token:
            print("✅ Twitter API already configured")
            response = input("Update Twitter API key? (y/N): ").lower().strip()
            if response != 'y':
                return True

        print("\n📝 Get your Twitter Bearer Token from:")
        print("   https://developer.twitter.com/")
        print("   1. Create a project/app")
        print("   2. Get Bearer Token from app settings")
        print("   Free tier: 1,500 tweets/month")

        response = input("Configure Twitter API? (y/N): ").lower().strip()
        if response != 'y':
            print("⏭️ Skipping Twitter API configuration")
            return True

        bearer_token = input("Twitter Bearer Token: ").strip()

        if bearer_token:
            self._set_env_var('TWITTER_BEARER_TOKEN', bearer_token)
            print("✅ Twitter API configured")
            return True
        else:
            print("⏭️ Twitter API not configured")
            return True

    def test_configuration(self):
        """Test all configured APIs."""
        print("\n🧪 TESTING API CONFIGURATION")
        print("=" * 50)

        results = {}

        # Test MT5 connection
        print("Testing MT5 connection...")
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                login = os.getenv('MT5_LOGIN')
                password = os.getenv('MT5_PASSWORD')
                server = os.getenv('MT5_SERVER')

                if login and password and server:
                    if mt5.login(int(login), password, server):
                        results['MT5'] = "✅ Connected"
                    else:
                        results['MT5'] = "❌ Login failed"
                else:
                    results['MT5'] = "❌ Credentials missing"
                mt5.shutdown()
            else:
                results['MT5'] = "❌ MT5 not initialized"
        except ImportError:
            results['MT5'] = "❌ MT5 not installed"
        except Exception as e:
            results['MT5'] = f"❌ Error: {e}"

        # Test FMP API
        fmp_key = os.getenv('FMP_API_KEY')
        if fmp_key:
            results['FMP'] = "✅ Configured" if self._test_fmp_api(fmp_key) else "❌ Invalid key"
        else:
            results['FMP'] = "❌ Not configured"

        # Test Twelve Data API
        twelve_key = os.getenv('TWELVE_DATA_KEY')
        if twelve_key:
            results['Twelve Data'] = "✅ Configured" if self._test_twelve_data_api(twelve_key) else "❌ Invalid key"
        else:
            results['Twelve Data'] = "⏭️ Not configured"

        # Test Telegram
        telegram_token = os.getenv('TELEGRAM_TOKEN')
        telegram_users = os.getenv('TELEGRAM_AUTHORIZED_USERS')
        if telegram_token and telegram_users:
            results['Telegram'] = "✅ Configured"
        else:
            results['Telegram'] = "⏭️ Not configured"

        # Test EventRegistry
        er_key = os.getenv('EVENTREGISTRY_API_KEY')
        if er_key:
            results['EventRegistry'] = "✅ Configured"
        else:
            results['EventRegistry'] = "⏭️ Not configured"

        # Test Twitter
        twitter_token = os.getenv('TWITTER_BEARER_TOKEN')
        if twitter_token:
            results['Twitter'] = "✅ Configured"
        else:
            results['Twitter'] = "⏭️ Not configured"

        # Print results
        print("\n📊 API CONFIGURATION STATUS:")
        for api, status in results.items():
            print(f"   {api}: {status}")

        return results

    def _set_env_var(self, key: str, value: str):
        """Set environment variable in .env file."""
        if not self.env_file.exists():
            self.env_file.touch()

        set_key(str(self.env_file), key, value)
        os.environ[key] = value

    def _test_fmp_api(self, api_key: str) -> bool:
        """Test FMP API key."""
        try:
            url = f"https://financialmodelingprep.com/api/v3/economic_calendar?apikey={api_key}"
            response = requests.get(url, timeout=10)
            return response.status_code == 200 and 'economicCalendar' in response.text
        except:
            return False

    def _test_twelve_data_api(self, api_key: str) -> bool:
        """Test Twelve Data API key."""
        try:
            url = f"https://api.twelvedata.com/time_series?symbol=EUR/USD&interval=1min&apikey={api_key}"
            response = requests.get(url, timeout=10)
            return response.status_code == 200 and 'values' in response.text
        except:
            return False

    def run_setup(self):
        """Run the complete setup process."""
        print("🚀 TRADING BOT API CONFIGURATION")
        print("=" * 50)
        print("This script will help you configure all APIs for your trading bot.")
        print("Required APIs are marked with (REQUIRED)")
        print("Optional APIs can be skipped")

        success = True

        # Required APIs
        success &= self.setup_mt5_credentials()
        success &= self.setup_fmp_api()

        # Optional APIs
        success &= self.setup_twelve_data_api()
        success &= self.setup_telegram_bot()
        success &= self.setup_eventregistry_api()
        success &= self.setup_twitter_api()

        if success:
            print("\n🎉 API CONFIGURATION COMPLETE!")
            print("Run 'python setup_api_config.py --test' to validate your configuration")
        else:
            print("\n❌ API CONFIGURATION INCOMPLETE")
            print("Some required APIs failed to configure")

        return success


def main():
    parser = argparse.ArgumentParser(description='Trading Bot API Configuration')
    parser.add_argument('--test', action='store_true', help='Test existing API configuration')
    args = parser.parse_args()

    configurator = APIConfigurator()

    if args.test:
        configurator.test_configuration()
    else:
        configurator.run_setup()


if __name__ == "__main__":
    main()
