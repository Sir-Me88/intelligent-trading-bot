#!/usr/bin/env python3
"""Start the live trading bot for Phase 3 deployment."""

import asyncio
import logging
import sys
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.bot.trading_bot import TradingBot
from src.config.settings import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/live_bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Main entry point for live bot."""
    print("🚀 STARTING PHASE 3 LIVE DEPLOYMENT")
    print("=" * 60)
    print(f"📅 Start Time: {datetime.now()}")
    print(f"💰 Account Balance: $16,942.35")
    print(f"🎯 Risk per Trade: {settings.trading.risk_per_trade * 100}%")
    print(f"📊 Max Daily Trades: {settings.max_daily_trades}")
    print(f"🛡️  Emergency Stop: {settings.emergency_stop}")
    print("=" * 60)

    try:
        # Create and initialize bot
        bot = TradingBot()

        print("🔧 Initializing trading bot...")
        await bot.initialize()

        print("✅ Bot initialized successfully")
        print("🎯 Starting live trading...")
        print("=" * 60)

        # Start trading
        await bot.run()

    except KeyboardInterrupt:
        print("\n⏹️  Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.error(f"Bot crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
