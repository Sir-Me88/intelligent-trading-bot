#!/usr/bin/env python3
"""
Simple Telegram notification system for trading bot alerts.

This script provides a standalone notification system that can be integrated
with the trading bot to send alerts via Telegram.
"""

import os
import asyncio
import logging
from typing import Optional
from telegram import Bot
from telegram.error import TelegramError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class TradingBotNotifier:
    """Simple Telegram notification system for trading alerts."""

    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_AUTHORIZED_USERS', '').split(',')[0].strip() if os.getenv('TELEGRAM_AUTHORIZED_USERS') else None
        self.bot = None

        if self.bot_token and self.chat_id:
            self.bot = Bot(token=self.bot_token)
            logger.info("✅ Telegram notifier initialized")
        else:
            logger.warning("⚠️ Telegram credentials not found - notifications disabled")

    async def send_message(self, message: str, urgent: bool = False) -> bool:
        """Send a message via Telegram."""
        if not self.bot or not self.chat_id:
            logger.warning("Telegram not configured - message not sent")
            return False

        try:
            emoji = "🚨" if urgent else "ℹ️"
            formatted_message = f"{emoji} **Trading Bot Alert**\n\n{message}"

            await self.bot.send_message(
                chat_id=self.chat_id,
                text=formatted_message,
                parse_mode='Markdown',
                disable_notification=not urgent
            )

            logger.info(f"✅ Telegram message sent: {message[:50]}...")
            return True

        except TelegramError as e:
            logger.error(f"❌ Telegram error: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error sending Telegram message: {e}")
            return False

    async def send_trade_alert(self, symbol: str, action: str, price: float, ticket: Optional[int] = None) -> bool:
        """Send trade execution alert."""
        message = f"""
**TRADE EXECUTED** 🎯

**Symbol:** {symbol}
**Action:** {action.upper()}
**Price:** {price:.5f}
**Ticket:** {ticket or 'N/A'}
**Time:** {asyncio.get_event_loop().time()}
        """.strip()

        return await self.send_message(message, urgent=True)

    async def send_profit_alert(self, symbol: str, profit: float, ticket: int) -> bool:
        """Send profit/loss alert."""
        emoji = "💚" if profit > 0 else "❤️"
        message = f"""
**TRADE CLOSED** {emoji}

**Symbol:** {symbol}
**Profit:** ${profit:.2f}
**Ticket:** {ticket}
        """.strip()

        return await self.send_message(message, urgent=(profit < -50))

    async def send_system_alert(self, alert_type: str, message: str) -> bool:
        """Send system status alert."""
        emojis = {
            'START': '🚀',
            'STOP': '🛑',
            'ERROR': '❌',
            'WARNING': '⚠️',
            'INFO': 'ℹ️'
        }

        emoji = emojis.get(alert_type.upper(), '📢')
        formatted_message = f"{emoji} **SYSTEM {alert_type.upper()}**\n\n{message}"

        return await self.send_message(formatted_message, urgent=(alert_type.upper() in ['ERROR', 'STOP']))

    async def send_daily_summary(self, stats: dict) -> bool:
        """Send daily trading summary."""
        message = f"""
**📊 DAILY TRADING SUMMARY**

**Trades Executed:** {stats.get('trades_today', 0)}
**Win Rate:** {stats.get('win_rate', 0):.1%}
**Total P&L:** ${stats.get('total_pnl', 0):.2f}
**Best Trade:** ${stats.get('best_trade', 0):.2f}
**Worst Trade:** ${stats.get('worst_trade', 0):.2f}

**Active Positions:** {stats.get('active_positions', 0)}
**Account Balance:** ${stats.get('balance', 0):.2f}
        """.strip()

        return await self.send_message(message, urgent=False)

async def test_notifications():
    """Test the notification system."""
    print("🧪 TESTING TELEGRAM NOTIFICATIONS")
    print("=" * 50)

    notifier = TradingBotNotifier()

    if not notifier.bot:
        print("❌ Telegram not configured - skipping tests")
        print("\nTo enable Telegram notifications:")
        print("1. Message @BotFather on Telegram")
        print("2. Create a bot and get the token")
        print("3. Get your chat ID from @userinfobot")
        print("4. Set TELEGRAM_TOKEN and TELEGRAM_AUTHORIZED_USERS in .env")
        return

    print("✅ Telegram configured - testing notifications...")

    # Test basic message
    success = await notifier.send_message("🧪 Test notification from trading bot!")
    print(f"Basic message: {'✅' if success else '❌'}")

    # Test trade alert
    success = await notifier.send_trade_alert("EURUSD", "BUY", 1.0850, 12345)
    print(f"Trade alert: {'✅' if success else '❌'}")

    # Test profit alert
    success = await notifier.send_profit_alert("EURUSD", 25.50, 12345)
    print(f"Profit alert: {'✅' if success else '❌'}")

    # Test system alert
    success = await notifier.send_system_alert("INFO", "System is running normally")
    print(f"System alert: {'✅' if success else '❌'}")

    print("\n✅ Telegram notification tests complete!")

async def main():
    """Main function for testing."""
    await test_notifications()

if __name__ == "__main__":
    asyncio.run(main())
