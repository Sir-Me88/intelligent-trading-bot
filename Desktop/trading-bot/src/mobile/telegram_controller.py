"""Telegram-based Android Control Interface for Trading Bot."""

import os
import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime
import json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, 
    CommandHandler, 
    CallbackQueryHandler, 
    MessageHandler,
    filters,
    ContextTypes
)

logger = logging.getLogger(__name__)


class TelegramController:
    """Telegram bot for Android control of trading bot."""
    
    def __init__(self, trading_bot):
        self.trading_bot = trading_bot
        self.telegram_token = os.getenv('TELEGRAM_TOKEN')
        self.authorized_users = self._load_authorized_users()
        self.app = None
        
        if not self.telegram_token:
            logger.warning("⚠️ TELEGRAM_TOKEN not found in environment variables")
            logger.info("To enable Telegram control:")
            logger.info("1. Create a bot with @BotFather on Telegram")
            logger.info("2. Set TELEGRAM_TOKEN in your .env file")
            logger.info("3. Set TELEGRAM_AUTHORIZED_USERS with your user ID")
    
    def _load_authorized_users(self) -> list:
        """Load authorized user IDs from environment."""
        users_str = os.getenv('TELEGRAM_AUTHORIZED_USERS', '')
        if users_str:
            try:
                return [int(user_id.strip()) for user_id in users_str.split(',')]
            except ValueError:
                logger.error("Invalid TELEGRAM_AUTHORIZED_USERS format")
        return []
    
    def _is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized to control the bot."""
        return user_id in self.authorized_users
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        user_id = update.effective_user.id
        
        if not self._is_authorized(user_id):
            await update.message.reply_text(
                "❌ Unauthorized access. Contact the bot administrator."
            )
            logger.warning(f"Unauthorized access attempt from user {user_id}")
            return
        
        # Get bot status
        status = await self._get_bot_status()
        
        keyboard = [
            [InlineKeyboardButton("📊 Status", callback_data="status")],
            [InlineKeyboardButton("💰 Account", callback_data="account")],
            [InlineKeyboardButton("📈 Positions", callback_data="positions")],
            [InlineKeyboardButton("⚡ Controls", callback_data="controls")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        welcome_msg = f"""
🤖 **Trading Bot Control Panel**

**Current Status:** {status['mode']}
**Balance:** ${status['balance']:,.2f}
**Equity:** ${status['equity']:,.2f}
**Open Positions:** {status['positions']}
**Today's Trades:** {status['trades_today']}

Use the buttons below to control your bot:
        """
        
        await update.message.reply_text(
            welcome_msg,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        user_id = update.effective_user.id
        
        if not self._is_authorized(user_id):
            await update.message.reply_text("❌ Unauthorized access.")
            return
        
        status = await self._get_detailed_status()
        
        status_msg = f"""
📊 **Detailed Bot Status**

**🔄 Mode:** {status['mode']}
**⏰ Uptime:** {status['uptime']}
**🔍 Scans:** {status['scans']}
**💹 Trades Executed:** {status['trades']}
**🔄 Reversals Detected:** {status['reversals']}
**🎯 Success Rate:** {status['success_rate']:.1f}%

**💰 Account Info:**
• Balance: ${status['balance']:,.2f}
• Equity: ${status['equity']:,.2f}
• Free Margin: ${status['free_margin']:,.2f}
• Margin Level: {status['margin_level']:.1f}%

**⚙️ Current Settings:**
• Confidence Threshold: {status['confidence_threshold']:.1f}%
• Risk per Trade: {status['risk_per_trade']:.1f}%
• Max Positions: {status['max_positions']}
        """
        
        await update.message.reply_text(status_msg, parse_mode='Markdown')
    
    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle emergency /stop command."""
        user_id = update.effective_user.id
        
        if not self._is_authorized(user_id):
            await update.message.reply_text("❌ Unauthorized access.")
            return
        
        # Confirmation keyboard
        keyboard = [
            [InlineKeyboardButton("🛑 CONFIRM STOP", callback_data="confirm_stop")],
            [InlineKeyboardButton("❌ Cancel", callback_data="cancel_stop")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "⚠️ **EMERGENCY STOP REQUESTED**\n\n"
            "This will immediately stop all trading activities.\n"
            "Are you sure you want to proceed?",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks."""
        query = update.callback_query
        user_id = query.from_user.id
        
        if not self._is_authorized(user_id):
            await query.answer("❌ Unauthorized access.")
            return
        
        await query.answer()
        
        if query.data == "status":
            status = await self._get_detailed_status()
            status_msg = f"""
📊 **Quick Status**
Mode: {status['mode']}
Balance: ${status['balance']:,.2f}
Positions: {status['positions']}
Trades Today: {status['trades_today']}
            """
            await query.edit_message_text(status_msg, parse_mode='Markdown')
            
        elif query.data == "account":
            account = await self._get_account_info()
            account_msg = f"""
💰 **Account Information**

**Balance:** ${account['balance']:,.2f}
**Equity:** ${account['equity']:,.2f}
**Free Margin:** ${account['free_margin']:,.2f}
**Margin Level:** {account['margin_level']:.1f}%
**Profit:** ${account['profit']:,.2f}
**Currency:** {account['currency']}
**Server:** {account['server']}
            """
            await query.edit_message_text(account_msg, parse_mode='Markdown')
            
        elif query.data == "positions":
            positions = await self._get_positions_info()
            await query.edit_message_text(positions, parse_mode='Markdown')
            
        elif query.data == "controls":
            keyboard = [
                [InlineKeyboardButton("▶️ Start Trading", callback_data="start_trading")],
                [InlineKeyboardButton("⏸️ Pause Trading", callback_data="pause_trading")],
                [InlineKeyboardButton("🛑 Emergency Stop", callback_data="emergency_stop")],
                [InlineKeyboardButton("🔄 Restart Bot", callback_data="restart_bot")],
                [InlineKeyboardButton("⬅️ Back", callback_data="back_to_main")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                "⚡ **Bot Controls**\n\nChoose an action:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
        elif query.data == "confirm_stop":
            try:
                # Stop the trading bot
                await self.trading_bot.emergency_stop()
                await query.edit_message_text(
                    "🛑 **EMERGENCY STOP EXECUTED**\n\n"
                    "All trading activities have been stopped.\n"
                    "The bot is now in safe mode.",
                    parse_mode='Markdown'
                )
                logger.info(f"Emergency stop executed by user {user_id}")
            except Exception as e:
                await query.edit_message_text(
                    f"❌ **Error executing stop:** {e}",
                    parse_mode='Markdown'
                )
                
        elif query.data == "cancel_stop":
            await query.edit_message_text("✅ Emergency stop cancelled.")
            
        elif query.data == "start_trading":
            try:
                await self.trading_bot.start_trading()
                await query.edit_message_text("▶️ Trading started successfully!")
            except Exception as e:
                await query.edit_message_text(f"❌ Error starting trading: {e}")
                
        elif query.data == "pause_trading":
            try:
                await self.trading_bot.pause_trading()
                await query.edit_message_text("⏸️ Trading paused successfully!")
            except Exception as e:
                await query.edit_message_text(f"❌ Error pausing trading: {e}")
    
    async def _get_bot_status(self) -> Dict:
        """Get basic bot status."""
        try:
            if hasattr(self.trading_bot, 'get_status'):
                return await self.trading_bot.get_status()
            else:
                # Fallback status
                return {
                    'mode': 'Unknown',
                    'balance': 0,
                    'equity': 0,
                    'positions': 0,
                    'trades_today': 0
                }
        except Exception as e:
            logger.error(f"Error getting bot status: {e}")
            return {
                'mode': 'Error',
                'balance': 0,
                'equity': 0,
                'positions': 0,
                'trades_today': 0
            }
    
    async def _get_detailed_status(self) -> Dict:
        """Get detailed bot status."""
        try:
            # Get basic status
            status = await self._get_bot_status()
            
            # Add detailed information
            if hasattr(self.trading_bot, 'broker_manager'):
                account_info = await self.trading_bot.broker_manager.get_account_info()
                status.update({
                    'balance': account_info.get('balance', 0),
                    'equity': account_info.get('equity', 0),
                    'free_margin': account_info.get('free_margin', 0),
                    'margin_level': account_info.get('margin_level', 0),
                })
            
            # Add bot-specific metrics
            status.update({
                'uptime': self._get_uptime(),
                'scans': getattr(self.trading_bot, 'scan_count', 0),
                'trades': getattr(self.trading_bot, 'trades_executed', 0),
                'reversals': getattr(self.trading_bot, 'reversals_detected', 0),
                'success_rate': self._calculate_success_rate(),
                'confidence_threshold': getattr(self.trading_bot, 'adaptive_params', {}).get('min_confidence', 0.75) * 100,
                'risk_per_trade': 2.0,  # Default value
                'max_positions': 10,  # Default value
                'trades_today': getattr(self.trading_bot, 'daily_trades_count', 0)
            })
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting detailed status: {e}")
            return await self._get_bot_status()
    
    async def _get_account_info(self) -> Dict:
        """Get account information."""
        try:
            if hasattr(self.trading_bot, 'broker_manager'):
                return await self.trading_bot.broker_manager.get_account_info()
            else:
                return {
                    'balance': 0,
                    'equity': 0,
                    'free_margin': 0,
                    'margin_level': 0,
                    'profit': 0,
                    'currency': 'USD',
                    'server': 'Unknown'
                }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {'error': str(e)}
    
    async def _get_positions_info(self) -> str:
        """Get formatted positions information."""
        try:
            if hasattr(self.trading_bot, 'broker_manager'):
                positions = await self.trading_bot.broker_manager.get_positions()
                
                if not positions:
                    return "📈 **Open Positions**\n\nNo open positions."
                
                positions_text = "📈 **Open Positions**\n\n"
                for i, pos in enumerate(positions[:10], 1):  # Limit to 10 positions
                    profit_emoji = "💚" if pos.get('profit', 0) >= 0 else "❤️"
                    positions_text += f"{i}. {pos.get('symbol', 'Unknown')} "
                    positions_text += f"{pos.get('type', 'Unknown').upper()} "
                    positions_text += f"{pos.get('volume', 0):.2f} lots "
                    positions_text += f"{profit_emoji} ${pos.get('profit', 0):.2f}\n"
                
                return positions_text
            else:
                return "📈 **Open Positions**\n\nBot not connected."
                
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return f"❌ Error getting positions: {e}"
    
    def _get_uptime(self) -> str:
        """Calculate bot uptime."""
        try:
            if hasattr(self.trading_bot, 'start_time'):
                uptime = datetime.now() - self.trading_bot.start_time
                hours = uptime.total_seconds() // 3600
                minutes = (uptime.total_seconds() % 3600) // 60
                return f"{int(hours)}h {int(minutes)}m"
            else:
                return "Unknown"
        except:
            return "Unknown"
    
    def _calculate_success_rate(self) -> float:
        """Calculate trading success rate."""
        try:
            if hasattr(self.trading_bot, 'trades_executed') and hasattr(self.trading_bot, 'successful_trades'):
                total = self.trading_bot.trades_executed
                successful = self.trading_bot.successful_trades
                if total > 0:
                    return (successful / total) * 100
            return 0.0
        except:
            return 0.0
    
    async def send_notification(self, message: str, urgent: bool = False):
        """Send notification to all authorized users."""
        if not self.app or not self.authorized_users:
            return
        
        emoji = "🚨" if urgent else "ℹ️"
        formatted_message = f"{emoji} **Trading Bot Notification**\n\n{message}"
        
        for user_id in self.authorized_users:
            try:
                await self.app.bot.send_message(
                    chat_id=user_id,
                    text=formatted_message,
                    parse_mode='Markdown'
                )
            except Exception as e:
                logger.error(f"Failed to send notification to user {user_id}: {e}")
    
    def run(self):
        """Start the Telegram bot."""
        if not self.telegram_token:
            logger.warning("Telegram bot not started - no token provided")
            return
        
        if not self.authorized_users:
            logger.warning("Telegram bot not started - no authorized users")
            return
        
        try:
            # Build application
            self.app = ApplicationBuilder().token(self.telegram_token).build()
            
            # Add handlers
            self.app.add_handler(CommandHandler("start", self.start_command))
            self.app.add_handler(CommandHandler("status", self.status_command))
            self.app.add_handler(CommandHandler("stop", self.stop_command))
            self.app.add_handler(CallbackQueryHandler(self.button_callback))
            
            logger.info("🤖 Telegram controller started")
            logger.info(f"📱 Authorized users: {len(self.authorized_users)}")
            
            # Run the bot
            self.app.run_polling()
            
        except Exception as e:
            logger.error(f"Error starting Telegram bot: {e}")
    
    async def start_async(self):
        """Start the Telegram bot asynchronously."""
        if not self.telegram_token or not self.authorized_users:
            return
        
        try:
            self.app = ApplicationBuilder().token(self.telegram_token).build()
            
            # Add handlers
            self.app.add_handler(CommandHandler("start", self.start_command))
            self.app.add_handler(CommandHandler("status", self.status_command))
            self.app.add_handler(CommandHandler("stop", self.stop_command))
            self.app.add_handler(CallbackQueryHandler(self.button_callback))
            
            logger.info("🤖 Telegram controller started (async)")
            
            # Initialize and start polling
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()
            
        except Exception as e:
            logger.error(f"Error starting Telegram bot async: {e}")
    
    async def stop_async(self):
        """Stop the Telegram bot asynchronously."""
        if self.app:
            try:
                await self.app.updater.stop()
                await self.app.stop()
                await self.app.shutdown()
                logger.info("🤖 Telegram controller stopped")
            except Exception as e:
                logger.error(f"Error stopping Telegram bot: {e}")
