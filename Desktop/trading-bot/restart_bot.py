#!/usr/bin/env python3
"""Restart the background trading bot."""

import subprocess
import time

def restart_trading_bot():
    """Restart the trading bot."""
    print("🔄 RESTARTING TRADING BOT")
    print("="*30)
    
    # Stop existing bot
    print("1. Stopping existing bot...")
    try:
        subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
                      capture_output=True, text=True)
        print("   ✅ Stopped existing processes")
    except:
        print("   ⚠️  No processes to stop")
    
    # Wait a moment
    time.sleep(3)
    
    # Start new bot
    print("2. Starting new bot...")
    try:
        process = subprocess.Popen(['python', 'run_core_trading_bot.py'],
                                 creationflags=subprocess.CREATE_NEW_CONSOLE)
        print(f"   ✅ Bot started with PID: {process.pid}")
        print("   🚀 Trading bot is now running in background")
        
    except Exception as e:
        print(f"   ❌ Error starting bot: {e}")

if __name__ == "__main__":
    restart_trading_bot()