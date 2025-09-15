#!/usr/bin/env python3
"""Stop the background trading bot."""

import subprocess
import time

def stop_trading_bot():
    """Stop the trading bot process."""
    print("🛑 STOPPING TRADING BOT")
    print("="*30)
    
    try:
        # Kill python processes (be careful - this kills ALL python processes)
        result = subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Trading bot stopped successfully")
        else:
            print("⚠️  No Python processes found to stop")
            
        # Wait a moment
        time.sleep(2)
        
        # Verify it's stopped
        check_result = subprocess.run(['tasklist', '/fi', 'imagename eq python.exe'], 
                                    capture_output=True, text=True)
        
        if 'python.exe' not in check_result.stdout:
            print("✅ Confirmed: No Python processes running")
        else:
            print("⚠️  Some Python processes may still be running")
            
    except Exception as e:
        print(f"❌ Error stopping bot: {e}")

if __name__ == "__main__":
    stop_trading_bot()