#!/usr/bin/env python3
"""Monitor the background trading bot."""

import json
import os
import time
from datetime import datetime
from pathlib import Path

def monitor_trading_bot():
    """Monitor the trading bot status."""
    print("🔍 TRADING BOT MONITOR")
    print("="*50)
    
    heartbeat_file = Path("logs/core_bot_heartbeat.json")
    log_file = Path("logs/core_trading_bot.log")
    
    if not heartbeat_file.exists():
        print("❌ Bot heartbeat file not found")
        print("💡 Bot may not be running or hasn't started yet")
        return
    
    try:
        # Read heartbeat
        with open(heartbeat_file, 'r') as f:
            heartbeat = json.load(f)
        
        # Display status
        timestamp = heartbeat.get('timestamp', 'Unknown')
        mode = heartbeat.get('mode', 'Unknown')
        scan_count = heartbeat.get('scan_count', 0)
        trades_executed = heartbeat.get('trades_executed', 0)
        signals_analyzed = heartbeat.get('signals_analyzed', 0)
        signals_rejected = heartbeat.get('signals_rejected', 0)
        active_pairs = heartbeat.get('active_pairs', [])
        
        print(f"📊 BOT STATUS (Last Update: {timestamp})")
        print(f"   🔄 Mode: {mode}")
        print(f"   📈 Scan Cycles: {scan_count}")
        print(f"   💰 Trades Executed: {trades_executed}")
        print(f"   🎯 Signals Analyzed: {signals_analyzed}")
        print(f"   ❌ Signals Rejected: {signals_rejected}")
        print(f"   📊 Active Pairs: {', '.join(active_pairs)}")
        
        # Calculate success rate
        if signals_analyzed > 0:
            success_rate = (trades_executed / signals_analyzed) * 100
            print(f"   📊 Signal Success Rate: {success_rate:.1f}%")
        
        # Check if bot is recent (last 2 minutes)
        try:
            last_update = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            now = datetime.now(last_update.tzinfo)
            time_diff = (now - last_update).total_seconds()
            
            if time_diff < 120:  # 2 minutes
                print(f"   ✅ Bot Status: ACTIVE (last seen {time_diff:.0f}s ago)")
            else:
                print(f"   ⚠️  Bot Status: INACTIVE (last seen {time_diff:.0f}s ago)")
        except:
            print(f"   ❓ Bot Status: UNKNOWN")
        
        print("\n📋 RECENT LOG ENTRIES:")
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Show last 5 lines
                for line in lines[-5:]:
                    print(f"   {line.strip()}")
        else:
            print("   ❌ Log file not found")
            
    except Exception as e:
        print(f"❌ Error reading bot status: {e}")

def check_bot_process():
    """Check if bot process is running."""
    print("\n🔍 PROCESS CHECK:")
    
    # Check for python processes
    import subprocess
    try:
        result = subprocess.run(['tasklist', '/fi', 'imagename eq python.exe'], 
                              capture_output=True, text=True)
        
        if 'python.exe' in result.stdout:
            lines = result.stdout.split('\n')
            python_processes = [line for line in lines if 'python.exe' in line]
            print(f"   ✅ Found {len(python_processes)} Python process(es)")
            for proc in python_processes:
                if proc.strip():
                    parts = proc.split()
                    if len(parts) >= 2:
                        print(f"      PID: {parts[1]}")
        else:
            print("   ❌ No Python processes found")
            print("   💡 Bot may have stopped")
            
    except Exception as e:
        print(f"   ❌ Error checking processes: {e}")

if __name__ == "__main__":
    monitor_trading_bot()
    check_bot_process()
    
    print("\n" + "="*50)
    print("💡 MONITORING COMMANDS:")
    print("   python monitor_bot.py          - Check bot status")
    print("   python stop_bot.py             - Stop the bot")
    print("   python restart_bot.py          - Restart the bot")
    print("="*50)