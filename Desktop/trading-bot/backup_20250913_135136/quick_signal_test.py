#!/usr/bin/env python3
"""Quick test to see current signal status."""

import sys
import asyncio
sys.path.append('src')

from src.data.market_data import MarketDataManager
from src.analysis.technical import TechnicalAnalyzer
from src.trading.broker_interface import BrokerManager
import MetaTrader5 as mt5

async def quick_test():
    print("🔍 QUICK SIGNAL TEST")
    print("="*30)
    
    data_manager = MarketDataManager()
    tech_analyzer = TechnicalAnalyzer()
    broker = BrokerManager()
    await broker.initialize()
    
    pair = 'EURUSD'
    print(f"📊 Testing {pair}...")
    
    df_15m = await data_manager.get_candles(pair, 'M15', 100)
    df_1h = await data_manager.get_candles(pair, 'H1', 50)
    
    signal = tech_analyzer.generate_signal(df_15m, df_1h)
    spread_valid = await broker.validate_spread(pair)
    
    print(f"🎯 Signal: {signal['direction']}")
    print(f"📊 Confidence: {signal.get('confidence', 0):.1%}")
    
    # Calculate R/R ratio manually
    entry_price = signal.get('entry_price', 0)
    stop_loss = signal.get('stop_loss', 0)
    take_profit = signal.get('take_profit', 0)
    
    if entry_price > 0 and stop_loss > 0 and take_profit > 0:
        from src.analysis.technical import SignalDirection
        if signal['direction'] == SignalDirection.BUY:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
        else:
            risk = abs(stop_loss - entry_price)
            reward = abs(entry_price - take_profit)
        
        rr_ratio = reward / risk if risk > 0 else 0
        print(f"💰 R/R Ratio: {rr_ratio:.2f}")
        print(f"💰 Entry: {entry_price:.5f}")
        print(f"🛡️ Stop Loss: {stop_loss:.5f}")
        print(f"🎯 Take Profit: {take_profit:.5f}")
    
    print(f"📊 Spread Valid: {spread_valid}")
    
    # Check current spread
    symbol_info = mt5.symbol_info(pair)
    if symbol_info:
        print(f"📊 Current Spread: {symbol_info.spread} points")
        print(f"📊 Bid/Ask: {symbol_info.bid:.5f}/{symbol_info.ask:.5f}")
    
    # Check with new parameters
    print(f"\n🔧 NEW PARAMETER CHECK:")
    print(f"   Min Confidence: 70% (was 75%)")
    print(f"   Min R/R Ratio: 2.0 (was 3.0)")
    print(f"   Max Spread: 30 pips (was 15)")
    
    # Determine if trade would be accepted
    confidence_ok = signal.get('confidence', 0) >= 0.70
    rr_ok = rr_ratio >= 2.0 if 'rr_ratio' in locals() else False
    spread_ok = spread_valid
    
    print(f"\n✅ TRADE DECISION:")
    print(f"   Confidence OK: {confidence_ok}")
    print(f"   R/R Ratio OK: {rr_ok}")
    print(f"   Spread OK: {spread_ok}")
    
    if confidence_ok and rr_ok and spread_ok:
        print(f"🚀 TRADE WOULD BE EXECUTED!")
    else:
        print(f"❌ Trade still rejected")

if __name__ == "__main__":
    asyncio.run(quick_test())