#!/usr/bin/env python3
"""Investigate the quality of signals being generated."""

import sys
import asyncio
sys.path.append('src')

from src.data.market_data import MarketDataManager
from src.analysis.technical import TechnicalAnalyzer, SignalDirection
from src.trading.broker_interface import BrokerManager

async def investigate_signals():
    """Investigate signal quality and parameters."""
    print("🔍 SIGNAL QUALITY INVESTIGATION")
    print("="*50)
    
    data_manager = MarketDataManager()
    tech_analyzer = TechnicalAnalyzer()
    broker = BrokerManager()
    await broker.initialize()
    
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
    
    # Current bot parameters (after our changes)
    current_params = {
        'min_confidence': 0.70,
        'min_rr_ratio': 1.2,
        'max_volatility': 0.003,
        'max_spread_pips': 30
    }
    
    print("🔧 CURRENT PARAMETERS:")
    for key, value in current_params.items():
        print(f"   {key}: {value}")
    
    print(f"\n📊 ANALYZING ALL {len(pairs)} PAIRS:")
    print("-"*50)
    
    all_signals = []
    
    for pair in pairs:
        try:
            print(f"\n📈 {pair}:")
            
            # Get market data
            df_15m = await data_manager.get_candles(pair, "M15", 100)
            df_1h = await data_manager.get_candles(pair, "H1", 50)
            
            if df_15m is None or df_1h is None:
                print(f"   ❌ No data")
                continue
            
            # Generate signal
            signal = tech_analyzer.generate_signal(df_15m, df_1h)
            
            direction = signal['direction']
            confidence = signal.get('confidence', 0)
            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', 0)
            take_profit = signal.get('take_profit', 0)
            
            # Calculate R/R ratio
            rr_ratio = 0
            if entry_price > 0 and stop_loss > 0 and take_profit > 0:
                if direction == SignalDirection.BUY:
                    risk = abs(entry_price - stop_loss)
                    reward = abs(take_profit - entry_price)
                else:
                    risk = abs(stop_loss - entry_price)
                    reward = abs(entry_price - take_profit)
                
                rr_ratio = reward / risk if risk > 0 else 0
            
            # Check spread
            spread_valid = await broker.validate_spread(pair)
            
            # Calculate volatility
            volatility = df_15m['close'].pct_change().std() if len(df_15m) > 1 else 0
            
            print(f"   🎯 Direction: {direction}")
            print(f"   📊 Confidence: {confidence:.1%}")
            print(f"   💰 R/R Ratio: {rr_ratio:.2f}")
            print(f"   📊 Volatility: {volatility:.6f}")
            print(f"   📊 Spread Valid: {spread_valid}")
            
            # Check if signal meets current criteria
            meets_criteria = (
                direction != SignalDirection.NONE and
                confidence >= current_params['min_confidence'] and
                rr_ratio >= current_params['min_rr_ratio'] and
                volatility <= current_params['max_volatility'] and
                spread_valid
            )
            
            print(f"   ✅ Meets Criteria: {meets_criteria}")
            
            all_signals.append({
                'pair': pair,
                'direction': direction,
                'confidence': confidence,
                'rr_ratio': rr_ratio,
                'volatility': volatility,
                'spread_valid': spread_valid,
                'meets_criteria': meets_criteria
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Analysis summary
    print(f"\n📊 SIGNAL QUALITY ANALYSIS")
    print("="*35)
    
    total_signals = len(all_signals)
    valid_signals = sum(1 for s in all_signals if s['meets_criteria'])
    
    print(f"📈 Total pairs analyzed: {total_signals}")
    print(f"✅ Signals meeting criteria: {valid_signals}")
    print(f"❌ Signals rejected: {total_signals - valid_signals}")
    print(f"📊 Acceptance rate: {(valid_signals/total_signals)*100:.1f}%" if total_signals > 0 else "0%")
    
    if valid_signals > 0:
        # Analyze signal quality
        confidences = [s['confidence'] for s in all_signals if s['meets_criteria']]
        rr_ratios = [s['rr_ratio'] for s in all_signals if s['meets_criteria']]
        
        print(f"\n📊 QUALITY METRICS:")
        print(f"   Average Confidence: {sum(confidences)/len(confidences):.1%}")
        print(f"   Average R/R Ratio: {sum(rr_ratios)/len(rr_ratios):.2f}")
        print(f"   Min R/R Ratio: {min(rr_ratios):.2f}")
        print(f"   Max R/R Ratio: {max(rr_ratios):.2f}")
    
    # Check if parameters are too loose
    print(f"\n🚨 PARAMETER ASSESSMENT:")
    
    if valid_signals == total_signals:
        print("   ⚠️  ALL signals accepted - parameters may be TOO LOOSE")
        print("   💡 Consider tightening criteria for better quality")
    elif valid_signals == 0:
        print("   ⚠️  NO signals accepted - parameters may be TOO STRICT")
        print("   💡 Consider relaxing criteria")
    else:
        print(f"   ✅ Balanced: {(valid_signals/total_signals)*100:.0f}% acceptance rate")
    
    # Suggest better parameters
    if all_signals:
        print(f"\n💡 SUGGESTED IMPROVEMENTS:")
        
        all_confidences = [s['confidence'] for s in all_signals if s['confidence'] > 0]
        all_rr_ratios = [s['rr_ratio'] for s in all_signals if s['rr_ratio'] > 0]
        
        if all_confidences:
            avg_confidence = sum(all_confidences) / len(all_confidences)
            suggested_confidence = min(0.85, avg_confidence + 0.05)  # Slightly above average
            print(f"   📊 Suggested min_confidence: {suggested_confidence:.2f} (currently {current_params['min_confidence']:.2f})")
        
        if all_rr_ratios:
            avg_rr = sum(all_rr_ratios) / len(all_rr_ratios)
            suggested_rr = max(1.5, avg_rr + 0.2)  # Slightly above average
            print(f"   💰 Suggested min_rr_ratio: {suggested_rr:.1f} (currently {current_params['min_rr_ratio']:.1f})")
    
    return all_signals

if __name__ == "__main__":
    asyncio.run(investigate_signals())