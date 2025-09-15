#!/usr/bin/env python3
"""Analyze why signals are being rejected and optimize parameters."""

import sys
import asyncio
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')

from src.data.market_data import MarketDataManager
from src.analysis.technical import TechnicalAnalyzer, SignalDirection
from src.trading.broker_interface import BrokerManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def analyze_signal_rejection():
    """Analyze why signals are being rejected and suggest optimizations."""
    print("🔍 SIGNAL REJECTION ANALYSIS")
    print("="*50)
    
    # Initialize components
    data_manager = MarketDataManager()
    tech_analyzer = TechnicalAnalyzer()
    broker = BrokerManager()
    
    # Connect to broker
    print("🔌 Connecting to MT5...")
    await broker.initialize()
    
    # Current bot parameters (from run_core_trading_bot.py)
    current_params = {
        'min_confidence': 0.75,
        'min_rr_ratio': 3.0,
        'profit_protection_percentage': 0.25,
        'max_volatility': 0.002,
        'minimum_profit_to_protect': 20.0
    }
    
    print(f"📊 Current Parameters:")
    for key, value in current_params.items():
        print(f"   {key}: {value}")
    
    # Test pairs
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
    
    print(f"\n🎯 ANALYZING {len(pairs)} CURRENCY PAIRS")
    print("-"*50)
    
    analysis_results = []
    
    for pair in pairs:
        try:
            print(f"\n📈 Analyzing {pair}...")
            
            # Get market data
            df_15m = await data_manager.get_candles(pair, "M15", 100)
            df_1h = await data_manager.get_candles(pair, "H1", 50)
            
            if df_15m is None or df_1h is None:
                print(f"   ❌ No data available for {pair}")
                continue
            
            print(f"   📊 Data: {len(df_15m)} x 15M, {len(df_1h)} x 1H candles")
            
            # Generate signal
            signal = tech_analyzer.generate_signal(df_15m, df_1h)
            
            # Analyze signal details
            direction = signal['direction']
            confidence = signal.get('confidence', 0)
            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', 0)
            take_profit = signal.get('take_profit', 0)
            
            print(f"   🎯 Signal: {direction}")
            print(f"   📊 Confidence: {confidence:.2%}")
            
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
                print(f"   💰 R/R Ratio: {rr_ratio:.2f}")
            
            # Check spread
            spread_valid = await broker.validate_spread(pair)
            print(f"   📊 Spread Valid: {spread_valid}")
            
            # Calculate current volatility
            if len(df_15m) > 1:
                volatility = df_15m['close'].pct_change().std()
                print(f"   📊 Volatility: {volatility:.6f}")
            else:
                volatility = 0
            
            # Determine rejection reasons
            rejection_reasons = []
            
            if direction == SignalDirection.NONE:
                rejection_reasons.append("No signal generated")
            
            if confidence < current_params['min_confidence']:
                rejection_reasons.append(f"Low confidence ({confidence:.1%} < {current_params['min_confidence']:.1%})")
            
            if rr_ratio > 0 and rr_ratio < current_params['min_rr_ratio']:
                rejection_reasons.append(f"Poor R/R ratio ({rr_ratio:.1f} < {current_params['min_rr_ratio']:.1f})")
            
            if not spread_valid:
                rejection_reasons.append("High spread")
            
            if volatility > current_params['max_volatility']:
                rejection_reasons.append(f"High volatility ({volatility:.6f} > {current_params['max_volatility']:.6f})")
            
            # Store results
            result = {
                'pair': pair,
                'direction': direction,
                'confidence': confidence,
                'rr_ratio': rr_ratio,
                'volatility': volatility,
                'spread_valid': spread_valid,
                'rejection_reasons': rejection_reasons,
                'would_trade': len(rejection_reasons) == 0
            }
            
            analysis_results.append(result)
            
            # Display rejection reasons
            if rejection_reasons:
                print(f"   ❌ Rejected: {', '.join(rejection_reasons)}")
            else:
                print(f"   ✅ Would trade!")
                
        except Exception as e:
            print(f"   ❌ Error analyzing {pair}: {e}")
    
    # Summary analysis
    print(f"\n📊 SUMMARY ANALYSIS")
    print("="*30)
    
    total_pairs = len(analysis_results)
    tradeable_pairs = sum(1 for r in analysis_results if r['would_trade'])
    
    print(f"📈 Pairs analyzed: {total_pairs}")
    print(f"✅ Tradeable signals: {tradeable_pairs}")
    print(f"❌ Rejected signals: {total_pairs - tradeable_pairs}")
    print(f"📊 Success rate: {(tradeable_pairs/total_pairs)*100:.1f}%" if total_pairs > 0 else "📊 Success rate: 0%")
    
    # Analyze rejection patterns
    print(f"\n🔍 REJECTION PATTERN ANALYSIS")
    print("-"*35)
    
    all_reasons = []
    for result in analysis_results:
        all_reasons.extend(result['rejection_reasons'])
    
    if all_reasons:
        from collections import Counter
        reason_counts = Counter(all_reasons)
        
        print("Most common rejection reasons:")
        for reason, count in reason_counts.most_common():
            percentage = (count / total_pairs) * 100
            print(f"   {percentage:5.1f}% - {reason}")
    else:
        print("   ✅ No rejections found!")
    
    # Parameter optimization suggestions
    print(f"\n💡 OPTIMIZATION SUGGESTIONS")
    print("="*35)
    
    # Analyze confidence levels
    confidences = [r['confidence'] for r in analysis_results if r['confidence'] > 0]
    if confidences:
        avg_confidence = np.mean(confidences)
        max_confidence = max(confidences)
        print(f"📊 Confidence Analysis:")
        print(f"   Average: {avg_confidence:.1%}")
        print(f"   Maximum: {max_confidence:.1%}")
        
        if avg_confidence < current_params['min_confidence']:
            suggested_confidence = max(0.60, avg_confidence - 0.05)
            print(f"   💡 Suggest lowering min_confidence to {suggested_confidence:.1%}")
    
    # Analyze R/R ratios
    rr_ratios = [r['rr_ratio'] for r in analysis_results if r['rr_ratio'] > 0]
    if rr_ratios:
        avg_rr = np.mean(rr_ratios)
        print(f"📊 R/R Ratio Analysis:")
        print(f"   Average: {avg_rr:.2f}")
        
        if avg_rr < current_params['min_rr_ratio']:
            suggested_rr = max(2.0, avg_rr - 0.2)
            print(f"   💡 Suggest lowering min_rr_ratio to {suggested_rr:.1f}")
    
    # Analyze volatility
    volatilities = [r['volatility'] for r in analysis_results if r['volatility'] > 0]
    if volatilities:
        avg_vol = np.mean(volatilities)
        max_vol = max(volatilities)
        print(f"📊 Volatility Analysis:")
        print(f"   Average: {avg_vol:.6f}")
        print(f"   Maximum: {max_vol:.6f}")
        
        if max_vol > current_params['max_volatility']:
            suggested_vol = min(0.005, max_vol + 0.001)
            print(f"   💡 Suggest raising max_volatility to {suggested_vol:.6f}")
    
    return analysis_results

async def suggest_optimized_parameters(analysis_results):
    """Suggest optimized parameters based on analysis."""
    print(f"\n🚀 OPTIMIZED PARAMETER SUGGESTIONS")
    print("="*45)
    
    # Current parameters
    current_params = {
        'min_confidence': 0.75,
        'min_rr_ratio': 3.0,
        'max_volatility': 0.002
    }
    
    # Calculate suggested parameters
    confidences = [r['confidence'] for r in analysis_results if r['confidence'] > 0]
    rr_ratios = [r['rr_ratio'] for r in analysis_results if r['rr_ratio'] > 0]
    volatilities = [r['volatility'] for r in analysis_results if r['volatility'] > 0]
    
    suggested_params = current_params.copy()
    
    if confidences:
        avg_confidence = np.mean(confidences)
        # Suggest 5% below average confidence, but not below 60%
        suggested_params['min_confidence'] = max(0.60, avg_confidence - 0.05)
    
    if rr_ratios:
        avg_rr = np.mean(rr_ratios)
        # Suggest 20% below average R/R, but not below 2.0
        suggested_params['min_rr_ratio'] = max(2.0, avg_rr - 0.2)
    
    if volatilities:
        max_vol = max(volatilities)
        # Suggest 50% above max volatility, but not above 0.005
        suggested_params['max_volatility'] = min(0.005, max_vol * 1.5)
    
    print("📊 PARAMETER COMPARISON:")
    print(f"{'Parameter':<20} {'Current':<12} {'Suggested':<12} {'Change'}")
    print("-" * 55)
    
    for key in current_params:
        current = current_params[key]
        suggested = suggested_params[key]
        change = "↓ Relaxed" if suggested < current else "↑ Stricter" if suggested > current else "→ Same"
        
        if key == 'min_confidence':
            print(f"{'min_confidence':<20} {current:<12.1%} {suggested:<12.1%} {change}")
        else:
            print(f"{key:<20} {current:<12.3f} {suggested:<12.3f} {change}")
    
    return suggested_params

async def apply_optimized_parameters(suggested_params):
    """Apply optimized parameters to the bot."""
    print(f"\n🔧 APPLYING OPTIMIZED PARAMETERS")
    print("="*40)
    
    # Read the current bot file
    bot_file = 'run_core_trading_bot.py'
    
    try:
        with open(bot_file, 'r') as f:
            content = f.read()
        
        # Update parameters in the file
        lines = content.split('\n')
        updated_lines = []
        
        in_params_section = False
        
        for line in lines:
            if 'trading_params = {' in line:
                in_params_section = True
                updated_lines.append(line)
            elif in_params_section and '}' in line and 'trading_params' not in line:
                in_params_section = False
                updated_lines.append(line)
            elif in_params_section:
                # Update parameter lines
                if "'min_confidence':" in line:
                    updated_lines.append(f"            'min_confidence': {suggested_params['min_confidence']:.2f},")
                elif "'min_rr_ratio':" in line:
                    updated_lines.append(f"            'min_rr_ratio': {suggested_params['min_rr_ratio']:.1f},")
                elif "'max_volatility':" in line:
                    updated_lines.append(f"            'max_volatility': {suggested_params['max_volatility']:.6f},")
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
        
        # Write updated content
        updated_content = '\n'.join(updated_lines)
        
        # Backup original file
        import shutil
        shutil.copy(bot_file, f"{bot_file}.backup")
        print(f"✅ Backup created: {bot_file}.backup")
        
        # Write updated file
        with open(bot_file, 'w') as f:
            f.write(updated_content)
        
        print(f"✅ Parameters updated in {bot_file}")
        print(f"💡 Restart the bot to apply changes: python restart_bot.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating parameters: {e}")
        return False

async def main():
    """Main analysis function."""
    print("🚀 TRADING BOT OPTIMIZATION ANALYSIS")
    print("="*50)
    
    # Run analysis
    results = await analyze_signal_rejection()
    
    if not results:
        print("❌ No analysis results available")
        return
    
    # Suggest optimized parameters
    suggested_params = await suggest_optimized_parameters(results)
    
    # Ask user if they want to apply changes
    print(f"\n" + "="*50)
    apply_changes = input("Apply optimized parameters? (y/n): ").strip().lower()
    
    if apply_changes == 'y':
        success = await apply_optimized_parameters(suggested_params)
        
        if success:
            print(f"\n🎉 OPTIMIZATION COMPLETE!")
            print("="*30)
            print("✅ Parameters optimized")
            print("✅ Backup created")
            print("🚀 Ready to restart bot with new settings")
            
            restart = input("\nRestart bot now? (y/n): ").strip().lower()
            if restart == 'y':
                import subprocess
                print("🔄 Restarting bot...")
                subprocess.run(['python', 'restart_bot.py'])
        else:
            print("❌ Failed to apply optimizations")
    else:
        print("💡 Parameters not changed. You can run this analysis again anytime.")

if __name__ == "__main__":
    asyncio.run(main())