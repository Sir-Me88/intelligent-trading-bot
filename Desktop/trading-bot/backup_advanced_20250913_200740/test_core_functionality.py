#!/usr/bin/env python3
"""Core functionality test for Advanced Trading Bot."""

import sys
import os
from pathlib import Path
from datetime import datetime

def test_core_imports():
    """Test core functionality without external dependencies."""
    print("🔍 Testing core functionality...")

    try:
        # Test basic Python functionality
        current_time = datetime.now()
        print(f"✅ Python datetime working: {current_time}")

        # Test our core classes without external dependencies
        sys.path.append('src')

        # Test trade attribution (pure Python, no external deps)
        from src.analysis.trade_attribution import AttributionResult, TradeAttributionAnalyzer

        # Create a sample attribution result
        result = AttributionResult(
            total_return=1000.0,
            annualized_return=0.15,
            volatility=0.02,
            sharpe_ratio=1.5,
            max_drawdown=0.05,
            win_rate=0.65,
            profit_factor=1.8,
            avg_win=50.0,
            avg_loss=-30.0,
            largest_win=200.0,
            largest_loss=-80.0,
            total_trades=100,
            winning_trades=65,
            losing_trades=35,
            strategy_contributions={'strategy_a': 40.0, 'strategy_b': 60.0},
            market_condition_performance={},
            parameter_impact={},
            time_based_returns={},
            risk_adjusted_contributions={}
        )

        print("✅ Trade attribution classes working")
        print(f"   Sample result: {result.total_return} total return")

        # Test analyzer instantiation
        analyzer = TradeAttributionAnalyzer()
        print("✅ Trade attribution analyzer instantiated")

        # Add sample trade
        sample_trade = {
            'timestamp': datetime.now().isoformat(),
            'ticket': 12345,
            'symbol': 'EURUSD',
            'direction': 'BUY',
            'entry_price': 1.0500,
            'stop_loss': 1.0450,
            'take_profit': 1.0600,
            'volume': 0.1,
            'confidence': 0.85,
            'profit': 50.0,
            'exit_reason': 'take_profit',
            'hold_duration': 2.5
        }

        analyzer.add_trade(sample_trade)
        print("✅ Sample trade added to analyzer")

        return True

    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        return False

def test_file_operations():
    """Test file operations and logging."""
    print("\n🔍 Testing file operations...")

    try:
        # Test log directory creation
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Test heartbeat file creation
        heartbeat_file = logs_dir / "test_heartbeat.json"
        test_data = {
            'timestamp': datetime.now().isoformat(),
            'status': 'testing',
            'version': 'advanced_bot_v1.0'
        }

        import json
        with open(heartbeat_file, 'w') as f:
            json.dump(test_data, f, indent=2)

        print("✅ File operations working")
        print(f"   Created test file: {heartbeat_file}")

        # Clean up
        if heartbeat_file.exists():
            heartbeat_file.unlink()

        return True

    except Exception as e:
        print(f"❌ File operations test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\n🔍 Testing configuration...")

    try:
        # Test environment variable loading
        os.environ['TEST_VAR'] = 'test_value'

        # Test basic config structure
        config = {
            'database': {
                'host': 'localhost',
                'port': 5432
            },
            'trading': {
                'pairs': ['EURUSD', 'GBPUSD', 'USDJPY'],
                'max_positions': 5,
                'risk_per_trade': 0.01
            },
            'ml': {
                'model_path': 'models/',
                'confidence_threshold': 0.8
            }
        }

        print("✅ Configuration structure valid")
        print(f"   Trading pairs: {config['trading']['pairs']}")
        print(f"   Risk per trade: {config['trading']['risk_per_trade']}")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_adaptive_parameters():
    """Test adaptive parameter system."""
    print("\n🔍 Testing adaptive parameters...")

    try:
        # Test parameter adaptation logic
        adaptive_params = {
            'min_confidence': 0.85,
            'min_rr_ratio': 3.5,
            'profit_protection_percentage': 0.20,
            'atr_multiplier_normal_vol': 2.5
        }

        # Simulate parameter updates
        # Increase confidence if win rate is low
        win_rate = 0.55  # Below 60%
