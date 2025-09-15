"""Tests for technical analysis module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.analysis.technical import TechnicalAnalyzer, PatternType, SignalDirection


class TestTechnicalAnalyzer:
    """Test technical analysis functionality."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly."""
        analyzer = TechnicalAnalyzer()
        assert analyzer is not None
    
    def test_bullish_engulfing_detection(self, sample_candle_data):
        """Test bullish engulfing pattern detection."""
        analyzer = TechnicalAnalyzer()
        
        # Create bullish engulfing pattern
        df = sample_candle_data.copy()
        df.iloc[-2] = [df.iloc[-2]['timestamp'], 1.1000, 1.1005, 1.0995, 1.0998, 1000]  # Small red candle
        df.iloc[-1] = [df.iloc[-1]['timestamp'], 1.0996, 1.1020, 1.0996, 1.1015, 1000]  # Large green candle
        
        pattern = analyzer._detect_bullish_engulfing(df)
        assert pattern is not None
        assert pattern['pattern'] == PatternType.BULLISH_ENGULFING
    
    def test_bearish_engulfing_detection(self, sample_candle_data):
        """Test bearish engulfing pattern detection."""
        analyzer = TechnicalAnalyzer()
        
        # Create bearish engulfing pattern
        df = sample_candle_data.copy()
        df.iloc[-2] = [df.iloc[-2]['timestamp'], 1.1000, 1.1015, 1.0995, 1.1012, 1000]  # Small green candle
        df.iloc[-1] = [df.iloc[-1]['timestamp'], 1.1015, 1.1015, 1.0990, 1.0995, 1000]  # Large red candle
        
        pattern = analyzer._detect_bearish_engulfing(df)
        assert pattern is not None
        assert pattern['pattern'] == PatternType.BEARISH_ENGULFING
    
    def test_support_resistance_calculation(self, sample_candle_data):
        """Test support and resistance level calculation."""
        analyzer = TechnicalAnalyzer()
        
        levels = analyzer._calculate_support_resistance(sample_candle_data)
        
        assert 'support' in levels
        assert 'resistance' in levels
        assert len(levels['support']) > 0
        assert len(levels['resistance']) > 0
        assert all(s < r for s in levels['support'] for r in levels['resistance'])
    
    def test_atr_calculation(self, sample_candle_data):
        """Test ATR calculation."""
        analyzer = TechnicalAnalyzer()
        
        atr = analyzer._calculate_atr(sample_candle_data)
        
        assert atr > 0
        assert isinstance(atr, float)
    
    def test_signal_generation_no_pattern(self, sample_candle_data):
        """Test signal generation when no pattern is found."""
        analyzer = TechnicalAnalyzer()
        
        # Use random data that shouldn't form clear patterns
        signal = analyzer.generate_signal(sample_candle_data, sample_candle_data)
        
        assert signal['direction'] == SignalDirection.NONE
        assert signal['confidence'] == 0.0
    
    @patch('src.analysis.technical.TechnicalAnalyzer._detect_bullish_engulfing')
    def test_signal_generation_with_pattern(self, mock_detect, sample_candle_data):
        """Test signal generation with detected pattern."""
        analyzer = TechnicalAnalyzer()
        
        # Mock pattern detection
        mock_detect.return_value = {
            'pattern': PatternType.BULLISH_ENGULFING,
            'confidence': 0.8,
            'entry_price': 1.1000,
            'stop_loss': 1.0950,
            'take_profit': 1.1100
        }
        
        signal = analyzer.generate_signal(sample_candle_data, sample_candle_data)
        
        assert signal['direction'] == SignalDirection.BUY
        assert signal['confidence'] == 0.8
        assert signal['entry_price'] == 1.1000
    
    def test_confluence_check(self, sample_candle_data):
        """Test confluence checking at support/resistance."""
        analyzer = TechnicalAnalyzer()
        
        # Test price near support level
        price = 1.0950
        levels = {'support': [1.0950, 1.0900], 'resistance': [1.1050, 1.1100]}
        
        confluence = analyzer._check_confluence(price, levels)
        assert confluence > 0
    
    def test_stop_loss_calculation(self, sample_candle_data):
        """Test stop loss calculation."""
        analyzer = TechnicalAnalyzer()
        
        entry_price = 1.1000
        direction = 'buy'
        atr = 0.0020
        
        stop_loss = analyzer._calculate_stop_loss(entry_price, direction, atr)
        
        assert stop_loss < entry_price  # Stop loss should be below entry for buy
        assert abs(entry_price - stop_loss) >= atr  # Should be at least 1 ATR away