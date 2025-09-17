"""Tests for correlation analysis module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock

from src.analysis.correlation import CorrelationAnalyzer


class TestCorrelationAnalyzer:
    """Test correlation analysis functionality."""
    
    @pytest.fixture
    def correlation_analyzer(self, mock_market_data_manager):
        """Create correlation analyzer for testing."""
        return CorrelationAnalyzer(mock_market_data_manager)
    
    @pytest.fixture
    def sample_price_data(self):
        """Sample price data for correlation calculation."""
        dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
        
        # Create correlated price series
        base_returns = np.random.randn(60) * 0.01
        
        data = {
            'EUR_USD': pd.Series(1.1000 + np.cumsum(base_returns), index=dates),
            'GBP_USD': pd.Series(1.2500 + np.cumsum(base_returns * 0.8 + np.random.randn(60) * 0.005), index=dates),
            'USD_JPY': pd.Series(150.0 + np.cumsum(-base_returns * 0.6 + np.random.randn(60) * 0.5), index=dates),
            'AUD_USD': pd.Series(0.6500 + np.cumsum(base_returns * 0.7 + np.random.randn(60) * 0.008), index=dates)
        }
        
        return data
    
    def test_calculate_correlation_matrix(self, correlation_analyzer, sample_price_data):
        """Test correlation matrix calculation."""
        # Mock data manager to return sample data
        async def mock_get_candles(pair, timeframe, count):
            prices = sample_price_data[pair]
            return pd.DataFrame({
                'timestamp': prices.index,
                'close': prices.values
            })
        
        correlation_analyzer.data_manager.get_candles = mock_get_candles
        
        # Calculate correlation matrix
        correlation_matrix = correlation_analyzer._calculate_correlation_matrix(
            list(sample_price_data.keys()),
            sample_price_data
        )
        
        # Verify matrix properties
        assert correlation_matrix.shape == (4, 4)
        assert np.allclose(np.diag(correlation_matrix), 1.0)  # Diagonal should be 1
        assert np.allclose(correlation_matrix, correlation_matrix.T)  # Should be symmetric
    
    @pytest.mark.asyncio
    async def test_update_correlation_matrix(self, correlation_analyzer, sample_price_data):
        """Test correlation matrix update."""
        pairs = list(sample_price_data.keys())
        
        # Mock data manager
        async def mock_get_candles(pair, timeframe, count):
            prices = sample_price_data[pair]
            return pd.DataFrame({
                'timestamp': prices.index,
                'close': prices.values
            })
        
        correlation_analyzer.data_manager.get_candles = mock_get_candles
        
        await correlation_analyzer.update_correlation_matrix(pairs)
        
        assert correlation_analyzer.correlation_matrix is not None
        assert correlation_analyzer.last_update is not None
    
    def test_get_correlation(self, correlation_analyzer):
        """Test getting correlation between two pairs."""
        # Set up mock correlation matrix
        correlation_analyzer.correlation_matrix = pd.DataFrame({
            'EUR_USD': [1.0, 0.8, -0.6, 0.7],
            'GBP_USD': [0.8, 1.0, -0.5, 0.6],
            'USD_JPY': [-0.6, -0.5, 1.0, -0.4],
            'AUD_USD': [0.7, 0.6, -0.4, 1.0]
        }, index=['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD'])
        
        correlation = correlation_analyzer.get_correlation('EUR_USD', 'GBP_USD')
        assert correlation == 0.8
        
        correlation = correlation_analyzer.get_correlation('EUR_USD', 'USD_JPY')
        assert correlation == -0.6
    
    def test_find_correlated_pairs(self, correlation_analyzer):
        """Test finding correlated pairs."""
        # Set up mock correlation matrix
        correlation_analyzer.correlation_matrix = pd.DataFrame({
            'EUR_USD': [1.0, 0.8, -0.6, 0.7],
            'GBP_USD': [0.8, 1.0, -0.5, 0.6],
            'USD_JPY': [-0.6, -0.5, 1.0, -0.4],
            'AUD_USD': [0.7, 0.6, -0.4, 1.0]
        }, index=['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD'])
        
        # Find pairs correlated with EUR_USD above 0.7 threshold
        correlated = correlation_analyzer.find_correlated_pairs('EUR_USD', 0.7)

        correlated_pairs = [item['pair'] for item in correlated]
        expected_pairs = ['GBP_USD', 'AUD_USD']  # 0.8 and 0.7 correlations
        assert set(correlated_pairs) == set(expected_pairs)
    
    def test_should_hedge_position(self, correlation_analyzer):
        """Test hedging recommendation logic."""
        # Set up mock correlation matrix
        correlation_analyzer.correlation_matrix = pd.DataFrame({
            'EUR_USD': [1.0, 0.9, -0.8, 0.7],
            'GBP_USD': [0.9, 1.0, -0.7, 0.6],
            'USD_JPY': [-0.8, -0.7, 1.0, -0.5],
            'AUD_USD': [0.7, 0.6, -0.5, 1.0]
        }, index=['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD'])
        
        # Test with existing EUR_USD position
        existing_positions = [
            {
                'symbol': 'EUR_USD',
                'type': 'buy',
                'volume': 0.1,
                'unrealized_pnl': 50.0
            }
        ]
        
        new_signal = {
            'pair': 'GBP_USD',
            'direction': 'buy',
            'size': 0.1
        }
        
        hedge_analysis = correlation_analyzer.should_hedge_position(
            existing_positions, new_signal, correlation_threshold=0.8
        )
        
        assert hedge_analysis['should_hedge'] is True
        assert len(hedge_analysis['hedge_pairs']) > 0
        
        # Check hedge pair details - should hedge EUR_USD against GBP_USD due to positive correlation
        hedge_pair = hedge_analysis['hedge_pairs'][0]
        assert hedge_pair['pair'] == 'EUR_USD'  # The existing position that needs hedging
        assert hedge_pair['correlation'] > 0.8  # Positive correlation
    
    def test_is_correlation_matrix_stale(self, correlation_analyzer):
        """Test correlation matrix staleness check."""
        # Initially should be stale (no last_update)
        assert correlation_analyzer.is_correlation_matrix_stale() is True
        
        # Set recent update
        from datetime import datetime, timedelta
        correlation_analyzer.last_update = datetime.now() - timedelta(hours=1)
        assert correlation_analyzer.is_correlation_matrix_stale() is False
        
        # Set old update
        correlation_analyzer.last_update = datetime.now() - timedelta(days=2)
        assert correlation_analyzer.is_correlation_matrix_stale() is True
    
    def test_calculate_hedge_ratio(self, correlation_analyzer):
        """Test hedge ratio calculation."""
        # Mock volatility data
        correlation_analyzer.pair_volatilities = {
            'EUR_USD': 0.0015,
            'GBP_USD': 0.0018,
            'USD_JPY': 0.0012
        }
        
        hedge_ratio = correlation_analyzer._calculate_hedge_ratio(
            'EUR_USD', 'GBP_USD', correlation=0.8
        )
        
        # Hedge ratio should be correlation * (vol1 / vol2)
        expected_ratio = 0.8 * (0.0015 / 0.0018)
        assert abs(hedge_ratio - expected_ratio) < 0.001
    
    def test_empty_correlation_matrix(self, correlation_analyzer):
        """Test behavior with empty correlation matrix."""
        correlation_analyzer.correlation_matrix = None
        
        correlation = correlation_analyzer.get_correlation('EUR_USD', 'GBP_USD')
        assert correlation == 0.0
        
        correlated_pairs = correlation_analyzer.find_correlated_pairs('EUR_USD', 0.8)
        assert correlated_pairs == []
