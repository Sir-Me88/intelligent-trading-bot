import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from src.analysis.correlation import CorrelationAnalyzer
from src.data.market_data import MarketDataManager

@pytest.fixture
def mock_market_data():
    market_data = Mock(spec=MarketDataManager)
    market_data.get_candles = AsyncMock()
    market_data.fetch_alternative_data = AsyncMock()
    return market_data

@pytest.fixture
def correlation_analyzer(mock_market_data):
    return CorrelationAnalyzer(mock_market_data, lookback_days=5)

@pytest.mark.asyncio
async def test_update_correlation_matrix(correlation_analyzer, mock_market_data):
    # Prepare test data
    test_data = pd.DataFrame({
        'close': [1.0, 1.1, 1.2, 1.1, 1.0]
    })
    
    # Configure mock
    mock_market_data.get_candles.return_value = test_data
    
    # Test with two pairs
    pairs = ['EURUSD', 'GBPUSD']
    result = await correlation_analyzer.update_correlation_matrix(pairs)
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)  # 2x2 correlation matrix
    assert list(result.columns) == pairs

@pytest.mark.asyncio
async def test_alternative_data_fallback(correlation_analyzer, mock_market_data):
    # Mock MT5 failure
    mock_market_data.get_candles.return_value = None
    
    # Mock Twelve Data success
    alt_data = {
        'values': [
            {'close': '1.0'},
            {'close': '1.1'},
            {'close': '1.2'}
        ]
    }
    mock_market_data.fetch_alternative_data.return_value = alt_data
    
    pairs = ['EURUSD']
    result = await correlation_analyzer.update_correlation_matrix(pairs)
    
    assert mock_market_data.fetch_alternative_data.called
    assert isinstance(result, pd.DataFrame)

def test_find_hedging_opportunities(correlation_analyzer):
    # Create test correlation matrix
    test_matrix = pd.DataFrame({
        'EURUSD': [1.0, 0.9, -0.8],
        'GBPUSD': [0.9, 1.0, -0.7],
        'USDJPY': [-0.8, -0.7, 1.0]
    }, index=['EURUSD', 'GBPUSD', 'USDJPY'])
    
    correlation_analyzer.correlation_matrix = test_matrix
    
    opportunities = correlation_analyzer.find_hedging_opportunities('EURUSD', 0.8)
    
    assert len(opportunities) == 2
    assert opportunities[0]['pair'] == 'GBPUSD'
    assert opportunities[0]['correlation'] == 0.9
    assert opportunities[1]['pair'] == 'USDJPY'
    assert opportunities[1]['correlation'] == -0.8

def test_should_hedge_position(correlation_analyzer):
    # Setup test data
    test_matrix = pd.DataFrame({
        'EURUSD': [1.0, 0.9],
        'GBPUSD': [0.9, 1.0]
    }, index=['EURUSD', 'GBPUSD'])
    
    correlation_analyzer.correlation_matrix = test_matrix
    
    open_positions = [
        {'symbol': 'EURUSD', 'type': 0, 'volume': 1.0}  # BUY position
    ]
    
    new_signal = {
        'pair': 'GBPUSD',
        'direction': 'BUY'
    }
    
    result = correlation_analyzer.should_hedge_position(open_positions, new_signal)
    
    assert result['should_hedge'] is True
    assert len(result['hedge_pairs']) == 1
    assert result['correlation_risk'] > 0.8

def test_get_currency_exposure(correlation_analyzer):
    positions = [
        {'symbol': 'EURUSD', 'type': 0, 'volume': 1.0},  # BUY
        {'symbol': 'GBPUSD', 'type': 1, 'volume': 0.5}   # SELL
    ]
    
    exposure = correlation_analyzer.get_currency_exposure(positions)
    
    assert 'EUR' in exposure
    assert 'USD' in exposure
    assert 'GBP' in exposure
    assert exposure['EUR'] == 1.0
    assert exposure['GBP'] == -0.5