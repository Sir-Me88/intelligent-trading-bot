"""Pytest configuration and fixtures."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.config.settings import Settings
from src.data.market_data import MarketDataManager
from src.trading.position_manager import Position, PositionStatus


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Settings()
    settings.mt5.login = "test_login"
    settings.mt5.password = "test_password"
    settings.mt5.server = "test_server"
    settings.trading.risk_per_trade = 0.01
    settings.trading.max_total_risk = 0.06
    return settings


@pytest.fixture
def sample_candle_data():
    """Sample OHLC data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='15T')
    
    # Generate realistic forex data
    np.random.seed(42)
    close_prices = 1.1000 + np.cumsum(np.random.randn(100) * 0.0001)
    
    data = {
        'timestamp': dates,
        'open': close_prices + np.random.randn(100) * 0.0001,
        'high': close_prices + np.abs(np.random.randn(100) * 0.0002),
        'low': close_prices - np.abs(np.random.randn(100) * 0.0002),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 100)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_market_data_manager():
    """Mock market data manager."""
    manager = AsyncMock(spec=MarketDataManager)
    
    # Mock current price data
    manager.get_current_price.return_value = {
        'bid': 1.1000,
        'ask': 1.1002,
        'timestamp': datetime.now()
    }
    
    return manager


@pytest.fixture
def sample_position():
    """Sample trading position for testing."""
    return Position(
        id="test_pos_1",
        pair="EUR_USD",
        direction="buy",
        size=0.1,
        entry_price=1.1000,
        current_price=1.1010,
        stop_loss=1.0950,
        take_profit=1.1100,
        unrealized_pnl=10.0,
        realized_pnl=0.0,
        status=PositionStatus.OPEN,
        open_time=datetime.now() - timedelta(hours=1)
    )


@pytest.fixture
def mock_account_info():
    """Mock account information."""
    return {
        'balance': 10000.0,
        'equity': 10050.0,
        'margin_used': 100.0,
        'margin_available': 9950.0,
        'unrealized_pnl': 50.0,
        'currency': 'USD'
    }


@pytest.fixture
def sample_signal():
    """Sample trading signal."""
    return {
        'pair': 'EUR_USD',
        'direction': 'buy',
        'entry_price': 1.1000,
        'stop_loss': 1.0950,
        'take_profit': 1.1100,
        'confidence': 0.75,
        'pattern': 'bullish_engulfing',
        'timeframe': 'M15'
    }