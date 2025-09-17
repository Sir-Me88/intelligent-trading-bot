"""Tests for position manager module."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from src.trading.position_manager import (
    Position, PositionStatus, PositionSizer, RiskManager, PositionManager
)


class TestPosition:
    """Test Position class functionality."""
    
    def test_position_creation(self, sample_position):
        """Test position object creation."""
        assert sample_position.id == "test_pos_1"
        assert sample_position.pair == "EUR_USD"
        assert sample_position.direction == "buy"
        assert sample_position.status == PositionStatus.OPEN
    
    def test_update_current_price_buy(self, sample_position):
        """Test price update for buy position."""
        sample_position.direction = "buy"
        sample_position.entry_price = 1.1000
        sample_position.size = 0.1
        
        sample_position.update_current_price(1.1010)
        
        assert sample_position.current_price == 1.1010
        assert abs(sample_position.unrealized_pnl - 1.0) < 0.01  # Allow for floating point precision
    
    def test_update_current_price_sell(self, sample_position):
        """Test price update for sell position."""
        sample_position.direction = "sell"
        sample_position.entry_price = 1.1000
        sample_position.size = 0.1
        
        sample_position.update_current_price(1.0990)
        
        assert sample_position.current_price == 1.0990
        assert abs(sample_position.unrealized_pnl - 1.0) < 0.01  # Allow for floating point precision
    
    def test_get_risk_amount(self, sample_position):
        """Test risk amount calculation."""
        sample_position.entry_price = 1.1000
        sample_position.stop_loss = 1.0950
        sample_position.size = 0.1
        sample_position.direction = "buy"
        
        risk = sample_position.get_risk_amount()
        # Risk is in pip units * lots: (50 pips) * 0.1 lots = 5.0
        expected_risk = 50 * 0.1  # 50 pips * 0.1 lots

        assert abs(risk - expected_risk) < 0.01


class TestPositionSizer:
    """Test PositionSizer class functionality."""
    
    def test_position_sizer_initialization(self):
        """Test position sizer initialization."""
        sizer = PositionSizer(max_total_risk=0.08)

        assert sizer.max_total_risk == 0.08
    
    def test_calculate_position_size_success(self):
        """Test successful position size calculation."""
        sizer = PositionSizer()
        
        result = sizer.calculate_position_size(
            account_equity=10000,
            entry_price=1.1000,
            stop_loss=1.0950,
            current_positions=[]
        )
        
        assert result['can_trade'] is True
        assert result['size'] > 0
    
    def test_calculate_position_size_max_risk_exceeded(self, sample_position):
        """Test position size calculation when max risk is exceeded."""
        sizer = PositionSizer(max_total_risk=0.001)  # Very low risk limit

        # Create a position with high risk
        sample_position.entry_price = 1.1000
        sample_position.stop_loss = 1.0950  # 50 pip stop
        sample_position.size = 1.0  # Large position

        result = sizer.calculate_position_size(
            account_equity=10000,
            entry_price=1.1000,
            stop_loss=1.0950,
            current_positions=[sample_position]
        )

        assert result['can_trade'] is False

    def test_calculate_position_size_invalid_stop(self):
        """Test position size calculation with invalid stop loss."""
        sizer = PositionSizer()

        result = sizer.calculate_position_size(
            account_equity=10000,
            entry_price=1.1000,
            stop_loss=1.1000,  # Same as entry price
            current_positions=[]
        )

        assert result['can_trade'] is False


class TestRiskManager:
    """Test RiskManager class functionality."""

    def test_validate_trade_success(self):
        """Test successful trade validation."""
        risk_manager = RiskManager()
        result = risk_manager.validate_trade(10000, [], None)
        assert result is True

    def test_validate_trade_invalid_equity(self):
        """Test trade validation with invalid equity."""
        risk_manager = RiskManager()
        result = risk_manager.validate_trade(500, [], None)
        assert result is False

    def test_validate_trade_max_positions(self):
        """Test trade validation when max positions reached."""
        risk_manager = RiskManager()
        positions = [None] * 6  # More than 5 positions
        result = risk_manager.validate_trade(10000, positions, None)
        assert result is False


class TestPositionManager:
    """Test PositionManager class functionality."""

    def test_create_position(self):
        """Test position creation."""
        manager = PositionManager()
        position = manager.create_position(
            "test_id", "EUR_USD", "buy", 0.1, 1.1000, 1.0950, 1.1100
        )

        assert position.pair == "EUR_USD"
        assert position.direction == "buy"
        assert position.size == 0.1
        assert position.status == PositionStatus.OPEN
        assert len(manager.positions) == 1

    def test_update_positions(self):
        """Test position updates."""
        manager = PositionManager()
        position = manager.create_position(
            "test_id", "EUR_USD", "buy", 0.1, 1.1000, 1.0950, 1.1100
        )

        manager.update_positions()

        # Verify price was updated (should stay the same since no new price)
        assert position.current_price == 1.1000

    def test_close_position(self):
        """Test position closing."""
        manager = PositionManager()
        position = manager.create_position(
            "test_id", "EUR_USD", "buy", 0.1, 1.1000, 1.0950, 1.1100
        )

        result = manager.close_position("test_id")

        assert result is True
        assert position.status == PositionStatus.CLOSED
        assert position.close_time is not None

    def test_get_open_positions(self):
        """Test getting open positions."""
        manager = PositionManager()

        # Create positions
        pos1 = manager.create_position(
            "test_id1", "EUR_USD", "buy", 0.1, 1.1000, 1.0950, 1.1100
        )
        pos2 = manager.create_position(
            "test_id2", "GBP_USD", "sell", 0.1, 1.2000, 1.2050, 1.1900
        )

        # Close one position
        manager.close_position("test_id2")

        open_positions = manager.get_open_positions()

        assert len(open_positions) == 1
        assert open_positions[0] == pos1

    def test_get_total_unrealized_pnl(self):
        """Test total unrealized PnL calculation."""
        manager = PositionManager()

        # Create positions with PnL
        pos1 = manager.create_position(
            "test_id1", "EUR_USD", "buy", 0.1, 1.1000, 1.0950, 1.1100
        )
        pos1.unrealized_pnl = 10.0

        pos2 = manager.create_position(
            "test_id2", "GBP_USD", "sell", 0.1, 1.2000, 1.2050, 1.1900
        )
        pos2.unrealized_pnl = 15.0

        total_pnl = manager.get_total_unrealized_pnl()

        assert total_pnl == 25.0
