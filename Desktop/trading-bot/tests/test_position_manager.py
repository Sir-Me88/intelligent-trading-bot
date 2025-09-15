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
        assert sample_position.unrealized_pnl == 1.0  # (1.1010 - 1.1000) * 0.1 * 10000
    
    def test_update_current_price_sell(self, sample_position):
        """Test price update for sell position."""
        sample_position.direction = "sell"
        sample_position.entry_price = 1.1000
        sample_position.size = 0.1
        
        sample_position.update_current_price(1.0990)
        
        assert sample_position.current_price == 1.0990
        assert sample_position.unrealized_pnl == 1.0  # (1.1000 - 1.0990) * 0.1 * 10000
    
    def test_get_risk_amount(self, sample_position):
        """Test risk amount calculation."""
        sample_position.entry_price = 1.1000
        sample_position.stop_loss = 1.0950
        sample_position.size = 0.1
        sample_position.direction = "buy"
        
        risk = sample_position.get_risk_amount()
        expected_risk = abs(1.1000 - 1.0950) * 0.1
        
        assert risk == expected_risk


class TestPositionSizer:
    """Test PositionSizer class functionality."""
    
    def test_position_sizer_initialization(self):
        """Test position sizer initialization."""
        sizer = PositionSizer(risk_per_trade=0.02, max_total_risk=0.08)
        
        assert sizer.risk_per_trade == 0.02
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
        assert result['risk_amount'] == 100  # 1% of 10000
    
    def test_calculate_position_size_max_risk_exceeded(self, sample_position):
        """Test position size calculation when max risk is exceeded."""
        sizer = PositionSizer(max_total_risk=0.02)  # 2% max risk
        
        # Create positions that exceed max risk
        high_risk_position = sample_position
        high_risk_position.stop_loss = 1.0500  # Very wide stop
        
        result = sizer.calculate_position_size(
            account_equity=10000,
            entry_price=1.1000,
            stop_loss=1.0950,
            current_positions=[high_risk_position]
        )
        
        assert result['can_trade'] is False
        assert "Maximum total risk" in result['reason']
    
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
        assert "Invalid stop loss distance" in result['reason']


class TestRiskManager:
    """Test RiskManager class functionality."""
    
    @pytest.fixture
    def risk_manager(self, mock_market_data_manager):
        """Create risk manager for testing."""
        return RiskManager(mock_market_data_manager)
    
    @pytest.mark.asyncio
    async def test_validate_trade_success(self, risk_manager, sample_signal, mock_account_info):
        """Test successful trade validation."""
        validation = await risk_manager.validate_trade(
            signal=sample_signal,
            account_info=mock_account_info,
            current_positions=[]
        )
        
        assert validation['approved'] is True
        assert validation['position_size'] > 0
        assert validation['risk_amount'] > 0
    
    @pytest.mark.asyncio
    async def test_validate_trade_invalid_equity(self, risk_manager, sample_signal):
        """Test trade validation with invalid equity."""
        invalid_account = {'equity': 0}
        
        validation = await risk_manager.validate_trade(
            signal=sample_signal,
            account_info=invalid_account,
            current_positions=[]
        )
        
        assert validation['approved'] is False
        assert "Invalid account equity" in validation['reason']
    
    @pytest.mark.asyncio
    async def test_validate_trade_max_positions(self, risk_manager, sample_signal, mock_account_info):
        """Test trade validation when max positions reached."""
        # Create 10 open positions (max limit)
        positions = []
        for i in range(10):
            pos = Position(
                id=f"pos_{i}",
                pair="EUR_USD",
                direction="buy",
                size=0.1,
                entry_price=1.1000,
                current_price=1.1000,
                stop_loss=1.0950,
                take_profit=1.1100,
                unrealized_pnl=0,
                realized_pnl=0,
                status=PositionStatus.OPEN,
                open_time=datetime.now()
            )
            positions.append(pos)
        
        validation = await risk_manager.validate_trade(
            signal=sample_signal,
            account_info=mock_account_info,
            current_positions=positions
        )
        
        assert validation['approved'] is False
        assert "Maximum positions" in validation['reason']


class TestPositionManager:
    """Test PositionManager class functionality."""
    
    @pytest.fixture
    def position_manager(self, mock_market_data_manager):
        """Create position manager for testing."""
        return PositionManager(mock_market_data_manager)
    
    def test_create_position(self, position_manager, sample_signal):
        """Test position creation."""
        position = position_manager.create_position(sample_signal, 0.1)
        
        assert position.pair == sample_signal['pair']
        assert position.direction == sample_signal['direction']
        assert position.size == 0.1
        assert position.status == PositionStatus.OPEN
        assert len(position_manager.positions) == 1
    
    @pytest.mark.asyncio
    async def test_update_positions(self, position_manager, sample_signal):
        """Test position updates."""
        # Create a position
        position = position_manager.create_position(sample_signal, 0.1)
        
        # Mock price update
        position_manager.data_manager.get_current_price.return_value = {
            'bid': 1.1010,
            'ask': 1.1012
        }
        
        await position_manager.update_positions()
        
        # Verify price was updated
        assert position.current_price == 1.1010  # Bid price for buy position
    
    @pytest.mark.asyncio
    async def test_close_position(self, position_manager, sample_signal):
        """Test position closing."""
        position = position_manager.create_position(sample_signal, 0.1)
        
        await position_manager.close_position(position, "Test close")
        
        assert position.status == PositionStatus.CLOSED
        assert position.close_time is not None
        assert position.realized_pnl == position.unrealized_pnl
    
    def test_get_open_positions(self, position_manager, sample_signal):
        """Test getting open positions."""
        # Create positions
        pos1 = position_manager.create_position(sample_signal, 0.1)
        pos2 = position_manager.create_position(sample_signal, 0.1)
        
        # Close one position
        pos2.status = PositionStatus.CLOSED
        
        open_positions = position_manager.get_open_positions()
        
        assert len(open_positions) == 1
        assert open_positions[0] == pos1
    
    def test_get_total_unrealized_pnl(self, position_manager, sample_signal):
        """Test total unrealized PnL calculation."""
        # Create positions with PnL
        pos1 = position_manager.create_position(sample_signal, 0.1)
        pos1.unrealized_pnl = 10.0
        
        pos2 = position_manager.create_position(sample_signal, 0.1)
        pos2.unrealized_pnl = 15.0
        
        total_pnl = position_manager.get_total_unrealized_pnl()
        
        assert total_pnl == 25.0