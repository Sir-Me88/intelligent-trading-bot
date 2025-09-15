#!/usr/bin/env python3
"""Configuration settings for the trading bot using dataclasses."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables once at module level
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class TradingSettings:
    """Trading bot configuration with validation."""

    # Core trading pairs
    currency_pairs: List[str] = field(default_factory=lambda: [
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
        'USDCHF', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY'
    ])

    # API Configuration
    news_api_url: str = "https://financialmodelingprep.com/api/v3/economic_calendar"
    news_api_key: str = field(default_factory=lambda: os.getenv('FMP_API_KEY', ''))
    backup_api_url: str = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"
    backup_api_enabled: bool = True

    # Twelve Data API (for future use)
    twelve_data_key: str = field(default_factory=lambda: os.getenv('TWELVE_DATA_KEY', ''))

    # MT5 Configuration
    mt5_login: str = field(default_factory=lambda: os.getenv('MT5_LOGIN', ''))
    mt5_password: str = field(default_factory=lambda: os.getenv('MT5_PASSWORD', ''))
    mt5_server: str = field(default_factory=lambda: os.getenv('MT5_SERVER', ''))

    # Trading parameters
    atr_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'low_vol': 2.0,
        'normal_vol': 2.5,
        'high_vol': 3.0,
        'xauusd': 3.5
    })
    volatility_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.001,
        'high': 0.003,
        'xauusd': 0.005
    })

    # Backtesting
    backtest_mode: bool = False
    backtest_period_days: int = 30

    # API limits
    api_daily_limit: int = 250
    twelve_data_daily_limit: int = 800
    twelve_data_minute_limit: int = 8

    # Runtime tracking
    api_calls_today: int = 0
    last_api_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Validate settings after initialization."""
        self._validate_settings()

    def _validate_settings(self):
        """Validate critical settings and log warnings."""
        if not self.news_api_key:
            logger.warning("FMP_API_KEY not found in environment variables")

        if not self.mt5_login or not self.mt5_password or not self.mt5_server:
            logger.warning("MT5 credentials not found in environment variables")

        # Validate ATR multipliers
        for key, value in self.atr_multipliers.items():
            if value <= 0:
                raise ValueError(f"ATR multiplier for {key} must be positive")

        # Validate volatility thresholds
        for key, value in self.volatility_thresholds.items():
            if value <= 0:
                raise ValueError(f"Volatility threshold for {key} must be positive")

        logger.info("✅ Settings validated successfully")

    def get_atr_multiplier(self, pair: str) -> float:
        """Get ATR multiplier for a given pair."""
        pair = pair.lower()
        if pair == 'xauusd':
            return self.atr_multipliers['xauusd']
        return self.atr_multipliers['normal_vol']

    def get_volatility_threshold(self, pair: str) -> float:
        """Get volatility threshold for a given pair."""
        pair = pair.lower()
        if pair == 'xauusd':
            return self.volatility_thresholds['xauusd']
        return self.volatility_thresholds['normal']

    def increment_api_calls(self) -> bool:
        """Track API call count and reset daily. Returns False if limit reached."""
        today = datetime.now(timezone.utc).date()
        if today > self.last_api_reset.date():
            self.api_calls_today = 0
            self.last_api_reset = datetime.now(timezone.utc)

        if self.api_calls_today >= self.api_daily_limit:
            logger.warning("Daily API call limit reached")
            return False

        self.api_calls_today += 1
        return True

    def get_currency_pairs(self) -> List[str]:
        """Return the list of currency pairs."""
        return self.currency_pairs.copy()

# Create global settings instance
settings = TradingSettings()

# Trading-related runtime settings (maintained for compatibility)
from types import SimpleNamespace
trading = SimpleNamespace(
    risk_per_trade=0.01,     # default risk per trade (1%)
    max_total_risk=0.02,     # total account risk limit (2%)
    max_open_positions=5,
    default_lot_size=0.1
)

# Attach trading settings to global settings for compatibility
settings.trading = trading
