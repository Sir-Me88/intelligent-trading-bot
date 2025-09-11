#!/usr/bin/env python3
"""Configuration settings for the trading bot."""

from typing import List, Dict, Optional
import logging
import aiohttp
import asyncio
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import os
from types import SimpleNamespace

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class Settings:
    """Trading bot configuration settings."""
    
    # Class constants
    API_DAILY_LIMIT: int = 250
    DEFAULT_BACKTEST_DAYS: int = 30
    CURRENCY_PAIRS: List[str] = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
        'USDCHF', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY',
        ]
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern and ensure attributes are initialized."""
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance._initialize_attributes()
        return cls._instance
    
    def _initialize_attributes(self):
        """Initialize attributes to ensure they are always set."""
        self.currency_pairs: List[str] = self.CURRENCY_PAIRS.copy()
        self.backtest_mode: bool = False
        self.backtest_period_days: int = self.DEFAULT_BACKTEST_DAYS
        self.news_api_url: str = "https://financialmodelingprep.com/api/v3/economic_calendar"
        self.news_api_key: str = os.getenv('FMP_API_KEY')  # Remove hardcoded key
        self.backup_api_url: str = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"
        self.backup_api_enabled: bool = True
        self.atr_multipliers: Dict[str, float] = {
            'low_vol': 2.0,
            'normal_vol': 2.5,
            'high_vol': 3.0,
            'xauusd': 3.5
        }
        self.volatility_thresholds: Dict[str, float] = {
            'low': 0.001,
            'high': 0.003,
            'xauusd': 0.005
        }
        self.initialized: bool = False
        self.api_calls_today: int = 0
        self.last_api_reset: datetime = datetime.now(timezone.utc)
        self._initialized: bool = True
    
    def __init__(self):
        """Initialize settings if not already initialized."""
        if not hasattr(self, '_initialized'):
            self._initialize_attributes()
        
        load_dotenv()
        self.fmp_api_key = os.getenv('FMP_API_KEY')
        self.twelve_data_key = os.getenv('TWELVE_DATA_KEY')
        
        # API rate limits
        self.twelve_data_daily_limit = 800  # Adjust based on your plan
        self.twelve_data_minute_limit = 8

    async def fetch_news_data(self, session: aiohttp.ClientSession, url: str, params: dict = None) -> Optional[dict]:
        """Fetch news data with proper error handling."""
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    try:
                        return await response.json()
                    except aiohttp.ContentTypeError:
                        logger.error(f"Invalid content type from {url}")
                        return None
                elif response.status == 401:
                    logger.error(f"Authentication failed for {url}")
                    return None
                elif response.status == 429:
                    logger.warning(f"Rate limit reached for {url}")
                    return None
                else:
                    logger.error(f"Failed to fetch news: Status {response.status}")
                    return None
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching news: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching news: {e}")
            return None

    async def initialize(self) -> None:
        """Initialize settings and validate API access."""
        if not self.initialized:
            if not self.news_api_key:
                logger.error("FMP API key not found in environment variables")
                raise ValueError("Missing API key")
            
            async with aiohttp.ClientSession() as session:
                # Try primary API
                primary_news = await self.fetch_news_data(
                    session, 
                    f"{self.news_api_url}",
                    {"apikey": self.news_api_key}
                )
                
                if primary_news:
                    logger.info("FMP API key validated successfully")
                    self.initialized = True
                    return
                
                # Try backup API if enabled
                if self.backup_api_enabled:
                    logger.info("Trying backup news source...")
                    backup_news = await self.fetch_news_data(
                        session,
                        self.backup_api_url
                    )
                    
                    if backup_news:
                        logger.info("Backup news source initialized")
                        self.initialized = True
                        return
                    
                    logger.error("Both primary and backup news sources failed")
                
                if not self.initialized and not self.backup_api_enabled:
                    raise ValueError("News API initialization failed")
    
    def get_currency_pairs(self) -> List[str]:
        """Return the list of currency pairs."""
        if not self.currency_pairs:
            logger.warning("Currency pairs list is empty")
        return self.currency_pairs.copy()  # Return a copy to prevent modification
    
    def increment_api_calls(self) -> bool:
        """Track API call count and reset daily. Returns False if limit reached."""
        today = datetime.now(timezone.utc).date()
        if today > self.last_api_reset.date():
            self.api_calls_today = 0
            self.last_api_reset = datetime.now(timezone.utc)
        
        if self.api_calls_today >= self.API_DAILY_LIMIT:
            logger.warning("Daily API call limit reached")
            return False
        
        self.api_calls_today += 1
        return True

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

# Trading-related runtime settings used by tests / position manager
trading = SimpleNamespace(
    risk_per_trade=0.01,     # default risk per trade (1%)
    max_total_risk=0.02,     # total account risk limit (2%)
    max_open_positions=5,
    default_lot_size=0.1
)

# Expose a top-level settings object expected by imports like `settings.trading`
# Keep this as the test-facing object and do NOT overwrite it with the Settings singleton.
settings = SimpleNamespace(trading=trading)

# Keep the Settings class instance available under a different name if needed by other code
app_settings = Settings()
