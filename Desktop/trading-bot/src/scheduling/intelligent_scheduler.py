#!/usr/bin/env python3
"""Intelligent trading scheduler for adaptive bot."""

import asyncio
import logging
from datetime import datetime, time, timezone, timedelta
from typing import Callable, Optional, Dict, List
import aiohttp
from src.config.settings import settings
from src.news.economic_calendar import EconomicCalendarFilter

logger = logging.getLogger(__name__)

class IntelligentTradingScheduler:
    """Manages trading schedule with intelligent mode switching."""
    
    def __init__(self):
        self.mode: str = 'TRADING'
        self.mode_change_callback: Optional[Callable] = None
        self.trading_hours = {
            'start': time(0, 0),  # 00:00 UTC
            'end': time(23, 59)   # 23:59 UTC
        }
        self.analysis_hours = {
            'daily': time(22, 0),  # US-Japan gap
            'weekend': time(0, 0)  # Saturday
        }
        self.economic_calendar = EconomicCalendarFilter()
        self.api_retries = 0
        self.max_retries = 3
        self.retry_delay = 60  # seconds
        self.using_backup = False
        self.initialized = False

        # Drawdown circuit breaker
        self.drawdown_threshold = 0.10  # 10% drawdown triggers pause
        self.drawdown_check_enabled = True
        self.last_drawdown_check = None
        self.drawdown_pause_duration = 3600  # 1 hour pause
        self.drawdown_pause_until = None

    async def initialize(self) -> None:
        """Initialize scheduler with API validation."""
        if not self.initialized:
            try:
                # Validate FMP API key
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{settings.news_api_url}?apikey={settings.news_api_key}") as response:
                        if response.status == 401:
                            logger.error("Invalid FMP API key. Generate new key at financialmodelingprep.com")
                            await self._init_backup_source()
                        elif response.status == 429:
                            logger.warning("FMP API limit reached, switching to backup source")
                            await self._init_backup_source()
                        elif response.status == 200:
                            logger.info("FMP API key validated successfully")
                            await self.economic_calendar.initialize()
                
                self.initialized = True
            except Exception as e:
                logger.error(f"Scheduler initialization failed: {e}")
                await self._init_backup_source()

    async def _init_backup_source(self) -> None:
        """Initialize backup news source."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0',
                    'Accept': 'application/json'
                }
                url = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"
                async with session.post(url, headers=headers, data={"timeZone": 0, "timeFilter": "today"}) as response:
                    if response.status == 200:
                        logger.info("Backup news source initialized")
                        self.using_backup = True
                        return
            logger.error("Failed to initialize backup source")
        except Exception as e:
            logger.error(f"Backup source initialization failed: {e}")
            raise

    async def _fetch_news_events(self) -> List[Dict]:
        """Fetch news events with fallback mechanism."""
        if not self.using_backup:
            try:
                async with aiohttp.ClientSession() as session:
                    url = f"{settings.news_api_url}?apikey={settings.news_api_key}&date={datetime.now(timezone.utc).date()}"
                    async with session.get(url) as response:
                        if response.status == 200:
                            return await response.json()
                        if response.status in (401, 429):
                            self.using_backup = True
            except Exception as e:
                logger.error(f"Primary API fetch failed: {e}")
                self.using_backup = True

        # Fallback to backup source
        return await self._fetch_backup_events()

    async def _fetch_backup_events(self) -> List[Dict]:
        """Fetch events from backup source."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
                url = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"
                async with session.post(url, headers=headers, data={"timeZone": 0, "timeFilter": "today"}) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._convert_backup_format(data.get("data", []))
            return []
        except Exception as e:
            logger.error(f"Backup fetch failed: {e}")
            return []

    def _convert_backup_format(self, events: List[Dict]) -> List[Dict]:
        """Convert backup source format to standard format."""
        converted = []
        for event in events:
            try:
                converted.append({
                    "event": event.get("event_name", ""),
                    "currency": event.get("currency", ""),
                    "impact": "high" if event.get("importance", 0) == 3 else "medium",
                    "time": event.get("date_time", ""),
                    "actual": event.get("actual", ""),
                    "forecast": event.get("forecast", ""),
                    "previous": event.get("previous", "")
                })
            except Exception as e:
                logger.warning(f"Error converting event format: {e}")
                continue
        return converted

    async def check_news_conflicts(self, pair: str) -> Dict:
        """Check for news conflicts that would prevent trading."""
        try:
            if not self.initialized:
                await self.initialize()

            news_check = await self.economic_calendar.should_skip_pair(pair)
            if not isinstance(news_check, dict):
                logger.error(f"Invalid news check response for {pair}: {news_check}")
                return {
                    'should_skip': False,
                    'reason': 'Invalid news response',
                    'conflicting_events': []
                }
            return news_check
        except Exception as e:
            logger.error(f"News check error for {pair}: {e}")
            return {
                'should_skip': False,
                'reason': f'News check failed: {e}',
                'conflicting_events': []
            }

    def register_mode_change_callback(self, callback: Callable[[str, str], None]):
        """Register callback for mode changes."""
        self.mode_change_callback = callback

    async def should_execute_trades(self, broker_manager=None) -> bool:
        """Determine if trading should occur based on schedule, news, and drawdown."""
        if settings.backtest_mode:
            return True  # Always trade in backtest mode

        current_time = datetime.utcnow().time()
        current_day = datetime.utcnow().weekday()

        if not (self.trading_hours['start'] <= current_time <= self.trading_hours['end']):
            return False

        if current_day == 5 and current_time >= self.analysis_hours['weekend']:  # Saturday
            self.mode = 'WEEKEND_ANALYSIS'
            return False
        elif current_time >= self.analysis_hours['daily']:
            self.mode = 'DAILY_ANALYSIS'
            return False

        if await self.pause_for_news():
            return False

        # Check drawdown circuit breaker
        if broker_manager and await self.check_drawdown_circuit_breaker(broker_manager):
            self.mode = 'PAUSED_DRAWDOWN'
            return False

        self.mode = 'TRADING'
        return True

    async def pause_for_news(self) -> bool:
        """Check if trading should pause due to high-impact news."""
        try:
            if not self.initialized:
                await self.initialize()

            events = await self._fetch_news_events()
            if not isinstance(events, list):
                logger.error(f"Invalid events data type: {type(events)}")
                return False

            now = datetime.now(timezone.utc)
            for event in events:
                if not isinstance(event, dict):
                    continue
                
                event_time = event.get('time')
                impact = event.get('impact', '').lower()
                
                if not event_time or not impact:
                    continue

                try:
                    event_time = datetime.fromisoformat(
                        event_time.replace('Z', '+00:00')
                    ) if isinstance(event_time, str) else event_time
                    
                    if (impact == 'high' and 
                        abs((now - event_time).total_seconds()) < 3600):
                        logger.warning(f"Pausing trading due to high-impact event: {event.get('event')}")
                        return True
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Error parsing event time: {e}")
                    continue

            return False
        except Exception as e:
            logger.error(f"News check failed: {e}")
            return False

    async def monitor_mode_changes(self):
        """Monitor for mode changes and trigger callbacks."""
        while True:
            try:
                current_mode = self.mode
                new_mode = 'TRADING' if await self.should_execute_trades() else self.mode
                
                if current_mode != new_mode and self.mode_change_callback:
                    await self.mode_change_callback(current_mode, new_mode)
                    self.mode = new_mode
                    
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in mode monitoring: {e}")
                await asyncio.sleep(60)
    
    def get_trading_schedule_info(self) -> str:
        """Return current schedule information."""
        drawdown_info = ""
        if self.drawdown_pause_until:
            remaining = max(0, (self.drawdown_pause_until - datetime.now(timezone.utc)).total_seconds() / 60)
            drawdown_info = f", Drawdown Pause: {remaining:.1f}min remaining"

        return f"Mode: {self.mode}, Trading Hours: {self.trading_hours['start']}-{self.trading_hours['end']}{drawdown_info}"

    async def check_drawdown_circuit_breaker(self, broker_manager) -> bool:
        """Check if drawdown circuit breaker should be triggered."""
        if not self.drawdown_check_enabled:
            return False

        try:
            # Rate limit drawdown checks to every 5 minutes
            now = datetime.now(timezone.utc)
            if (self.last_drawdown_check and
                (now - self.last_drawdown_check).total_seconds() < 300):
                return False

            self.last_drawdown_check = now

            # Check if we're still in a drawdown pause
            if self.drawdown_pause_until and now < self.drawdown_pause_until:
                remaining_time = (self.drawdown_pause_until - now).total_seconds() / 60
                logger.info(f"Drawdown circuit breaker active: {remaining_time:.1f} minutes remaining")
                return True

            # Get account information
            account_info = await broker_manager.get_account_info()
            if not account_info:
                logger.warning("Unable to get account info for drawdown check")
                return False

            balance = account_info.get('balance', 0)
            equity = account_info.get('equity', 0)

            if balance <= 0:
                logger.error("Invalid balance for drawdown calculation")
                return False

            # Calculate drawdown percentage
            drawdown = (balance - equity) / balance

            logger.debug(f"Drawdown check: Balance ${balance:.2f}, Equity ${equity:.2f}, Drawdown {drawdown:.2%}")

            # Trigger circuit breaker if drawdown exceeds threshold
            if drawdown >= self.drawdown_threshold:
                await self._trigger_drawdown_circuit_breaker(drawdown, broker_manager)
                return True

            # Reset pause if drawdown has recovered
            if self.drawdown_pause_until and drawdown < (self.drawdown_threshold * 0.5):
                logger.info(f"Drawdown recovered to {drawdown:.2%}, resuming trading")
                self.drawdown_pause_until = None

            return False

        except Exception as e:
            logger.error(f"Error in drawdown circuit breaker check: {e}")
            return False

    async def _trigger_drawdown_circuit_breaker(self, drawdown: float, broker_manager):
        """Trigger the drawdown circuit breaker and initiate recovery procedures."""
        try:
            logger.critical(f"🚨 DRAWDOWN CIRCUIT BREAKER TRIGGERED: {drawdown:.2%} >= {self.drawdown_threshold:.2%}")

            # Set pause mode
            old_mode = self.mode
            self.mode = 'PAUSED_DRAWDOWN'
            self.drawdown_pause_until = datetime.now(timezone.utc) + timedelta(seconds=self.drawdown_pause_duration)

            # Close all positions immediately for safety
            try:
                positions = await broker_manager.get_positions()
                if positions:
                    logger.warning(f"Closing {len(positions)} positions due to drawdown circuit breaker")
                    for position in positions:
                        ticket = position.get('ticket')
                        if ticket:
                            await broker_manager.close_position(ticket)
                            logger.info(f"Closed position #{ticket} due to drawdown protection")
            except Exception as e:
                logger.error(f"Error closing positions during drawdown: {e}")

            # Trigger callback for mode change
            if self.mode_change_callback:
                await self.mode_change_callback(old_mode, self.mode)

            # Trigger RL retraining for strategy optimization
            try:
                from src.ml.trading_ml_engine import TradingMLEngine
                ml_engine = TradingMLEngine()

                logger.info("🔄 Triggering RL retraining due to drawdown circuit breaker")
                rl_results = await ml_engine.train_rl_agent({})

                if rl_results.get('status') == 'TRAINED':
                    logger.info("✅ RL retraining completed after drawdown event")
                else:
                    logger.warning(f"RL retraining failed: {rl_results.get('error', 'Unknown error')}")

            except Exception as e:
                logger.error(f"Error during RL retraining: {e}")

            # Log circuit breaker activation
            logger.critical(f"Drawdown circuit breaker activated for {self.drawdown_pause_duration/3600:.1f} hours")
            logger.critical(f"Trading will resume after: {self.drawdown_pause_until}")

        except Exception as e:
            logger.error(f"Error triggering drawdown circuit breaker: {e}")

    def configure_drawdown_settings(self, threshold: float = None, pause_duration: int = None,
                                  enable_check: bool = None):
        """Configure drawdown circuit breaker settings."""
        if threshold is not None:
            self.drawdown_threshold = max(0.01, min(0.50, threshold))  # 1% to 50%
            logger.info(f"Drawdown threshold set to {self.drawdown_threshold:.2%}")

        if pause_duration is not None:
            self.drawdown_pause_duration = max(300, min(86400, pause_duration))  # 5min to 24hrs
            logger.info(f"Drawdown pause duration set to {self.drawdown_pause_duration/3600:.1f} hours")

        if enable_check is not None:
            self.drawdown_check_enabled = enable_check
            logger.info(f"Drawdown circuit breaker {'enabled' if enable_check else 'disabled'}")

    def get_drawdown_status(self) -> Dict:
        """Get current drawdown circuit breaker status."""
        now = datetime.now(timezone.utc)

        if self.drawdown_pause_until and now < self.drawdown_pause_until:
            remaining_seconds = (self.drawdown_pause_until - now).total_seconds()
            return {
                'status': 'ACTIVE',
                'pause_remaining_seconds': int(remaining_seconds),
                'pause_remaining_minutes': round(remaining_seconds / 60, 1),
                'threshold': self.drawdown_threshold,
                'enabled': self.drawdown_check_enabled
            }
        else:
            return {
                'status': 'INACTIVE',
                'threshold': self.drawdown_threshold,
                'enabled': self.drawdown_check_enabled,
                'last_check': self.last_drawdown_check.isoformat() if self.last_drawdown_check else None
            }
