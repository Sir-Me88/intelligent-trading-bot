"""Economic calendar and news filtering."""

from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import logging
import aiohttp
import asyncio

from ..config.settings import settings

logger = logging.getLogger(__name__)

# Constants
IMPACT_LEVELS = {
    'low': 1,
    'medium': 2,
    'high': 3
}
DEFAULT_TIME_BUFFER_HOURS = 2
DEFAULT_CACHE_MAX_AGE_HOURS = 6
MAX_API_RETRIES = 3
CALLS_PER_MINUTE = 30

class RateLimiter:
    """Simple rate limiter for API calls."""
    def __init__(self, calls_per_minute: int = CALLS_PER_MINUTE):
        self.calls_per_minute = calls_per_minute
        self.calls: List[datetime] = []
    
    async def wait_if_needed(self):
        """Wait if rate limit is reached."""
        now = datetime.now()
        self.calls = [t for t in self.calls if (now - t).total_seconds() < 60]
        if len(self.calls) >= self.calls_per_minute:
            wait_time = 60 - (now - self.calls[0]).total_seconds()
            await asyncio.sleep(wait_time)
        self.calls.append(now)

class EconomicEvent:
    """Economic calendar event."""
    
    def __init__(self, title: str, currency: str, impact: str, 
                 time: datetime, actual: Optional[str] = None,
                 forecast: Optional[str] = None, previous: Optional[str] = None):
        self.title = title
        self.currency = currency.upper()
        self.impact = impact.lower()
        self.time = time
        self.actual = actual
        self.forecast = forecast
        self.previous = previous
    
    def affects_currency_pair(self, pair: str) -> bool:
        """Check if event affects a currency pair."""
        if not self._validate_currency_pair(pair):
            logger.warning(f"Invalid currency pair format: {pair}")
            return False
        base_currency = pair[:3]
        quote_currency = pair[3:]
        return self.currency in [base_currency, quote_currency]
    
    @staticmethod
    def _validate_currency_pair(pair: str) -> bool:
        """Validate currency pair format."""
        return len(pair) == 6 and pair.isalpha() and pair.isupper()
    
    def is_high_impact(self) -> bool:
        """Check if event is high impact."""
        return self.impact == 'high'
    
    def is_within_time_buffer(self, buffer_hours: int = DEFAULT_TIME_BUFFER_HOURS) -> bool:
        """Check if event is within time buffer of current time."""
        now = datetime.now()
        time_diff = abs((self.time - now).total_seconds() / 3600)
        return time_diff <= buffer_hours

class EconomicCalendarFilter:
    """Filters trading signals based on economic events (FMP API)."""

    def __init__(self):
        self.cached_events: List[EconomicEvent] = []
        self.last_update: Optional[datetime] = None
        self.initialized = False
        self.rate_limiter = RateLimiter()
    
    async def initialize(self):
        """Initialize the calendar with first event fetch."""
        if not self.initialized:
            await self.update_events()
            self.initialized = True

    async def update_events(self, days_ahead: int = 1):
        """Fetch events from FMP API."""
        try:
            async with aiohttp.ClientSession() as session:
                from_date = datetime.now(timezone.utc).date()
                to_date = from_date + timedelta(days=days_ahead)
                url = (
                    f"{settings.news_api_url}?from={from_date}&to={to_date}"
                    f"&apikey={settings.news_api_key}"
                )
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"FMP API error: {response.status}")
                        return
                    data = await response.json()
                    self.cached_events = []
                    for item in data:
                        try:
                            event_time = datetime.fromisoformat(
                                item["date"].replace('Z', '+00:00')
                            )
                            event = EconomicEvent(
                                title=item["event"],
                                currency=item["currency"],
                                impact=item["impact"],
                                time=event_time,
                                actual=item.get("actual"),
                                forecast=item.get("forecast"),
                                previous=item.get("previous")
                            )
                            self.cached_events.append(event)
                        except (ValueError, KeyError) as e:
                            logger.error(f"Error processing event data: {e}")
                            continue
                    
                    self.last_update = datetime.now(timezone.utc)
                    logger.info(f"Fetched {len(self.cached_events)} events from FMP")
        except Exception as e:
            logger.error(f"FMP fetch failed: {e}")

    def _process_api_response(self, data: List[Dict]) -> None:
        """Process API response and update cached events."""
        self.cached_events = []
        for item in data:
            if not all(k in item for k in ("event", "currency", "impact", "date")):
                logger.warning(f"Skipping incomplete event data: {item}")
                continue
            try:
                event = EconomicEvent(
                    title=item["event"],
                    currency=item["currency"],
                    impact=item["impact"],
                    time=datetime.fromisoformat(item["date"]),
                    actual=item.get("actual"),
                    forecast=item.get("forecast"),
                    previous=item.get("previous"),
                )
                self.cached_events.append(event)
            except (ValueError, KeyError) as e:
                logger.error(f"Error processing event: {e}")
                continue
        
        self.last_update = datetime.now()
        logger.info(f"Fetched {len(self.cached_events)} events from FMP")

    def should_skip_pair(self, currency_pair: str, 
                        impact_threshold: str = "high",
                        time_buffer_hours: int = 2) -> Dict:
        """Check if trading should be skipped for a currency pair."""
        result = {
            'should_skip': False,
            'reason': '',
            'conflicting_events': []
        }
        
        if not self.cached_events:
            return result
        
        # Check for conflicting events
        conflicting_events = []
        
        for event in self.cached_events:
            if not event.affects_currency_pair(currency_pair):
                continue
            
            if not event.is_within_time_buffer(time_buffer_hours):
                continue
            
            # Check impact threshold
            event_impact_level = IMPACT_LEVELS.get(event.impact, 0)
            threshold_level = IMPACT_LEVELS.get(impact_threshold.lower(), 3)
            
            if event_impact_level >= threshold_level:
                conflicting_events.append({
                    'title': event.title,
                    'currency': event.currency,
                    'impact': event.impact,
                    'time': event.time.isoformat(),
                    'time_until': (event.time - datetime.now()).total_seconds() / 3600
                })
        
        if conflicting_events:
            result['should_skip'] = True
            result['reason'] = f"High-impact {conflicting_events[0]['currency']} event within {time_buffer_hours}h"
            result['conflicting_events'] = conflicting_events
        
        return result
    
    def get_upcoming_events(self, currency_pair: str, hours_ahead: int = 24) -> List[Dict]:
        """Get upcoming events for a currency pair."""
        upcoming = []
        cutoff_time = datetime.now() + timedelta(hours=hours_ahead)
        
        for event in self.cached_events:
            if (event.affects_currency_pair(currency_pair) and 
                event.time <= cutoff_time and 
                event.time >= datetime.now()):
                upcoming.append({
                    'title': event.title,
                    'currency': event.currency,
                    'impact': event.impact,
                    'time': event.time.isoformat(),
                    'hours_until': (event.time - datetime.now()).total_seconds() / 3600,
                    'forecast': event.forecast,
                    'previous': event.previous
                })
        
        return sorted(upcoming, key=lambda x: x['hours_until'])
    
    def is_cache_stale(self, max_age_hours: int = DEFAULT_CACHE_MAX_AGE_HOURS) -> bool:
        """Check if event cache needs updating."""
        if self.last_update is None:
            return True
        
        age = datetime.now() - self.last_update
        return age > timedelta(hours=max_age_hours)
