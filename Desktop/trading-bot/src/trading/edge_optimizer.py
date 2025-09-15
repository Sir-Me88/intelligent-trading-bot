#!/usr/bin/env python3
"""Edge Computing Optimization for Low-Latency Trading Executions."""

import asyncio
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

class EdgeSentimentPrefetcher:
    """Prefetches sentiment analysis for low-latency execution during high volatility."""

    def __init__(self, sentiment_aggregator, prefetch_window_seconds: int = 10):
        self.sentiment_aggregator = sentiment_aggregator
        self.prefetch_window_seconds = prefetch_window_seconds
        self.prefetch_cache: Dict[str, Dict] = {}
        self.prefetch_timestamps: Dict[str, datetime] = {}
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sentiment_prefetch")
        self.prefetch_task: Optional[asyncio.Task] = None
        self.is_running = False

        # Performance metrics
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.avg_prefetch_time = 0.0
        self.prefetch_count = 0

    async def start_prefetching(self, currency_pairs: List[str]):
        """Start background prefetching for specified currency pairs."""
        if self.is_running:
            logger.warning("Prefetcher already running")
            return

        self.is_running = True
        logger.info(f"Starting edge sentiment prefetching for {len(currency_pairs)} pairs")

        self.prefetch_task = asyncio.create_task(self._prefetch_loop(currency_pairs))

    async def stop_prefetching(self):
        """Stop background prefetching."""
        self.is_running = False
        if self.prefetch_task:
            self.prefetch_task.cancel()
            try:
                await self.prefetch_task
            except asyncio.CancelledError:
                pass
        self.executor.shutdown(wait=True)
        logger.info("Edge sentiment prefetching stopped")

    async def _prefetch_loop(self, currency_pairs: List[str]):
        """Background prefetching loop."""
        while self.is_running:
            try:
                # Prefetch all pairs in parallel
                tasks = []
                for pair in currency_pairs:
                    if self._should_prefetch(pair):
                        tasks.append(self._prefetch_pair_sentiment(pair))

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                # Wait before next prefetch cycle
                await asyncio.sleep(self.prefetch_window_seconds / 2)  # Overlap prefetch windows

            except Exception as e:
                logger.error(f"Error in prefetch loop: {e}")
                await asyncio.sleep(5)  # Brief pause on error

    def _should_prefetch(self, pair: str) -> bool:
        """Determine if sentiment should be prefetched for this pair."""
        if pair not in self.prefetch_timestamps:
            return True

        # Check if cache is stale
        time_since_last_prefetch = (datetime.now() - self.prefetch_timestamps[pair]).total_seconds()
        return time_since_last_prefetch >= (self.prefetch_window_seconds * 0.8)  # 80% of window

    async def _prefetch_pair_sentiment(self, pair: str):
        """Prefetch sentiment for a specific pair."""
        try:
            start_time = time.time()

            # Run sentiment analysis in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            sentiment_result = await loop.run_in_executor(
                self.executor,
                self._sync_get_sentiment,
                pair
            )

            # Cache the result
            self.prefetch_cache[pair] = sentiment_result
            self.prefetch_timestamps[pair] = datetime.now()

            # Update performance metrics
            prefetch_time = time.time() - start_time
            self.prefetch_count += 1
            self.avg_prefetch_time = ((self.avg_prefetch_time * (self.prefetch_count - 1)) + prefetch_time) / self.prefetch_count

            logger.debug(f"Prefetched sentiment for {pair}: {sentiment_result.get('overall_sentiment', 0):.3f} ({prefetch_time:.3f}s)")

        except Exception as e:
            logger.error(f"Failed to prefetch sentiment for {pair}: {e}")

    def _sync_get_sentiment(self, pair: str) -> Dict:
        """Synchronous wrapper for sentiment analysis."""
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async sentiment analysis
            result = loop.run_until_complete(self.sentiment_aggregator.get_overall_sentiment(pair))
            return result
        finally:
            loop.close()

    def get_cached_sentiment(self, pair: str) -> Optional[Dict]:
        """Get cached sentiment if available and fresh."""
        if pair not in self.prefetch_cache or pair not in self.prefetch_timestamps:
            self.prefetch_misses += 1
            return None

        # Check if cache is still fresh
        cache_age = (datetime.now() - self.prefetch_timestamps[pair]).total_seconds()
        if cache_age > self.prefetch_window_seconds:
            self.prefetch_misses += 1
            return None

        self.prefetch_hits += 1
        return self.prefetch_cache[pair]

    def get_performance_stats(self) -> Dict:
        """Get prefetching performance statistics."""
        total_requests = self.prefetch_hits + self.prefetch_misses
        hit_rate = (self.prefetch_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            'prefetch_hits': self.prefetch_hits,
            'prefetch_misses': self.prefetch_misses,
            'hit_rate_percent': hit_rate,
            'avg_prefetch_time_seconds': self.avg_prefetch_time,
            'total_prefetches': self.prefetch_count,
            'cache_size': len(self.prefetch_cache),
            'prefetch_window_seconds': self.prefetch_window_seconds
        }


class LatencyOptimizer:
    """Optimizes execution latency during high-volatility periods."""

    def __init__(self, broker_interface, sentiment_prefetcher: EdgeSentimentPrefetcher):
        self.broker = broker_interface
        self.sentiment_prefetcher = sentiment_prefetcher
        self.latency_threshold_ms = 100  # Target latency in milliseconds
        self.volatility_multiplier = 1.5  # Latency reduction during high vol

        # Performance tracking
        self.execution_times: List[float] = []
        self.slippage_events = 0
        self.total_executions = 0

    async def execute_with_optimization(self, order_params: Dict) -> Dict:
        """Execute order with latency optimizations."""
        start_time = time.time()
        pair = order_params.get('pair', '')

        try:
            # Check for cached sentiment (edge computing optimization)
            cached_sentiment = self.sentiment_prefetcher.get_cached_sentiment(pair)

            if cached_sentiment:
                # Use cached sentiment for faster decision making
                order_params['sentiment_context'] = cached_sentiment
                logger.debug(f"Using cached sentiment for {pair} (latency optimization)")

            # Execute with semaphore for parallel processing control
            semaphore = asyncio.Semaphore(10)  # Limit concurrent executions
            async with semaphore:
                result = await self.broker.place_order(order_params)

            # Track performance
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.execution_times.append(execution_time)
            self.total_executions += 1

            # Check for slippage
            if execution_time > self.latency_threshold_ms:
                self.slippage_events += 1
                logger.warning(f"High latency execution: {execution_time:.1f}ms for {pair}")

            result['execution_time_ms'] = execution_time
            result['used_cached_sentiment'] = cached_sentiment is not None

            return result

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Execution failed after {execution_time:.1f}ms: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time_ms': execution_time
            }

    def get_latency_stats(self) -> Dict:
        """Get latency performance statistics."""
        if not self.execution_times:
            return {'message': 'No execution data available'}

        avg_latency = sum(self.execution_times) / len(self.execution_times)
        max_latency = max(self.execution_times)
        min_latency = min(self.execution_times)

        # Calculate slippage rate
        slippage_rate = (self.slippage_events / self.total_executions * 100) if self.total_executions > 0 else 0

        return {
            'average_latency_ms': avg_latency,
            'max_latency_ms': max_latency,
            'min_latency_ms': min_latency,
            'total_executions': self.total_executions,
            'slippage_events': self.slippage_events,
            'slippage_rate_percent': slippage_rate,
            'latency_threshold_ms': self.latency_threshold_ms,
            'performance_rating': self._calculate_performance_rating(avg_latency, slippage_rate)
        }

    def _calculate_performance_rating(self, avg_latency: float, slippage_rate: float) -> str:
        """Calculate performance rating based on latency and slippage."""
        if avg_latency < 50 and slippage_rate < 5:
            return "EXCELLENT"
        elif avg_latency < 100 and slippage_rate < 10:
            return "GOOD"
        elif avg_latency < 200 and slippage_rate < 20:
            return "FAIR"
        else:
            return "NEEDS_IMPROVEMENT"

    def adjust_latency_settings(self, market_conditions: Dict):
        """Dynamically adjust latency settings based on market conditions."""
        volatility = market_conditions.get('volatility', 0.001)
        volume = market_conditions.get('volume', 100)

        # Increase prefetch frequency during high volatility
        if volatility > 0.003:  # High volatility threshold
            self.sentiment_prefetcher.prefetch_window_seconds = max(5, self.sentiment_prefetcher.prefetch_window_seconds * 0.8)
            logger.info(f"Reduced prefetch window to {self.sentiment_prefetcher.prefetch_window_seconds}s (high volatility)")

        # Adjust latency threshold based on market conditions
        if volume > 1000:  # High volume
            self.latency_threshold_ms = 150  # More lenient during high volume
        else:
            self.latency_threshold_ms = 100  # Stricter during normal conditions


class EdgeComputingManager:
    """Manages edge computing optimizations for the trading system."""

    def __init__(self, broker_interface, sentiment_aggregator):
        self.broker = broker_interface
        self.sentiment_aggregator = sentiment_aggregator

        # Initialize components
        self.sentiment_prefetcher = EdgeSentimentPrefetcher(sentiment_aggregator)
        self.latency_optimizer = LatencyOptimizer(broker_interface, self.sentiment_prefetcher)

        # Configuration
        self.active_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
        self.is_optimized = False

    async def enable_edge_optimizations(self):
        """Enable all edge computing optimizations."""
        if self.is_optimized:
            logger.warning("Edge optimizations already enabled")
            return

        logger.info("Enabling edge computing optimizations for low-latency trading")

        # Start sentiment prefetching
        await self.sentiment_prefetcher.start_prefetching(self.active_pairs)

        self.is_optimized = True
        logger.info("Edge computing optimizations enabled successfully")

    async def disable_edge_optimizations(self):
        """Disable all edge computing optimizations."""
        if not self.is_optimized:
            return

        logger.info("Disabling edge computing optimizations")

        # Stop sentiment prefetching
        await self.sentiment_prefetcher.stop_prefetching()

        self.is_optimized = False
        logger.info("Edge computing optimizations disabled")

    async def execute_optimized_order(self, order_params: Dict) -> Dict:
        """Execute order with full edge computing optimization."""
        if not self.is_optimized:
            # Fallback to normal execution
            return await self.broker.place_order(order_params)

        # Use latency optimizer for execution
        return await self.latency_optimizer.execute_with_optimization(order_params)

    def update_active_pairs(self, new_pairs: List[str]):
        """Update the list of actively traded pairs for prefetching."""
        self.active_pairs = new_pairs
        logger.info(f"Updated active pairs for edge optimization: {new_pairs}")

        # Restart prefetching with new pairs if currently running
        if self.is_optimized:
            asyncio.create_task(self._restart_prefetching())

    async def _restart_prefetching(self):
        """Restart prefetching with updated pairs."""
        await self.sentiment_prefetcher.stop_prefetching()
        await self.sentiment_prefetcher.start_prefetching(self.active_pairs)

    def get_system_status(self) -> Dict:
        """Get comprehensive status of edge computing system."""
        return {
            'edge_optimizations_enabled': self.is_optimized,
            'active_pairs': self.active_pairs,
            'sentiment_prefetch_stats': self.sentiment_prefetcher.get_performance_stats(),
            'latency_stats': self.latency_optimizer.get_latency_stats(),
            'timestamp': datetime.now().isoformat()
        }

    def log_performance_report(self):
        """Log comprehensive performance report."""
        status = self.get_system_status()

        logger.info("="*60)
        logger.info("EDGE COMPUTING PERFORMANCE REPORT")
        logger.info("="*60)
        logger.info(f"Optimizations Enabled: {status['edge_optimizations_enabled']}")
        logger.info(f"Active Pairs: {', '.join(status['active_pairs'])}")

        prefetch = status['sentiment_prefetch_stats']
        logger.info(f"Prefetch Hit Rate: {prefetch['hit_rate_percent']:.1f}%")
        logger.info(f"Avg Prefetch Time: {prefetch['avg_prefetch_time_seconds']:.3f}s")

        latency = status['latency_stats']
        if isinstance(latency, dict) and 'average_latency_ms' in latency:
            logger.info(f"Avg Execution Latency: {latency['average_latency_ms']:.1f}ms")
            logger.info(f"Slippage Rate: {latency['slippage_rate_percent']:.1f}%")
            logger.info(f"Performance Rating: {latency.get('performance_rating', 'UNKNOWN')}")

        logger.info("="*60)
