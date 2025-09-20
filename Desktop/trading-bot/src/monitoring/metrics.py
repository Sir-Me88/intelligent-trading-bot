"""Advanced monitoring and metrics collection system."""

import time
import psutil
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and exposes Prometheus metrics."""

    def __init__(self):
        # Trading metrics
        self.trades_total = Counter('forex_bot_trades_total', 'Total number of trades', ['pair', 'direction'])
        self.trades_profit = Counter('forex_bot_trades_profit_total', 'Total profit from trades', ['pair'])
        self.trades_loss = Counter('forex_bot_trades_loss_total', 'Total loss from trades', ['pair'])

        # Account metrics
        self.account_equity = Gauge('forex_bot_account_equity', 'Account equity')
        self.account_balance = Gauge('forex_bot_account_balance', 'Account balance')
        self.unrealized_pnl = Gauge('forex_bot_unrealized_pnl', 'Unrealized PnL')
        self.margin_used = Gauge('forex_bot_margin_used', 'Margin used')
        
        # Position metrics
        self.open_positions = Gauge('forex_bot_open_positions', 'Number of open positions')
        self.total_risk = Gauge('forex_bot_total_risk', 'Total risk exposure')
        
        # Performance metrics
        self.loop_duration = Histogram('forex_bot_loop_duration_seconds', 'Main loop duration')
        self.api_latency = Histogram('forex_bot_api_latency_seconds', 'API call latency', ['endpoint'])
        
        # Error metrics
        self.errors_total = Counter('forex_bot_errors_total', 'Total number of errors', ['type'])
        
        # Sentiment metrics
        self.sentiment_score = Gauge('forex_bot_sentiment_score', 'Sentiment score', ['pair'])
        
        # Signal metrics
        self.signals_generated = Counter('forex_bot_signals_generated_total', 'Signals generated', ['pair', 'pattern'])
        self.signals_executed = Counter('forex_bot_signals_executed_total', 'Signals executed', ['pair'])

        # Advanced monitoring metrics
        self.system_health = Gauge('forex_bot_system_health', 'System health score (0-100)')
        self.memory_usage = Gauge('forex_bot_memory_usage_mb', 'Memory usage in MB')
        self.cpu_usage = Gauge('forex_bot_cpu_usage_percent', 'CPU usage percentage')
        self.disk_usage = Gauge('forex_bot_disk_usage_percent', 'Disk usage percentage')

        # Performance metrics
        self.scan_duration = Histogram('forex_bot_scan_duration_seconds', 'Trading scan duration')
        self.data_fetch_duration = Histogram('forex_bot_data_fetch_duration_seconds', 'Data fetch duration', ['pair'])
        self.signal_processing_duration = Histogram('forex_bot_signal_processing_duration_seconds', 'Signal processing duration')

        # ML metrics
        self.ml_predictions_total = Counter('forex_bot_ml_predictions_total', 'ML predictions made')
        self.ml_accuracy = Gauge('forex_bot_ml_accuracy', 'ML model accuracy')
        self.rl_rewards_total = Counter('forex_bot_rl_rewards_total', 'RL rewards accumulated')

        # Risk metrics
        self.portfolio_volatility = Gauge('forex_bot_portfolio_volatility', 'Portfolio volatility')
        self.max_drawdown = Gauge('forex_bot_max_drawdown', 'Maximum drawdown percentage')
        self.sharpe_ratio = Gauge('forex_bot_sharpe_ratio', 'Sharpe ratio')

        # Connectivity metrics
        self.api_connectivity = Gauge('forex_bot_api_connectivity', 'API connectivity status (0=down, 1=up)', ['service'])
        self.data_feed_status = Gauge('forex_bot_data_feed_status', 'Data feed status', ['feed'])

        # Alert metrics
        self.alerts_triggered = Counter('forex_bot_alerts_triggered_total', 'Alerts triggered', ['type', 'severity'])

        self.server_started = False
        self.health_check_interval = 30  # seconds
        self.last_health_check = 0
        self._last_system_health_value = 100  # Default to 100 at startup
    
    async def start(self):
        """Start Prometheus metrics server."""
        if not self.server_started:
            try:
                start_http_server(8000)  # Default Prometheus port
                self.server_started = True
                logger.info("Prometheus metrics server started on port 8000")
            except Exception as e:
                logger.error(f"Failed to start metrics server: {e}")
    
    async def stop(self):
        """Stop metrics collection."""
        # Prometheus server doesn't have a clean shutdown method
        logger.info("Metrics collection stopped")
    
    def record_trade(self, signal: Dict, position_size: float, sentiment_data: Dict):
        """Record trade metrics. Robust to both enum and string direction types."""
        pair = signal.get('pair') or signal.get('symbol', 'UNKNOWN')
        direction = signal.get('direction')
        # Handle enum or string direction
        if hasattr(direction, 'value'):
            direction_value = direction.value
        else:
            direction_value = str(direction)
        pattern = signal.get('pattern', {})
        # Record trade count
        self.trades_total.labels(pair=pair, direction=direction_value).inc()
        # Record signal execution
        self.signals_executed.labels(pair=pair).inc()
        # Record pattern if available
        if pattern:
            pattern_name = pattern.value if hasattr(pattern, 'value') else str(pattern)
            self.signals_generated.labels(pair=pair, pattern=pattern_name).inc()
        # Record sentiment
        overall_sentiment = sentiment_data.get('overall_sentiment', 0)
        self.sentiment_score.labels(pair=pair).set(overall_sentiment)
        logger.debug(f"Recorded trade metrics for {pair}")
    
    def update_account_metrics(self, account_info: Dict):
        """Update account-related metrics."""
        self.account_equity.set(account_info.get('equity', 0))
        self.account_balance.set(account_info.get('balance', 0))
        self.unrealized_pnl.set(account_info.get('unrealized_pnl', 0))
        self.margin_used.set(account_info.get('margin_used', 0))
    
    def update_position_metrics(self, positions: list, total_risk: float):
        """Update position-related metrics."""
        self.open_positions.set(len(positions))
        self.total_risk.set(total_risk)
    
    def record_loop_duration(self, duration: float):
        """Record main loop duration."""
        self.loop_duration.observe(duration)
    
    def record_api_latency(self, endpoint: str, duration: float):
        """Record API call latency."""
        self.api_latency.labels(endpoint=endpoint).observe(duration)
    
    def increment_error_count(self, error_type: str = "general"):
        """Increment error counter."""
        self.errors_total.labels(type=error_type).inc()
    
    def record_pnl(self, pair: str, pnl: float):
        """Record profit/loss for a trade."""
        if pnl > 0:
            self.trades_profit.labels(pair=pair).inc(pnl)
        else:
            self.trades_loss.labels(pair=pair).inc(abs(pnl))

    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        current_time = time.time()

        if current_time - self.last_health_check < self.health_check_interval:
            return self._get_cached_health_status()

        try:
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'overall_health': 100,
                'components': {},
                'alerts': []
            }

            # System resource monitoring
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent(interval=1)
            disk_percent = psutil.disk_usage('/').percent

            self.memory_usage.set(psutil.virtual_memory().used / 1024 / 1024)  # MB
            self.cpu_usage.set(cpu_percent)
            self.disk_usage.set(disk_percent)

            # Health scoring based on resource usage
            health_score = 100

            if memory_percent > 90:
                health_score -= 30
                health_status['alerts'].append({
                    'type': 'CRITICAL',
                    'message': f'High memory usage: {memory_percent:.1f}%',
                    'severity': 'critical'
                })
            elif memory_percent > 80:
                health_score -= 15
                health_status['alerts'].append({
                    'type': 'WARNING',
                    'message': f'Elevated memory usage: {memory_percent:.1f}%',
                    'severity': 'warning'
                })

            if cpu_percent > 95:
                health_score -= 25
                health_status['alerts'].append({
                    'type': 'CRITICAL',
                    'message': f'High CPU usage: {cpu_percent:.1f}%',
                    'severity': 'critical'
                })
            elif cpu_percent > 80:
                health_score -= 10
                health_status['alerts'].append({
                    'type': 'WARNING',
                    'message': f'Elevated CPU usage: {cpu_percent:.1f}%',
                    'severity': 'warning'
                })

            if disk_percent > 95:
                health_score -= 20
                health_status['alerts'].append({
                    'type': 'CRITICAL',
                    'message': f'High disk usage: {disk_percent:.1f}%',
                    'severity': 'critical'
                })

            # Component health checks
            health_status['components'] = {
                'memory': {
                    'usage_percent': memory_percent,
                    'status': 'healthy' if memory_percent < 80 else 'warning' if memory_percent < 90 else 'critical'
                },
                'cpu': {
                    'usage_percent': cpu_percent,
                    'status': 'healthy' if cpu_percent < 80 else 'warning' if cpu_percent < 95 else 'critical'
                },
                'disk': {
                    'usage_percent': disk_percent,
                    'status': 'healthy' if disk_percent < 90 else 'warning' if disk_percent < 95 else 'critical'
                }
            }

            health_status['overall_health'] = max(0, health_score)
            self.system_health.set(health_status['overall_health'])
            self._last_system_health_value = health_status['overall_health']

            # Trigger alerts for critical issues
            for alert in health_status['alerts']:
                if alert['severity'] == 'critical':
                    self.alerts_triggered.labels(type=alert['type'], severity=alert['severity']).inc()
                    logger.critical(f"🚨 SYSTEM ALERT: {alert['message']}")

            self.last_health_check = current_time
            self._cache_health_status(health_status)

            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.increment_error_count("health_check")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_health': 0,
                'error': str(e),
                'components': {},
                'alerts': []
            }

    def _get_cached_health_status(self) -> Dict[str, Any]:
        """Get cached health status."""
        # Return basic cached status
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_health': getattr(self, '_last_system_health_value', 50),
            'cached': True
        }

    def _cache_health_status(self, status: Dict[str, Any]):
        """Cache health status for quick access."""
        # Simple in-memory cache
        self._last_health_status = status

    def record_scan_performance(self, scan_duration: float, pairs_processed: int,
                              signals_generated: int, trades_executed: int):
        """Record comprehensive scan performance metrics."""
        self.scan_duration.observe(scan_duration)

        # Calculate performance ratios
        signals_per_second = signals_generated / scan_duration if scan_duration > 0 else 0
        trades_per_second = trades_executed / scan_duration if scan_duration > 0 else 0
        pairs_per_second = pairs_processed / scan_duration if scan_duration > 0 else 0

        logger.info(f"📊 Scan Performance: {scan_duration:.2f}s, {pairs_processed} pairs, "
                   f"{signals_generated} signals, {trades_executed} trades")
        logger.info(f"   Rates: {pairs_per_second:.2f} pairs/s, {signals_per_second:.2f} signals/s, "
                   f"{trades_per_second:.2f} trades/s")

    def record_data_fetch_performance(self, pair: str, fetch_duration: float, data_points: int):
        """Record data fetching performance."""
        self.data_fetch_duration.labels(pair=pair).observe(fetch_duration)

        if fetch_duration > 5.0:  # Slow fetch alert
            self.alerts_triggered.labels(type='SLOW_DATA_FETCH', severity='warning').inc()
            logger.warning(f"🐌 Slow data fetch for {pair}: {fetch_duration:.2f}s")

    def record_signal_processing_performance(self, processing_duration: float, signals_processed: int):
        """Record signal processing performance."""
        self.signal_processing_duration.observe(processing_duration)

        signals_per_second = signals_processed / processing_duration if processing_duration > 0 else 0

        if processing_duration > 10.0:  # Slow processing alert
            self.alerts_triggered.labels(type='SLOW_SIGNAL_PROCESSING', severity='warning').inc()
            logger.warning(f"🐌 Slow signal processing: {processing_duration:.2f}s for {signals_processed} signals")

    def update_risk_metrics(self, portfolio_volatility: float, max_drawdown: float, sharpe_ratio: float):
        """Update portfolio risk metrics."""
        self.portfolio_volatility.set(portfolio_volatility)
        self.max_drawdown.set(max_drawdown)
        self.sharpe_ratio.set(sharpe_ratio)

        # Risk alerts
        if max_drawdown > 0.15:  # 15% drawdown
            self.alerts_triggered.labels(type='HIGH_DRAWDOWN', severity='critical').inc()
            logger.critical(f"🚨 HIGH DRAWDOWN ALERT: {max_drawdown:.2%}")

        if portfolio_volatility > 0.05:  # 5% daily volatility
            self.alerts_triggered.labels(type='HIGH_VOLATILITY', severity='warning').inc()
            logger.warning(f"⚠️ HIGH VOLATILITY ALERT: {portfolio_volatility:.2%}")

    def update_connectivity_status(self, service: str, status: bool):
        """Update service connectivity status."""
        self.api_connectivity.labels(service=service).set(1 if status else 0)

        if not status:
            self.alerts_triggered.labels(type='CONNECTIVITY_LOSS', severity='critical').inc()
            logger.critical(f"🚨 CONNECTIVITY LOSS: {service} is down")

    def update_data_feed_status(self, feed: str, status: bool, latency: Optional[float] = None):
        """Update data feed status."""
        self.data_feed_status.labels(feed=feed).set(1 if status else 0)

        if not status:
            self.alerts_triggered.labels(type='DATA_FEED_DOWN', severity='critical').inc()
            logger.critical(f"🚨 DATA FEED DOWN: {feed} is unavailable")

        if latency and latency > 2.0:  # High latency alert
            self.alerts_triggered.labels(type='HIGH_LATENCY', severity='warning').inc()
            logger.warning(f"⚠️ HIGH LATENCY: {feed} - {latency:.2f}s")

    def record_ml_performance(self, predictions_made: int, accuracy: float, rewards: float):
        """Record ML model performance metrics."""
        self.ml_predictions_total.inc(predictions_made)
        self.ml_accuracy.set(accuracy)
        self.rl_rewards_total.inc(rewards)

        if accuracy < 0.5:  # Low accuracy alert
            self.alerts_triggered.labels(type='LOW_ML_ACCURACY', severity='warning').inc()
            logger.warning(f"⚠️ LOW ML ACCURACY: {accuracy:.2%}")

    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        try:
            health_status = asyncio.run(self.perform_health_check())

            # Get recent metrics
            recent_metrics = {
                'trades_last_hour': self._get_recent_trade_count(3600),
                'errors_last_hour': self._get_recent_error_count(3600),
                'avg_scan_duration': self._get_avg_scan_duration(3600),
                'system_uptime': self._get_system_uptime()
            }

            report = {
                'timestamp': datetime.now().isoformat(),
                'health_score': health_status.get('overall_health', 0),
                'status': 'healthy' if health_status.get('overall_health', 0) > 70 else 'warning' if health_status.get('overall_health', 0) > 40 else 'critical',
                'components': health_status.get('components', {}),
                'alerts': health_status.get('alerts', []),
                'recent_metrics': recent_metrics,
                'recommendations': self._generate_health_recommendations(health_status, recent_metrics)
            }

            return report

        except Exception as e:
            logger.error(f"Failed to generate health report: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }

    def _get_recent_trade_count(self, seconds: int) -> int:
        """Get trade count in the last N seconds."""
        # This would require storing timestamps, simplified for now
        return 0

    def _get_recent_error_count(self, seconds: int) -> int:
        """Get error count in the last N seconds."""
        # This would require storing timestamps, simplified for now
        return 0

    def _get_avg_scan_duration(self, seconds: int) -> float:
        """Get average scan duration in the last N seconds."""
        # This would require storing historical data, simplified for now
        return 2.5

    def _get_system_uptime(self) -> str:
        """Get system uptime."""
        try:
            uptime_seconds = time.time() - psutil.boot_time()
            uptime_str = str(timedelta(seconds=int(uptime_seconds)))
            return uptime_str
        except:
            return "unknown"

    def _generate_health_recommendations(self, health_status: Dict, recent_metrics: Dict) -> List[str]:
        """Generate health recommendations based on current status."""
        recommendations = []

        if health_status.get('overall_health', 100) < 70:
            recommendations.append("Consider restarting the system to clear memory")
            recommendations.append("Monitor CPU usage and consider optimizing parallel processing")

        if recent_metrics.get('errors_last_hour', 0) > 10:
            recommendations.append("High error rate detected - check logs for issues")
            recommendations.append("Consider implementing circuit breakers for failing components")

        if recent_metrics.get('avg_scan_duration', 0) > 10:
            recommendations.append("Slow scan performance - consider reducing concurrent operations")
            recommendations.append("Optimize data fetching and signal processing")

        if not recommendations:
            recommendations.append("System is running optimally")

        return recommendations
