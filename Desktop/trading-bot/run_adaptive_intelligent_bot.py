#!/usr/bin/env python3
"""Adaptive Intelligent Trading Bot with ML Learning and Real-time Reversal Detection."""

import os
import sys
import asyncio
import logging
import signal
import traceback
import time
import warnings
from collections import deque
import numpy as np  # Added for RL state calculations
from datetime import datetime, timedelta, timezone  # Added timezone
from pathlib import Path
from typing import Dict, List, Optional, Any  # Added type hints
from dotenv import load_dotenv
import json
import requests

# Suppress gym deprecation warnings
warnings.filterwarnings("ignore", message=".*Gym.*unmaintained.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*gym.*deprecated.*", category=DeprecationWarning)

# Add src to path
sys.path.append('src')

from src.config.settings import settings
from src.data.market_data import MarketDataManager
from src.trading.broker_interface import BrokerManager
from src.analysis.technical import TechnicalAnalyzer, SignalDirection, safe_get_sentiment_value
from src.analysis.correlation import CorrelationAnalyzer

# Advanced components (lazy-loaded to prevent import errors)
try:
    from src.analysis.trend_reversal_detector import TrendReversalDetector
    REVERSAL_DETECTOR_AVAILABLE = True
except ImportError:
    REVERSAL_DETECTOR_AVAILABLE = False

try:
    from src.ml.trading_ml_engine import TradingMLEngine
    ML_ENGINE_AVAILABLE = True
except ImportError:
    ML_ENGINE_AVAILABLE = False

try:
    from src.ml.trade_analyzer import TradeAnalyzer
    TRADE_ANALYZER_AVAILABLE = True
except ImportError:
    TRADE_ANALYZER_AVAILABLE = False

try:
    from src.scheduling.intelligent_scheduler import IntelligentTradingScheduler
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False

try:
    from src.news.sentiment import SentimentAggregator
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

try:
    from src.monitoring.metrics import MetricsCollector
    METRICS_AVAILABLE = True
    print('[METRICS] MetricsCollector import succeeded')
except ImportError as e:
    METRICS_AVAILABLE = False
    print(f'[METRICS] MetricsCollector import failed: {e}')

try:
    from src.analysis.trade_attribution import TradeAttributionAnalyzer
    ATTRIBUTION_AVAILABLE = True
except ImportError:
    ATTRIBUTION_AVAILABLE = False

# Setup logging
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Create per-currency log directory
currency_logs_dir = logs_dir / "currencies"
currency_logs_dir.mkdir(exist_ok=True)

# Main bot logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / "adaptive_intelligent_bot.log", encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

# Per-currency loggers cache
currency_loggers = {}

def get_currency_logger(pair: str) -> logging.Logger:
    """Get or create a logger for a specific currency pair."""
    if pair not in currency_loggers:
        currency_logger = logging.getLogger(f"currency.{pair}")
        currency_logger.setLevel(logging.INFO)
        currency_logger.propagate = False  # Don't propagate to root logger

        # Create handler for this currency
        handler = logging.FileHandler(
            currency_logs_dir / f"{pair}.log",
            encoding='utf-8'
        )
        handler.setFormatter(logging.Formatter(
            f'%(asctime)s - {pair} - %(levelname)s - %(message)s'
        ))
        currency_logger.addHandler(handler)

        currency_loggers[pair] = currency_logger

    return currency_loggers[pair]

class AdaptiveIntelligentBot:
    """Advanced adaptive trading bot with ML learning and real-time reversal detection."""
    
    def __init__(self):
        logger.info("🤖 INITIALIZING ADAPTIVE INTELLIGENT BOT")
        print("[INIT] AdaptiveIntelligentBot initializing...")

        # Core components (always available)
        self.data_manager = MarketDataManager()
        self.broker_manager = BrokerManager()
        self.technical_analyzer = TechnicalAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer(self.data_manager)

        # Advanced components (instantiate if available)
        self.reversal_detector = TrendReversalDetector() if REVERSAL_DETECTOR_AVAILABLE else None
        self.ml_engine = TradingMLEngine() if ML_ENGINE_AVAILABLE else None
        self.trade_analyzer = TradeAnalyzer() if TRADE_ANALYZER_AVAILABLE else None
        self.scheduler = IntelligentTradingScheduler() if SCHEDULER_AVAILABLE else None
        self.sentiment_aggregator = SentimentAggregator() if SENTIMENT_AVAILABLE else None
        if METRICS_AVAILABLE:
            self.metrics_collector = MetricsCollector()
        else:
            self.metrics_collector = None
        self.attribution_analyzer = TradeAttributionAnalyzer() if ATTRIBUTION_AVAILABLE else None

        # Load adaptive parameters from file if exists, otherwise use defaults
        self.adaptive_params = self._load_adaptive_params()

        # Trading state
        self.running = True
        self.current_mode = 'TRADING'
        self.position_trackers = {}
        self.paper_mode = False  # Paper mode disabled, real trades will be placed

        # Capped buffers to prevent memory leaks
        self.daily_trade_data = deque(maxlen=100)   # Max 100 trades per day
        self.weekly_trade_data = deque(maxlen=500)  # Max 500 trades per week
        self.rl_state_buffer = deque(maxlen=1000)   # Max 1000 states for RL

        # Performance tracking
        self.heartbeat_file = Path("logs") / "adaptive_bot_heartbeat.json"
        self.scan_count = 0
        self.trades_executed = 0
        self.signals_analyzed = 0
        self.signals_rejected = 0
        self.reversals_detected = 0
        self.positions_closed_reversal = 0
        self.positions_closed_profit = 0
        self.positions_closed_loss = 0
        self.ml_analyses_performed = 0

        # 🚨 TRADE FREQUENCY PROTECTION 🚨
        self.last_trade_times = {}  # Track last trade time per pair
        self.pair_trade_counts = {}  # Track trades per pair in current session
        self.hourly_trade_count = 0  # Track trades in current hour
        self.hourly_reset_time = datetime.now()
        self.daily_trade_count = 0  # Track trades in current day
        self.daily_reset_time = datetime.now()

        # Trade frequency limits
        self.min_trade_interval_seconds = 300  # 5 minutes between trades on same pair
        self.max_trades_per_hour = 12  # Maximum 12 trades per hour
        self.max_trades_per_day = 96  # Maximum 96 trades per day
        self.max_trades_per_pair_per_hour = 2  # Maximum 2 trades per pair per hour

        # 🚀 INTELLIGENT OVERRIDE SYSTEM 🚀
        self.override_enabled = True  # Master switch for override system
        self.override_conditions = {
            'min_confidence_override': 0.95,      # 95%+ confidence for override
            'min_rr_ratio_override': 4.0,         # 4:1+ risk/reward for override
            'max_volatility_override': 0.001,     # Very low volatility for override
            'min_win_rate_recent': 0.80,          # 80%+ recent win rate for override
            'max_drawdown_override': 0.02,        # Max 2% drawdown for override
            'min_signal_strength': 0.90,          # Strong signal strength
            'session_optimization': True,         # Allow override in optimal sessions
        }

        # Override tracking
        self.override_trades_today = 0
        self.max_override_trades_per_day = 8  # Maximum 8 override trades per day
        self.override_cooldown_minutes = 15   # 15-minute cooldown after override
        self.last_override_time = None

        # Learning data
        self.last_daily_analysis = None
        self.last_weekly_analysis = None

        # 🕐 TIME CONSCIOUSNESS & MARKET SESSIONS 🕐
        self.market_sessions = {
            'tokyo': {'open': 0, 'close': 9, 'timezone': 'Asia/Tokyo'},      # 00:00 - 09:00 UTC
            'london': {'open': 8, 'close': 16, 'timezone': 'Europe/London'}, # 08:00 - 16:00 UTC
            'new_york': {'open': 13, 'close': 22, 'timezone': 'America/New_York'} # 13:00 - 22:00 UTC
        }

        # Trading schedule control
        self.new_orders_enabled = True  # Controls new order placement
        self.position_management_enabled = True  # Always enabled for existing positions
        self.current_session = 'unknown'
        self.last_session_check = None
        self.session_transition_time = None

        # 📊 COMPREHENSIVE ANALYSIS FRAMEWORK 📊
        self.daily_analysis_data = {
            'trades_executed': [],
            'trades_missed': [],
            'market_conditions': [],
            'performance_metrics': {},
            'optimal_parameters': {},
            'missed_opportunities': []
        }

        self.weekly_analysis_data = {
            'sentiment_correlations': {},
            'news_impacts': {},
            'session_performance': {},
            'parameter_evolution': {},
            'predictive_models': {}
        }

        # 🎯 SWEET SPOT FORMULA DEVELOPMENT 🎯
        self.sweet_spot_formula = {
            'min_conditions_required': 3,  # Minimum conditions to align for auto-execution
            'confidence_threshold': 0.85,  # Minimum confidence for sweet spot
            'conditions': {
                'technical_alignment': {'weight': 0.3, 'threshold': 0.8},
                'sentiment_positive': {'weight': 0.2, 'threshold': 0.7},
                'volatility_optimal': {'weight': 0.15, 'threshold': 0.002},
                'session_optimal': {'weight': 0.15, 'threshold': 0.8},
                'correlation_favorable': {'weight': 0.1, 'threshold': 0.6},
                'momentum_strong': {'weight': 0.1, 'threshold': 0.75}
            },
            'auto_execute_threshold': 0.9  # Score threshold for automatic execution
        }

        # 📈 ML TRAINING INTEGRATION 📈
        self.ml_training_schedule = {
            'daily_training': {'enabled': True, 'after_session': 'new_york'},
            'weekly_training': {'enabled': True, 'day': 'friday', 'after_session': 'new_york'},
            'model_update_frequency': 'daily'
        }

        logger.info("✅ Time consciousness and analysis framework initialized")
        print("[INIT] Time consciousness system ready.")

    async def check_session_and_permissions(self) -> bool:
        """Check current session and determine if new orders are allowed."""
        try:
            # Update session detection
            await self.detect_current_session()

            # Check if new orders are allowed based on session
            new_orders_allowed = await self.should_allow_new_orders()

            # Update trading control flags
            self.new_orders_enabled = new_orders_allowed

            # Always allow position management
            self.position_management_enabled = True

            logger.info(f"🕐 SESSION STATUS: {self.current_session}, New Orders: {'✅' if new_orders_allowed else '❌'}")

            return new_orders_allowed

        except Exception as e:
            logger.error(f"Error in session check: {e}")
            return False

    async def detect_current_session(self) -> str:
        """Detect current market session based on UTC time."""
        try:
            now = datetime.now(timezone.utc)
            current_hour = now.hour

            # Determine current session
            if 0 <= current_hour < 9:
                session = 'tokyo'
            elif 8 <= current_hour < 16:
                session = 'london'
            elif 13 <= current_hour < 22:
                session = 'new_york'
            else:
                session = 'overlap'  # Session overlap periods

            # Update session tracking
            if session != self.current_session:
                logger.info(f"🔄 SESSION CHANGE: {self.current_session} → {session}")
                self.current_session = session
                self.session_transition_time = now

            return session

        except Exception as e:
            logger.error(f"Error detecting current session: {e}")
            return 'unknown'

    async def should_allow_new_orders(self) -> bool:
        """Determine if new orders should be allowed based on session and time."""
        try:
            session = await self.detect_current_session()

            # Allow new orders during active trading sessions
            if session in ['tokyo', 'london', 'new_york']:
                # Additional check: Don't allow new orders in the last hour of NY session
                # to prepare for daily analysis
                now = datetime.now(timezone.utc)
                if session == 'new_york' and now.hour >= 21:  # Last hour of NY session
                    logger.info("🕐 FINAL HOUR: New orders paused - preparing for daily analysis")
                    return False

                return True

            # Don't allow new orders during session overlaps or unknown times
            logger.info(f"🕐 SESSION RESTRICTED: New orders not allowed during {session} session")
            return False

        except Exception as e:
            logger.error(f"Error checking new order permission: {e}")
            return False  # Default to restrictive

    async def perform_session_based_analysis(self):
        """Perform analysis based on current session."""
        try:
            session = await self.detect_current_session()
            now = datetime.now(timezone.utc)

            # Daily analysis after NY close (around 22:00 UTC)
            if session == 'overlap' and now.hour >= 22 and now.hour < 23:
                # Check if we haven't done daily analysis today
                if (self.last_daily_analysis is None or
                    datetime.fromisoformat(self.last_daily_analysis.get('timestamp', '2020-01-01')).date() < now.date()):

                    logger.info("🧠 TRIGGERING DAILY ANALYSIS: US market closed")
                    await self.perform_comprehensive_daily_analysis()

            # Weekend analysis (Friday after NY close)
            elif now.weekday() == 4 and session == 'overlap' and now.hour >= 22:  # Friday
                if (self.last_weekly_analysis is None or
                    datetime.fromisoformat(self.last_weekly_analysis.get('timestamp', '2020-01-01')).date() < now.date()):

                    logger.info("🧠 TRIGGERING WEEKEND ANALYSIS: Friday market closed")
                    await self.perform_comprehensive_weekend_analysis()

        except Exception as e:
            logger.error(f"Error in session-based analysis: {e}")

    async def perform_comprehensive_daily_analysis(self):
        """Perform comprehensive daily analysis after US market close."""
        try:
            logger.info("🧠 STARTING COMPREHENSIVE DAILY ANALYSIS")
            logger.info("="*60)

            # 1. TRADE PERFORMANCE ANALYSIS
            logger.info("📊 PHASE 1: TRADE PERFORMANCE ANALYSIS")

            executed_trades = list(self.daily_trade_data)
            total_trades = len(executed_trades)
            profitable_trades = sum(1 for trade in executed_trades if trade.get('profit', 0) > 0)
            losing_trades = total_trades - profitable_trades

            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            total_pnl = sum(trade.get('profit', 0) for trade in executed_trades)
            avg_win = sum(trade.get('profit', 0) for trade in executed_trades if trade.get('profit', 0) > 0) / profitable_trades if profitable_trades > 0 else 0
            avg_loss = sum(trade.get('profit', 0) for trade in executed_trades if trade.get('profit', 0) < 0) / losing_trades if losing_trades > 0 else 0

            logger.info(f"   Total Trades: {total_trades}")
            logger.info(f"   Win Rate: {win_rate:.1%}")
            logger.info(f"   Total P&L: ${total_pnl:.2f}")
            logger.info(f"   Average Win: ${avg_win:.2f}")
            logger.info(f"   Average Loss: ${avg_loss:.2f}")

            # 2. MISSED OPPORTUNITIES ANALYSIS
            logger.info("📊 PHASE 2: MISSED OPPORTUNITIES ANALYSIS")

            # Analyze signals that were rejected
            rejection_reasons = {}
            for trade in executed_trades:
                if 'rejection_reason' in trade:
                    reason = trade['rejection_reason']
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

            logger.info("   Top Rejection Reasons:")
            for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"      {reason}: {count} times")

            # 3. MARKET CONDITION ANALYSIS
            logger.info("📊 PHASE 3: MARKET CONDITION ANALYSIS")

            # Analyze volatility patterns
            volatility_levels = [trade.get('market_conditions', {}).get('volatility_15m', 0) for trade in executed_trades]
            avg_volatility = sum(volatility_levels) / len(volatility_levels) if volatility_levels else 0

            # Analyze session performance
            session_performance = {}
            for trade in executed_trades:
                session = trade.get('session', 'unknown')
                if session not in session_performance:
                    session_performance[session] = {'trades': 0, 'wins': 0, 'pnl': 0}
                session_performance[session]['trades'] += 1
                if trade.get('profit', 0) > 0:
                    session_performance[session]['wins'] += 1
                session_performance[session]['pnl'] += trade.get('profit', 0)

            logger.info("   Session Performance:")
            for session, stats in session_performance.items():
                win_rate_session = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
                logger.info(f"      {session}: {stats['wins']}/{stats['trades']} wins (${stats['pnl']:.2f})")

            # 4. PARAMETER OPTIMIZATION
            logger.info("📊 PHASE 4: PARAMETER OPTIMIZATION")

            # Analyze which parameters led to successful trades
            successful_params = []
            failed_params = []

            for trade in executed_trades:
                params = trade.get('adaptive_params_used', {})
                if trade.get('profit', 0) > 0:
                    successful_params.append(params)
                else:
                    failed_params.append(params)

            # Calculate optimal parameter ranges
            optimal_params = {}
            if successful_params:
                for param in ['min_confidence', 'min_rr_ratio', 'atr_multiplier_normal_vol']:
                    values = [p.get(param, 0) for p in successful_params if param in p]
                    if values:
                        optimal_params[param] = {
                            'avg': sum(values) / len(values),
                            'min': min(values),
                            'max': max(values)
                        }

            logger.info("   Optimal Parameters for Success:")
            for param, stats in optimal_params.items():
                logger.info(f"      {param}: {stats['avg']:.3f} (range: {stats['min']:.3f} - {stats['max']:.3f})")

            # 5. SWEET SPOT FORMULA DEVELOPMENT
            logger.info("📊 PHASE 5: SWEET SPOT FORMULA DEVELOPMENT")

            # Analyze conditions that led to successful trades
            sweet_spot_conditions = {
                'high_confidence': sum(1 for trade in executed_trades if trade.get('confidence', 0) >= 0.9 and trade.get('profit', 0) > 0),
                'good_rr_ratio': sum(1 for trade in executed_trades if trade.get('rr_ratio', 0) >= 3.0 and trade.get('profit', 0) > 0),
                'low_volatility': sum(1 for trade in executed_trades if trade.get('market_conditions', {}).get('volatility_15m', 1) <= 0.001 and trade.get('profit', 0) > 0),
                'optimal_session': sum(1 for trade in executed_trades if trade.get('session', '') in ['london', 'new_york'] and trade.get('profit', 0) > 0)
            }

            logger.info("   Successful Trade Conditions:")
            for condition, count in sweet_spot_conditions.items():
                success_rate = count / profitable_trades if profitable_trades > 0 else 0
                logger.info(f"      {condition}: {count} trades ({success_rate:.1%} of winners)")

            # Update sweet spot formula based on analysis
            if profitable_trades >= 5:  # Need minimum sample size
                # Adjust formula weights based on performance
                total_successful = sum(sweet_spot_conditions.values())
                if total_successful > 0:
                    self.sweet_spot_formula['conditions']['technical_alignment']['weight'] = sweet_spot_conditions['high_confidence'] / total_successful
                    self.sweet_spot_formula['conditions']['sentiment_positive']['weight'] = 0.1  # Keep low until sentiment analysis is better
                    self.sweet_spot_formula['conditions']['volatility_optimal']['weight'] = sweet_spot_conditions['low_volatility'] / total_successful
                    self.sweet_spot_formula['conditions']['session_optimal']['weight'] = sweet_spot_conditions['optimal_session'] / total_successful

                    logger.info("   ✅ SWEET SPOT FORMULA UPDATED based on daily performance")

            # 6. ML MODEL TRAINING
            logger.info("📊 PHASE 6: ML MODEL TRAINING")

            if ML_ENGINE_AVAILABLE and self.ml_engine:
                # Prepare training data
                training_data = {
                    'daily_performance': {
                        'win_rate': win_rate,
                        'total_pnl': total_pnl,
                        'avg_win': avg_win,
                        'avg_loss': avg_loss,
                        'total_trades': total_trades
                    },
                    'market_conditions': {
                        'avg_volatility': avg_volatility,
                        'session_performance': session_performance
                    },
                    'optimal_parameters': optimal_params,
                    'sweet_spot_conditions': sweet_spot_conditions
                }

                # Train ML model
                training_result = await self.ml_engine.train_daily_model(training_data)
                logger.info(f"   ML Training Result: {training_result}")

            # Store analysis results
            self.last_daily_analysis = {
                'timestamp': datetime.now().isoformat(),
                'performance_summary': {
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss
                },
                'optimal_parameters': optimal_params,
                'sweet_spot_formula': self.sweet_spot_formula.copy(),
                'session_performance': session_performance,
                'market_conditions': {
                    'avg_volatility': avg_volatility
                }
            }

            logger.info("✅ COMPREHENSIVE DAILY ANALYSIS COMPLETE")
            logger.info("="*60)

        except Exception as e:
            logger.error(f"Error in comprehensive daily analysis: {e}")

    async def perform_comprehensive_weekend_analysis(self):
        """Perform comprehensive weekend analysis covering the full week."""
        try:
            logger.info("🧠 STARTING COMPREHENSIVE WEEKEND ANALYSIS")
            logger.info("="*60)

            # 1. WEEKLY PERFORMANCE SUMMARY
            logger.info("📊 PHASE 1: WEEKLY PERFORMANCE SUMMARY")

            weekly_trades = list(self.weekly_trade_data)
            total_weekly_trades = len(weekly_trades)
            weekly_pnl = sum(trade.get('profit', 0) for trade in weekly_trades)

            logger.info(f"   Weekly Trades: {total_weekly_trades}")
            logger.info(f"   Weekly P&L: ${weekly_pnl:.2f}")

            # 2. SENTIMENT ANALYSIS CORRELATION
            logger.info("📊 PHASE 2: SENTIMENT ANALYSIS CORRELATION")

            if SENTIMENT_AVAILABLE and self.sentiment_aggregator:
                # Analyze how sentiment affected trade outcomes
                sentiment_impacts = {}
                pairs = settings.get_currency_pairs()

                for pair in pairs[:5]:  # Analyze top 5 pairs
                    try:
                        # Get historical sentiment data for the week
                        sentiment_history = await self.sentiment_aggregator.get_sentiment_history(pair, days=7)

                        # Correlate with trade performance for this pair
                        pair_trades = [trade for trade in weekly_trades if trade.get('symbol') == pair]
                        if pair_trades:
                            profitable_with_positive_sentiment = 0
                            profitable_with_negative_sentiment = 0

                            for trade in pair_trades:
                                sentiment_score = trade.get('sentiment_data', {}).get('overall_sentiment', 0)
                                if trade.get('profit', 0) > 0:
                                    if sentiment_score > 0.1:
                                        profitable_with_positive_sentiment += 1
                                    elif sentiment_score < -0.1:
                                        profitable_with_negative_sentiment += 1

                            total_profitable = profitable_with_positive_sentiment + profitable_with_negative_sentiment
                            if total_profitable > 0:
                                sentiment_impacts[pair] = {
                                    'positive_sentiment_win_rate': profitable_with_positive_sentiment / total_profitable,
                                    'negative_sentiment_win_rate': profitable_with_negative_sentiment / total_profitable,
                                    'total_profitable_trades': total_profitable
                                }

                    except Exception as e:
                        logger.warning(f"Error analyzing sentiment for {pair}: {e}")

                logger.info("   Sentiment Impact Analysis:")
                for pair, impact in sentiment_impacts.items():
                    logger.info(f"      {pair}:")
                    logger.info(f"         Positive sentiment win rate: {impact['positive_sentiment_win_rate']:.1%}")
                    logger.info(f"         Negative sentiment win rate: {impact['negative_sentiment_win_rate']:.1%}")

            # 3. NEWS IMPACT ANALYSIS
            logger.info("📊 PHASE 3: NEWS IMPACT ANALYSIS")

            if SCHEDULER_AVAILABLE and self.scheduler:
                # Analyze how news events affected price movements
                news_impacts = {}

                for pair in pairs[:5]:
                    try:
                        # Get news events for the week
                        weekly_news = await self.scheduler.get_weekly_news_events(pair)

                        # Analyze price movements around news events
                        for news_event in weekly_news:
                            event_time = datetime.fromisoformat(news_event['time'])
                            # Look for trades within 1 hour of news event
                            nearby_trades = [
                                trade for trade in weekly_trades
                                if trade.get('symbol') == pair and
                                abs((datetime.fromisoformat(trade['timestamp']) - event_time).total_seconds()) < 3600
                            ]

                            if nearby_trades:
                                avg_pnl_near_news = sum(trade.get('profit', 0) for trade in nearby_trades) / len(nearby_trades)
                                news_impacts[f"{pair}_{news_event['title'][:30]}"] = {
                                    'impact': news_event['impact'],
                                    'avg_pnl': avg_pnl_near_news,
                                    'trades_affected': len(nearby_trades)
                                }

                    except Exception as e:
                        logger.warning(f"Error analyzing news impact for {pair}: {e}")

                logger.info("   News Impact Analysis:")
                for news_key, impact in list(news_impacts.items())[:10]:  # Show top 10
                    logger.info(f"      {news_key}: {impact['impact']} impact, ${impact['avg_pnl']:.2f} avg P&L, {impact['trades_affected']} trades")

            # 4. PARAMETER EVOLUTION ANALYSIS
            logger.info("📊 PHASE 4: PARAMETER EVOLUTION ANALYSIS")

            # Analyze how parameters evolved throughout the week
            parameter_evolution = {}
            daily_analyses = [self.last_daily_analysis] if self.last_daily_analysis else []

            for analysis in daily_analyses:
                for param, stats in analysis.get('optimal_parameters', {}).items():
                    if param not in parameter_evolution:
                        parameter_evolution[param] = []
                    parameter_evolution[param].append(stats['avg'])

            logger.info("   Parameter Evolution:")
            for param, values in parameter_evolution.items():
                if values:
                    trend = "increasing" if values[-1] > values[0] else "decreasing"
                    change = ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0
                    logger.info(f"      {param}: {trend} by {change:.1f}% over week")

            # 5. PREDICTIVE MODEL DEVELOPMENT
            logger.info("📊 PHASE 5: PREDICTIVE MODEL DEVELOPMENT")

            if ML_ENGINE_AVAILABLE and self.ml_engine:
                # Develop predictive models for next week
                predictive_data = {
                    'weekly_performance': {
                        'total_trades': total_weekly_trades,
                        'weekly_pnl': weekly_pnl,
                        'sentiment_impacts': sentiment_impacts,
                        'news_impacts': news_impacts,
                        'parameter_evolution': parameter_evolution
                    },
                    'market_patterns': {
                        'session_performance': self.last_daily_analysis.get('session_performance', {}) if self.last_daily_analysis else {},
                        'volatility_patterns': {},
                        'correlation_changes': {}
                    }
                }

                # Train predictive models
                predictive_result = await self.ml_engine.train_predictive_models(predictive_data)
                logger.info(f"   Predictive Model Training Result: {predictive_result}")

            # Store weekend analysis results
            self.last_weekly_analysis = {
                'timestamp': datetime.now().isoformat(),
                'weekly_summary': {
                    'total_trades': total_weekly_trades,
                    'weekly_pnl': weekly_pnl,
                    'sentiment_impacts': sentiment_impacts,
                    'news_impacts': news_impacts,
                    'parameter_evolution': parameter_evolution
                },
                'predictive_models': predictive_result if 'predictive_result' in locals() else None
            }

            logger.info("✅ COMPREHENSIVE WEEKEND ANALYSIS COMPLETE")
            logger.info("="*60)

        except Exception as e:
            logger.error(f"Error in comprehensive weekend analysis: {e}")

        logger.info("✅ Adaptive intelligent bot initialized with capped buffers")
        print("[INIT] Initialization complete.")

    def _load_adaptive_params(self) -> Dict:
        """Load adaptive parameters from file or use defaults, with debug logging."""
        param_file = Path('data/adaptive_params.json')
        default_params = {
            'min_confidence': 0.75,  # Conservative default
            'min_rr_ratio': 2.5,     # Reasonable risk/reward
            'profit_protection_percentage': 0.25,
            'reversal_confidence_threshold': 0.75,
            'max_volatility': 0.002,
            'minimum_profit_to_protect': 20.0,
            'atr_multiplier_low_vol': 2.0,
            'atr_multiplier_normal_vol': 2.5,
            'atr_multiplier_high_vol': 3.0,
            'volatility_threshold_low': 0.001,
            'volatility_threshold_high': 0.003
        }

        if param_file.exists():
            try:
                with open(param_file, 'r') as f:
                    file_params = json.load(f)
                merged_params = {**default_params, **file_params}
                logger.info(f"📄 Loaded adaptive parameters from {param_file}: {merged_params}")
                print(f"[PARAMS] Loaded adaptive parameters: {merged_params}")
                return merged_params
            except Exception as e:
                logger.warning(f"Failed to load parameters from {param_file}: {e}")

        logger.info(f"📄 Using default adaptive parameters: {default_params}")
        print(f"[PARAMS] Using default adaptive parameters: {default_params}")
        return default_params
        
    def signal_handler(self, signum: int, frame: Optional[Any]) -> None:
        """Handle system signals gracefully."""
        logger.info(f"SIGNAL RECEIVED: {signum} - Shutting down gracefully...")
        self.running = False
        
    async def write_heartbeat(self):
        """Write comprehensive heartbeat with adaptive parameters."""
        try:
            # Safely get scheduler info
            schedule_info = "Scheduler not available"
            if self.scheduler:
                try:
                    schedule_info = str(self.scheduler.get_trading_schedule_info())
                except:
                    schedule_info = "Scheduler error"

            heartbeat_data = {
                'timestamp': datetime.now().isoformat(),
                'mode': self.current_mode,
                'scan_count': self.scan_count,
                'trades_executed': self.trades_executed,
                'signals_analyzed': self.signals_analyzed,
                'signals_rejected': self.signals_rejected,
                'reversals_detected': self.reversals_detected,
                'positions_closed_reversal': self.positions_closed_reversal,
                'positions_closed_profit': self.positions_closed_profit,
                'positions_closed_loss': self.positions_closed_loss,
                'ml_analyses_performed': self.ml_analyses_performed,
                'adaptive_parameters': self.adaptive_params,
                'running': self.running,
                'schedule_info': schedule_info,
                'components_available': {
                    'reversal_detector': self.reversal_detector is not None,
                    'ml_engine': self.ml_engine is not None,
                    'scheduler': self.scheduler is not None,
                    'sentiment_aggregator': self.sentiment_aggregator is not None,
                    'metrics_collector': self.metrics_collector is not None,
                    'attribution_analyzer': self.attribution_analyzer is not None
                }
            }

            with open(self.heartbeat_file, 'w') as f:
                json.dump(heartbeat_data, f, indent=2)

        except Exception as e:
            logger.error(f"Error writing heartbeat: {e}")
            
    async def mode_change_handler(self, old_mode: str, new_mode: str):
        """Handle mode changes from scheduler."""
        logger.info(f"🔄 MODE CHANGE: {old_mode} → {new_mode}")
        self.current_mode = new_mode
        
        if new_mode == 'DAILY_ANALYSIS':
            await self.perform_daily_ml_analysis()
        elif new_mode == 'WEEKEND_ANALYSIS':
            await self.perform_weekend_ml_analysis()
        elif new_mode == 'TRADING':
            logger.info("🚀 RESUMING TRADING MODE")
            
    async def real_time_reversal_monitoring(self):
        """Enhanced monitoring with trailing stops and Chandelier Exit integration."""
        try:
            # Skip if reversal detector not available
            if not REVERSAL_DETECTOR_AVAILABLE or self.reversal_detector is None:
                logger.debug("Reversal detector not available - skipping advanced monitoring")
                return

            positions = await self.broker_manager.get_positions()

            for position in positions:
                ticket = position.get('ticket')
                symbol = position.get('symbol')
                profit = position.get('profit', 0)
                position_type = position.get('type')  # 0=BUY, 1=SELL
                entry_price = position.get('price_open', 0)
                current_sl = position.get('sl', 0)

                # Get current market data for reversal analysis - ALL TIMEFRAMES
                try:
                    df_1m = await self.data_manager.get_candles(symbol, "M1", 100)
                    df_5m = await self.data_manager.get_candles(symbol, "M5", 100)
                    df_15m = await self.data_manager.get_candles(symbol, "M15", 50)
                    df_1h = await self.data_manager.get_candles(symbol, "H1", 50)
                    df_h4 = await self.data_manager.get_candles(symbol, "H4", 30)
                    if df_1m is not None:
                        df_1m = df_1m.reset_index()
                    if df_5m is not None:
                        df_5m = df_5m.reset_index()
                    if df_15m is not None:
                        df_15m = df_15m.reset_index()
                    if df_1h is not None:
                        df_1h = df_1h.reset_index()
                    if df_h4 is not None:
                        df_h4 = df_h4.reset_index()

                    if df_15m is None or df_1h is None:
                        continue

                    # Prepare comprehensive data for reversal detection
                    reversal_data = {
                        'df_1m': df_1m,
                        'df_5m': df_5m,
                        'df_15m': df_15m,
                        'df_1h': df_1h,
                        'df_h4': df_h4,
                        'current_position': 'LONG' if position_type == 0 else 'SHORT'
                    }

                    # ENHANCED TRAILING STOPS WITH CHANDELIER EXIT
                    await self._apply_trailing_stops(ticket, symbol, position_type, profit, entry_price, current_sl, df_15m)

                    # Detect trend reversal
                    reversal_result = self.reversal_detector.detect_trend_reversal(symbol, reversal_data)

                    # Check Chandelier Exit signals
                    chandelier_result = self.reversal_detector.detect_chandelier_exit_signal(
                        symbol, reversal_data, 'LONG' if position_type == 0 else 'SHORT'
                    )

                    # Combine reversal and Chandelier signals
                    should_exit = False
                    exit_reason = None
                    exit_confidence = 0.0

                    if reversal_result.get('reversal_detected', False):
                        self.reversals_detected += 1
                        should_exit = True
                        exit_reason = 'TREND_REVERSAL'
                        exit_confidence = reversal_result['confidence']

                        logger.info(f"🔄 TREND REVERSAL DETECTED: {symbol} #{ticket}")
                        logger.info(f"   Confidence: {reversal_result['confidence']:.2f}")
                        logger.info(f"   Direction: {reversal_result['direction']}")
                        logger.info(f"   Action: {reversal_result['immediate_action']}")

                    # Check Chandelier Exit (dynamic trailing stop)
                    if chandelier_result.get('chandelier_signal', 'HOLD').startswith('EXIT'):
                        if not should_exit or chandelier_result['confidence'] > exit_confidence:
                            should_exit = True
                            exit_reason = 'CHANDELIER_EXIT'
                            exit_confidence = chandelier_result['confidence']

                        logger.info(f"🚪 CHANDELIER EXIT TRIGGERED: {symbol} #{ticket}")
                        logger.info(f"   Confidence: {chandelier_result['confidence']:.2f}")
                        logger.info(f"   Exit Level: {chandelier_result['details']['exit_level_15m']:.5f}")
                        logger.info(f"   Distance: {chandelier_result['details']['distance_pips_15m']:.1f} pips")

                    # Execute exit if either signal triggered
                    if should_exit:
                        logger.info(f"🚨 EXECUTING {exit_reason}: {symbol} #{ticket}")

                        try:
                            close_result = await self.broker_manager.close_position(ticket)
                            if close_result:
                                self.positions_closed_reversal += 1

                                # Enhanced trade logging for ML learning
                                exit_data = {
                                    'timestamp': datetime.now().isoformat(),
                                    'ticket': ticket,
                                    'symbol': symbol,
                                    'exit_reason': exit_reason,
                                    'exit_confidence': exit_confidence,
                                    'profit_at_exit': profit,
                                    'position_type': 'LONG' if position_type == 0 else 'SHORT',
                                    'hold_duration': (datetime.now() - datetime.fromisoformat(self.position_trackers.get(ticket, {}).get('entry_time', datetime.now().isoformat()))).total_seconds() / 3600,
                                    'market_conditions': {
                                        'volatility_15m': df_15m['close'].pct_change().std() if len(df_15m) > 1 else 0,
                                        'volatility_1h': df_1h['close'].pct_change().std() if len(df_1h) > 1 else 0,
                                        'price_at_exit': df_15m['close'].iloc[-1] if len(df_15m) > 0 else 0
                                    }
                                }

                                # Add specific data based on exit reason
                                if exit_reason == 'TREND_REVERSAL':
                                    exit_data.update({
                                        'reversal_direction': reversal_result['direction'],
                                        'reversal_signals': reversal_result.get('signals', [])
                                    })
                                elif exit_reason == 'CHANDELIER_EXIT':
                                    exit_data.update({
                                        'chandelier_exit_level': chandelier_result['details']['exit_level_15m'],
                                        'distance_pips': chandelier_result['details']['distance_pips_15m'],
                                        'atr_value': chandelier_result['details']['atr_15m']
                                    })

                                self.daily_trade_data.append(exit_data)
                                self.weekly_trade_data.append(exit_data)

                                logger.info(f"✅ {exit_reason} SUCCESSFUL: {symbol} at ${profit:.2f}")

                                # Clean up position tracker
                                if ticket in self.position_trackers:
                                    del self.position_trackers[ticket]

                        except Exception as e:
                            logger.error(f"Error executing {exit_reason} for {ticket}: {e}")

                except Exception as e:
                    logger.error(f"Error in reversal monitoring for {symbol}: {e}")

        except Exception as e:
            logger.error(f"Error in real-time reversal monitoring: {e}")
    
    async def _apply_trailing_stops(self, ticket: int, symbol: str, position_type: int, 
                                  profit: float, entry_price: float, current_sl: float, df_15m) -> None:
        """Apply trailing stops using Chandelier Exit after position reaches minimum profit."""
        try:
            # Only apply trailing stops if position is profitable above minimum threshold
            if profit < self.adaptive_params['minimum_profit_to_protect']:
                return
            
            # Calculate Chandelier Exit levels
            chandelier_levels = self.technical_analyzer.calculate_chandelier_exit(df_15m, multiplier=3.0)
            
            if position_type == 0:  # LONG position
                new_sl = chandelier_levels['long_exit']
                
                # Only move stop loss up (never down for long positions)
                if new_sl > current_sl and new_sl < entry_price + (profit * 0.8):  # Don't trail too aggressively
                    try:
                        # Note: This would require implementing modify_position in broker_interface
                        # For now, we'll log the trailing stop action
                        logger.info(f"🔄 TRAILING STOP UPDATE: {symbol} #{ticket}")
                        logger.info(f"   Current SL: {current_sl:.5f} → New SL: {new_sl:.5f}")
                        logger.info(f"   Chandelier Exit Level: {new_sl:.5f}")
                        logger.info(f"   Profit Protection: ${profit:.2f}")
                        
                        # TODO: Implement broker_manager.modify_position(ticket, sl=new_sl)
                        # await self.broker_manager.modify_position(ticket, sl=new_sl)
                        
                    except Exception as e:
                        logger.error(f"Error updating trailing stop for {ticket}: {e}")
                        
            else:  # SHORT position
                new_sl = chandelier_levels['short_exit']
                
                # Only move stop loss down (never up for short positions)
                if new_sl < current_sl and new_sl > entry_price - (profit * 0.8):  # Don't trail too aggressively
                    try:
                        logger.info(f"🔄 TRAILING STOP UPDATE: {symbol} #{ticket}")
                        logger.info(f"   Current SL: {current_sl:.5f} → New SL: {new_sl:.5f}")
                        logger.info(f"   Chandelier Exit Level: {new_sl:.5f}")
                        logger.info(f"   Profit Protection: ${profit:.2f}")
                        
                        # TODO: Implement broker_manager.modify_position(ticket, sl=new_sl)
                        # await self.broker_manager.modify_position(ticket, sl=new_sl)
                        
                    except Exception as e:
                        logger.error(f"Error updating trailing stop for {ticket}: {e}")
                        
        except Exception as e:
            logger.error(f"Error in trailing stops for {ticket}: {e}")
            
    async def adaptive_percentage_protection(self):
        """Enhanced percentage-based protection with adaptive parameters."""
        try:
            positions = await self.broker_manager.get_positions()
            
            for position in positions:
                ticket = position.get('ticket')
                profit = position.get('profit', 0)
                symbol = position.get('symbol', '')
                
                # Initialize or update position tracker
                if ticket not in self.position_trackers:
                    self.position_trackers[ticket] = {
                        'peak_profit': max(0, profit),
                        'symbol': symbol,
                        'entry_time': datetime.now().isoformat()
                    }
                else:
                    if profit > self.position_trackers[ticket]['peak_profit']:
                        self.position_trackers[ticket]['peak_profit'] = profit
                        
                tracker = self.position_trackers[ticket]
                peak_profit = tracker['peak_profit']
                
                # Adaptive loss protection
                if profit <= -self.adaptive_params['minimum_profit_to_protect']:
                    try:
                        close_result = await self.broker_manager.close_position(ticket)
                        if close_result:
                            self.positions_closed_loss += 1
                            logger.info(f"🔴 ADAPTIVE LOSS PROTECTION: {symbol} closed at ${profit:.2f}")
                            del self.position_trackers[ticket]
                    except Exception as e:
                        logger.error(f"Error in adaptive loss protection {ticket}: {e}")
                        
                # Adaptive profit protection
                elif peak_profit >= self.adaptive_params['minimum_profit_to_protect']:
                    drawdown_amount = peak_profit - profit
                    drawdown_percentage = (drawdown_amount / peak_profit) if peak_profit > 0 else 0
                    
                    if drawdown_percentage >= self.adaptive_params['profit_protection_percentage']:
                        try:
                            close_result = await self.broker_manager.close_position(ticket)
                            if close_result:
                                self.positions_closed_profit += 1
                                
                                # Enhanced trade logging for ML learning
                                protection_data = {
                                    'timestamp': datetime.now().isoformat(),
                                    'ticket': ticket,
                                    'symbol': symbol,
                                    'exit_reason': 'ADAPTIVE_PROFIT_PROTECTION',
                                    'peak_profit': peak_profit,
                                    'exit_profit': profit,
                                    'drawdown_percentage': drawdown_percentage,
                                    'protection_threshold': self.adaptive_params['profit_protection_percentage'],
                                    'hold_duration': (datetime.now() - datetime.fromisoformat(tracker.get('entry_time', datetime.now().isoformat()))).total_seconds() / 3600,
                                    'market_conditions': {
                                        'current_time': datetime.now().isoformat(),
                                        'protection_trigger': 'PROFIT_DRAWDOWN'
                                    }
                                }
                                
                                self.daily_trade_data.append(protection_data)
                                self.weekly_trade_data.append(protection_data)
                                
                                logger.info(f"🛡️ ADAPTIVE PROFIT PROTECTION: {symbol} closed at ${profit:.2f}")
                                logger.info(f"   Peak: ${peak_profit:.2f}, Drawdown: {drawdown_percentage:.1%}")
                                
                                del self.position_trackers[ticket]
                        except Exception as e:
                            logger.error(f"Error in adaptive profit protection {ticket}: {e}")
                            
        except Exception as e:
            logger.error(f"Error in adaptive percentage protection: {e}")
            
    async def get_dynamic_atr_multiplier(self, pair: str, df_recent) -> float:
        """Calculate dynamic ATR multiplier based on current market conditions."""
        try:
            if df_recent is None or len(df_recent) < 10:
                return self.adaptive_params['atr_multiplier_normal_vol']
            
            # Calculate current volatility
            current_volatility = df_recent['close'].pct_change().std()
            
            # Get market session info for additional context
            current_hour = datetime.now().hour
            
            # Session-based adjustments
            session_multiplier = 1.0
            if 8 <= current_hour <= 10:  # London open
                session_multiplier = 1.2
            elif 13 <= current_hour <= 15:  # NY open
                session_multiplier = 1.3
            elif 20 <= current_hour <= 22:  # Asian session
                session_multiplier = 0.9
            
            # Volatility-based ATR multiplier
            if current_volatility < self.adaptive_params['volatility_threshold_low']:
                base_multiplier = self.adaptive_params['atr_multiplier_low_vol']
                vol_condition = "LOW_VOLATILITY"
            elif current_volatility > self.adaptive_params['volatility_threshold_high']:
                base_multiplier = self.adaptive_params['atr_multiplier_high_vol']
                vol_condition = "HIGH_VOLATILITY"
            else:
                base_multiplier = self.adaptive_params['atr_multiplier_normal_vol']
                vol_condition = "NORMAL_VOLATILITY"
            
            # Apply session adjustment
            final_multiplier = base_multiplier * session_multiplier
            
            logger.debug(f"   {pair}: Dynamic ATR multiplier = {final_multiplier:.2f} ({vol_condition}, session={session_multiplier:.1f})")
            
            return final_multiplier
            
        except Exception as e:
            logger.error(f"Error calculating dynamic ATR multiplier for {pair}: {e}")
            return self.adaptive_params['atr_multiplier_normal_vol']

    async def check_news_events(self, pair: str) -> bool:
        """Check for high-impact news events that could affect the pair."""
        try:
            # Skip if scheduler not available
            if not SCHEDULER_AVAILABLE or self.scheduler is None:
                logger.debug("Scheduler not available - skipping news check")
                return True

            events = await self.scheduler.get_upcoming_events(pair)
            now = datetime.now(timezone.utc)

            for event in events:
                if (event['impact'].lower() == 'high' and
                    abs((datetime.fromisoformat(event['time']) - now).total_seconds()) < 3600):
                    logger.warning(f"🚨 HIGH-IMPACT NEWS for {pair}: {event['title']}")
                    return False
            return True
        except Exception as e:
            logger.error(f"News check failed: {e}")
            return True

    def get_atr_multiplier(self, volatility: float, pair: str) -> float:
        """Get ATR multiplier based on volatility and pair."""
        if pair.lower() == 'xauusd':
            return self.adaptive_params['atr_multiplier_high_vol']
        if volatility < self.adaptive_params['volatility_threshold_low']:
            return self.adaptive_params['atr_multiplier_low_vol']
        elif volatility > self.adaptive_params['volatility_threshold_high']:
            return self.adaptive_params['atr_multiplier_high_vol']
        return self.adaptive_params['atr_multiplier_normal_vol']

    def get_volatility_level(self, volatility: float, pair: str) -> str:
        """Get volatility level classification."""
        if pair.lower() == 'xauusd':
            return 'HIGH'
        if volatility < self.adaptive_params['volatility_threshold_low']:
            return 'LOW'
        elif volatility > self.adaptive_params['volatility_threshold_high']:
            return 'HIGH'
        return 'NORMAL'
            
    async def adaptive_signal_analysis(self, pair: str, df_15m, df_1h, df_h4=None):
        """Enhanced signal analysis with adaptive parameters and multi-timeframe confirmation."""
        try:
            # PHASE 1 DIAGNOSTICS: Log signal generation start
            logger.info(f"🔍 PHASE 1 DIAGNOSTICS - Signal Analysis for {pair}")
            logger.info(f"   Data quality: 15M={len(df_15m) if df_15m is not None else 0}, 1H={len(df_1h) if df_1h is not None else 0}, H4={len(df_h4) if df_h4 is not None else 0}")

            # Generate base signal with dynamic stops
            signal = self.technical_analyzer.generate_signal(
                df_15m, df_1h,
                adaptive_params=self.adaptive_params,
                pair=pair,
                correlation_analyzer=self.correlation_analyzer,
                economic_calendar_filter=None  # Could be added later
            )

            logger.info(f"   Base signal: direction={signal['direction'].value if signal else 'NONE'}, confidence={signal.get('confidence', 0):.3f}")

            if signal['direction'] == SignalDirection.NONE:
                logger.info(f"   ❌ REJECTION: No signal generated by technical analyzer")
                return None

            # MULTI-TIMEFRAME CONFIRMATION - Check H4 alignment
            h4_confirmation = True
            h4_details = "No H4 data"

            if df_h4 is not None and len(df_h4) >= 20:
                try:
                    # Generate H4 signal for confirmation
                    h4_signal = self.technical_analyzer.generate_signal(df_h4, df_h4)  # Use H4 for both timeframes

                    logger.info(f"   H4 signal: direction={h4_signal['direction'].value if h4_signal else 'NONE'}, confidence={h4_signal.get('confidence', 0):.3f}")

                    if h4_signal['direction'] != SignalDirection.NONE:
                        # Check if H4 direction aligns with our signal
                        if h4_signal['direction'] == signal['direction']:
                            h4_confirmation = True
                            h4_details = f"H4 ALIGNED ({h4_signal['confidence']:.1%})"
                            # Boost confidence for aligned signals
                            signal['confidence'] = min(signal['confidence'] * 1.1, 0.95)
                        else:
                            h4_confirmation = False
                            h4_details = f"H4 CONFLICT ({h4_signal['direction'].value} vs {signal['direction'].value})"
                    else:
                        # H4 is neutral - check trend direction
                        h4_sma_20 = df_h4['close'].rolling(20).mean()
                        h4_sma_50 = df_h4['close'].rolling(50).mean()

                        if len(h4_sma_20) >= 2 and len(h4_sma_50) >= 2:
                            h4_trend_up = h4_sma_20.iloc[-1] > h4_sma_50.iloc[-1]

                            if (signal['direction'] == SignalDirection.BUY and h4_trend_up) or \
                               (signal['direction'] == SignalDirection.SELL and not h4_trend_up):
                                h4_confirmation = True
                                h4_details = f"H4 TREND ALIGNED ({'UP' if h4_trend_up else 'DOWN'})"
                            else:
                                h4_confirmation = False
                                h4_details = f"H4 TREND CONFLICT ({'UP' if h4_trend_up else 'DOWN'} vs {signal['direction'].value})"
                        else:
                            h4_details = "H4 insufficient data for trend"

                except Exception as e:
                    logger.error(f"Error in H4 confirmation for {pair}: {e}")
                    h4_details = "H4 analysis error"

            logger.info(f"   H4 Confirmation: {h4_confirmation} - {h4_details}")
            logger.info(f"   Signal confidence after H4: {signal.get('confidence', 0):.3f}")

            # Reject signals without H4 confirmation (unless very high confidence)
            if not h4_confirmation and signal['confidence'] < 0.90:
                self.signals_rejected += 1
                logger.info(f"   ❌ REJECTION: H4 timeframe conflict (confidence {signal.get('confidence', 0):.3f} < 0.90)")
                return None

            df_recent = await self.data_manager.get_candles(pair, "M15", 20)
            if df_recent is not None:
                df_recent = df_recent.reset_index()
            if df_recent is not None and len(df_recent) >= 10:
                recent_volatility = df_recent['close'].pct_change().std()

                # Adaptive confidence based on market conditions
                if recent_volatility > 0.003:  # High volatility
                    adaptive_threshold = 0.80  # RAISED from 0.75 for high volatility
                    condition = "HIGH_VOLATILITY"
                elif recent_volatility > 0.0015:  # Normal volatility
                    adaptive_threshold = self.adaptive_params['min_confidence']  # 0.85
                    condition = "NORMAL_VOLATILITY"
                else:  # Low volatility
                    adaptive_threshold = 0.75  # RAISED from 0.70 for low volatility
                    condition = "LOW_VOLATILITY"
            else:
                adaptive_threshold = self.adaptive_params['min_confidence']
                condition = "DEFAULT"

            logger.info(f"   Volatility: {recent_volatility:.6f}, Adaptive threshold: {adaptive_threshold:.3f} ({condition})")
            logger.info(f"   ✅ SIGNAL APPROVED: {pair} {signal['direction'].value} (confidence: {signal.get('confidence', 0):.3f})")

            return signal

        except Exception as e:
            logger.error(f"Error in adaptive signal analysis for {pair}: {e}")
            return None
            
    async def perform_daily_ml_analysis(self):
        """Perform daily ML analysis during US-Japan gap."""
        try:
            # Skip if ML engine not available
            if not ML_ENGINE_AVAILABLE or self.ml_engine is None:
                logger.debug("ML engine not available - skipping daily analysis")
                return

            logger.info("🧠 STARTING DAILY ML ANALYSIS...")

            # Prepare daily data
            daily_data = {
                'session_date': datetime.now().date().isoformat(),
                'executed_trades': list(self.daily_trade_data),  # Convert deque to list
                'signals_analyzed': self.signals_analyzed,
                'signals_rejected': self.signals_rejected,
                'reversals_detected': self.reversals_detected,
                'adaptive_parameters_used': self.adaptive_params.copy()
            }

            # Perform ML analysis
            analysis_result = self.ml_engine.perform_daily_analysis(daily_data)
            self.last_daily_analysis = analysis_result
            self.ml_analyses_performed += 1

            # ENHANCED ML FEEDBACK LOOP - Apply recommended parameter adjustments
            adjustments = analysis_result.get('strategy_adjustments', [])
            insights = analysis_result.get('ml_insights', {})

            # Apply ML-driven parameter updates
            for adjustment in adjustments:
                param = adjustment.get('parameter')
                new_value = adjustment.get('recommended_value')

                if param in self.adaptive_params:
                    old_value = self.adaptive_params[param]
                    self.adaptive_params[param] = new_value

                    logger.info(f"🔧 ADAPTIVE PARAMETER UPDATE:")
                    logger.info(f"   {param}: {old_value} → {new_value}")
                    logger.info(f"   Reason: {adjustment.get('reason')}")

            # ADDITIONAL ML-DRIVEN ADJUSTMENTS based on insights
            win_rate = insights.get('win_rate', 0.5)
            profit_factor = insights.get('profit_factor', 1.0)
            confidence_correlation = insights.get('confidence_correlation', 0)

            # Adjust ATR multipliers based on win rate performance
            if win_rate < 0.6:
                # Poor win rate - widen stops for safety
                self.adaptive_params['atr_multiplier_normal_vol'] = min(
                    self.adaptive_params['atr_multiplier_normal_vol'] + 0.5, 4.0
                )
                logger.info(f"🔧 ML ADJUSTMENT: Widening ATR multiplier due to low win rate ({win_rate:.1%})")
            elif win_rate > 0.75:
                # High win rate - can tighten stops slightly
                self.adaptive_params['atr_multiplier_normal_vol'] = max(
                    self.adaptive_params['atr_multiplier_normal_vol'] - 0.2, 2.0
                )
                logger.info(f"🔧 ML ADJUSTMENT: Tightening ATR multiplier due to high win rate ({win_rate:.1%})")

            # Adjust confidence threshold based on correlation with performance
            if confidence_correlation < 0.2:
                # Weak correlation - increase confidence requirement
                self.adaptive_params['min_confidence'] = min(
                    self.adaptive_params['min_confidence'] + 0.05, 0.95
                )
                logger.info(f"🔧 ML ADJUSTMENT: Raising confidence threshold due to weak correlation ({confidence_correlation:.2f})")
            elif confidence_correlation > 0.6:
                # Strong correlation - can slightly lower threshold
                self.adaptive_params['min_confidence'] = max(
                    self.adaptive_params['min_confidence'] - 0.02, 0.75
                )
                logger.info(f"🔧 ML ADJUSTMENT: Lowering confidence threshold due to strong correlation ({confidence_correlation:.2f})")

            # Adjust profit protection based on profit factor
            if profit_factor < 1.2:
                # Poor profit factor - tighten protection
                self.adaptive_params['profit_protection_percentage'] = max(
                    self.adaptive_params['profit_protection_percentage'] - 0.05, 0.15
                )
                logger.info(f"🔧 ML ADJUSTMENT: Tightening profit protection due to low profit factor ({profit_factor:.2f})")
            elif profit_factor > 2.0:
                # Excellent profit factor - can relax protection slightly
                self.adaptive_params['profit_protection_percentage'] = min(
                    self.adaptive_params['profit_protection_percentage'] + 0.02, 0.25
                )
                logger.info(f"🔧 ML ADJUSTMENT: Relaxing profit protection due to high profit factor ({profit_factor:.2f})")

            # Reset daily data
            self.daily_trade_data.clear()

            logger.info("✅ DAILY ML ANALYSIS COMPLETE")

        except Exception as e:
            logger.error(f"Error in daily ML analysis: {e}")
            
    async def perform_weekend_ml_analysis(self):
        """Perform comprehensive weekend ML analysis."""
        try:
            logger.info("🧠 STARTING WEEKEND ML ANALYSIS...")
            
            # Prepare weekly data
            weekly_data = {
                'week_period': f"{datetime.now().date() - timedelta(days=7)} to {datetime.now().date()}",
                'weekly_trades': self.weekly_trade_data,
                'daily_analyses': [self.last_daily_analysis] if self.last_daily_analysis else [],
                'adaptive_parameter_evolution': self.adaptive_params.copy()
            }
            
            # Perform comprehensive weekly analysis
            analysis_result = self.ml_engine.perform_weekly_analysis(weekly_data)
            self.last_weekly_analysis = analysis_result
            self.ml_analyses_performed += 1
            
            # Apply major strategy optimizations
            optimizations = analysis_result.get('next_week_strategy', {})
            optimized_params = optimizations.get('optimized_parameters', {})
            
            for param, value in optimized_params.items():
                if param in self.adaptive_params:
                    old_value = self.adaptive_params[param]
                    self.adaptive_params[param] = value
                    
                    logger.info(f"🚀 WEEKLY OPTIMIZATION:")
                    logger.info(f"   {param}: {old_value} → {value}")
                    
            # Reset weekly data
            self.weekly_trade_data = []
            
            logger.info("✅ WEEKEND ML ANALYSIS COMPLETE")
            
        except Exception as e:
            logger.error(f"Error in weekend ML analysis: {e}")

    async def generate_attribution_report(self, start_date: Optional[str] = None,
                                        end_date: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive trade attribution report."""
        try:
            # Skip if attribution analyzer not available
            if not ATTRIBUTION_AVAILABLE or self.attribution_analyzer is None:
                logger.debug("Attribution analyzer not available - skipping report generation")
                return {'error': 'Attribution analyzer not available'}

            logger.info("📊 GENERATING TRADE ATTRIBUTION REPORT...")

            # Generate attribution analysis
            report = self.attribution_analyzer.generate_attribution_report(start_date, end_date)

            # Add additional insights
            if self.attribution_analyzer.trade_history:
                report['insights'] = {
                    'top_strategies': self.attribution_analyzer.get_top_performing_strategies(5),
                    'worst_strategies': self.attribution_analyzer.get_worst_performing_strategies(5),
                    'seasonal_performance': self.attribution_analyzer.analyze_seasonal_performance(),
                    'confidence_intervals': self.attribution_analyzer.calculate_attribution_confidence()
                }

                # Add strategy recommendations
                top_strategies = report['insights']['top_strategies']
                if top_strategies:
                    best_strategy = top_strategies[0][0]
                    report['recommendations'] = {
                        'best_performing_strategy': best_strategy,
                        'suggested_focus': f"Increase allocation to {best_strategy} strategy",
                        'risk_adjusted_insights': "Consider strategies with Sharpe ratios > 1.0"
                    }

            # Save report to file
            report_file = logs_dir / f"attribution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"✅ ATTRIBUTION REPORT GENERATED: {report_file}")
            return report

        except Exception as e:
            logger.error(f"Error generating attribution report: {e}")
            return {'error': str(e)}
            
    async def get_rl_parameter_adjustments(self) -> Dict:
        """Get parameter adjustments from trained RL model."""
        try:
            # Skip if ML engine not available
            if not ML_ENGINE_AVAILABLE or self.ml_engine is None:
                logger.debug("ML engine not available - skipping RL adjustments")
                return {}

            # Get current market state for RL
            current_state = await self._get_current_market_state()
            if current_state is None:
                return {}

            # Get RL action
            rl_action = self.ml_engine.get_rl_action(current_state)
            if rl_action is None:
                return {}

            # Convert RL action to parameter adjustments
            adjustments = {
                'min_confidence': self.adaptive_params['min_confidence'] + rl_action[0] * 0.05,  # ±5% adjustment
                'min_rr_ratio': self.adaptive_params['min_rr_ratio'] + rl_action[1] * 0.5,      # ±0.5 adjustment
                'atr_multiplier_normal_vol': self.adaptive_params['atr_multiplier_normal_vol'] + rl_action[2] * 0.2  # ±0.2 adjustment
            }

            # Clamp adjustments to reasonable ranges
            adjustments['min_confidence'] = max(0.75, min(0.95, adjustments['min_confidence']))
            adjustments['min_rr_ratio'] = max(2.0, min(4.5, adjustments['min_rr_ratio']))
            adjustments['atr_multiplier_normal_vol'] = max(2.0, min(4.0, adjustments['atr_multiplier_normal_vol']))

            logger.info("🎯 RL Parameter Adjustments Applied:")
            for param, value in adjustments.items():
                logger.info(f"   {param}: {self.adaptive_params[param]:.3f} → {value:.3f}")

            return adjustments

        except Exception as e:
            logger.error(f"Error getting RL adjustments: {e}")
            return {}

    async def _get_current_market_state(self) -> Optional[np.ndarray]:
        """Get current market state for RL model."""
        try:
            # Get average market conditions across all pairs
            pairs = settings.get_currency_pairs()
            rsi_values = []
            volatility_values = []
            correlation_values = []
            sentiment_values = []

            for pair in pairs[:3]:  # Use top 3 pairs for state calculation
                try:
                    df = await self.data_manager.get_candles(pair, "M15", 20)
                    if df is not None:
                        df = df.reset_index()
                    if df is not None and len(df) >= 10:
                        # Calculate RSI (simplified)
                        rsi = 50  # Default
                        if 'close' in df.columns:
                            price_changes = df['close'].pct_change()
                            gains = price_changes.where(price_changes > 0, 0).rolling(14).mean()
                            losses = (-price_changes.where(price_changes < 0, 0)).rolling(14).mean()
                            if len(gains) > 0 and len(losses) > 0 and losses.iloc[-1] != 0:
                                rs = gains.iloc[-1] / losses.iloc[-1]
                                rsi = 100 - (100 / (1 + rs))
                        rsi_values.append(rsi / 100)  # Normalize to 0-1

                        # Calculate volatility
                        if len(df) >= 5:
                            vol = df['close'].pct_change().std()
                            volatility_values.append(min(vol * 100, 0.02))  # Cap at 2%

                except Exception as e:
                    logger.debug(f"Error getting state for {pair}: {e}")
                    continue

            # Calculate average correlation (simplified)
            if len(pairs) >= 2:
                try:
                    await self.correlation_analyzer.update_correlation_matrix(pairs)
                    corr_matrix = self.correlation_analyzer.correlation_matrix
                    avg_corr = corr_matrix.mean().mean() if hasattr(corr_matrix, 'mean') else 0.0
                    correlation_values.append(avg_corr)
                except:
                    correlation_values.append(0.0)

            # Default sentiment (can be enhanced later)
            sentiment_values.append(0.0)

            # Aggregate to single state vector
            if rsi_values and volatility_values:
                state = np.array([
                    np.mean(rsi_values),           # Average RSI (0-1)
                    np.mean(volatility_values),     # Average volatility
                    self.adaptive_params['min_confidence'] / 0.95,  # Normalized confidence
                    np.mean(correlation_values) if correlation_values else 0.0,  # Average correlation
                    np.mean(sentiment_values)       # Average sentiment
                ])

                # Normalize to [-2, 2] range as expected by RL model
                state = np.clip(state * 4 - 2, -2, 2)
                return state

            return None

        except Exception as e:
            logger.error(f"Error calculating market state: {e}")
            return None

    async def adaptive_trading_scan(self):
        """Enhanced trading scan with adaptive parameters and real-time monitoring."""
        scan_start_time = time.time()
        try:
            # Check session and permissions first
            new_orders_allowed = await self.check_session_and_permissions()

            # Perform session-based analysis if needed
            await self.perform_session_based_analysis()

            logger.info(f"🤖 ADAPTIVE SCAN #{self.scan_count + 1} - New Orders: {'✅' if new_orders_allowed else '❌'}")

            # Perform health check only if metrics_collector is available
            if self.metrics_collector is not None:
                health_status = await self.metrics_collector.perform_health_check()
                if health_status.get('overall_health', 100) < 50:
                    logger.warning(f"⚠️ SYSTEM HEALTH WARNING: {health_status.get('overall_health', 0)}%")
                    for alert in health_status.get('alerts', []):
                        logger.warning(f"   {alert['type']}: {alert['message']}")
            else:
                logger.info("⚠️ Metrics collector not available - skipping health check")

            # Get RL-based parameter adjustments
            rl_adjustments = await self.get_rl_parameter_adjustments()
            if rl_adjustments:
                # Apply RL adjustments to adaptive parameters
                for param, value in rl_adjustments.items():
                    self.adaptive_params[param] = value

            # Real-time reversal monitoring
            await self.real_time_reversal_monitoring()

            # Adaptive protection
            await self.adaptive_percentage_protection()

            # PARALLEL PROCESSING - Process multiple currency pairs concurrently
            pairs = settings.get_currency_pairs()
            # PAPER_MODE: restrict to one pair for debug
            if self.paper_mode:
                pairs = pairs[:1]
                logger.info(f"[PAPER_MODE] Restricting to single pair: {pairs[0]}")
            logger.info(f"🔄 Processing {len(pairs)} currency pairs with parallel processing")

            # Create parallel tasks for data fetching
            async def fetch_pair_data(pair: str) -> Dict:
                """Fetch all required data for a currency pair in parallel."""
                try:
                    # Create currency-specific logger at the beginning of data fetching
                    currency_logger = get_currency_logger(pair)
                    
                    # Determine if this is a high-stakes event (e.g., major news)
                    is_high_stakes_event = False
                    if SCHEDULER_AVAILABLE and self.scheduler and hasattr(self.scheduler, "is_high_impact_news"):
                        try:
                            is_high_stakes_event = await self.scheduler.is_high_impact_news(pair)
                        except Exception as e:
                            logger.warning(f"Scheduler high-stakes check failed for {pair}: {e}")
                            is_high_stakes_event = False

                    # Parallel data fetching
                    df_15m_task = asyncio.create_task(self.data_manager.get_candles(pair, "M15", 100))
                    df_1h_task = asyncio.create_task(self.data_manager.get_candles(pair, "H1", 100))
                    df_h4_task = asyncio.create_task(self.data_manager.get_candles(pair, "H4", 50))
                    spread_task = asyncio.create_task(self.broker_manager.get_spread_pips(pair))
                    async def empty_sentiment():
                        return {}
                    sentiment_task = (
                        self.sentiment_aggregator.get_overall_sentiment(pair, high_stakes=is_high_stakes_event)
                        if SENTIMENT_AVAILABLE and self.sentiment_aggregator
                        else empty_sentiment()
                    )

                    # Execute all tasks concurrently
                    df_15m, df_1h, df_h4, spread_pips, sentiment_data = await asyncio.gather(
                        df_15m_task, df_1h_task, df_h4_task, spread_task, sentiment_task,
                        return_exceptions=True
                    )
                    # Always force sentiment_data to dict
                    if isinstance(sentiment_data, Exception) or sentiment_data is None:
                        sentiment_data = {}
                    elif not isinstance(sentiment_data, dict):
                        logger.warning(f"[SENTIMENT] Unexpected sentiment_data type {type(sentiment_data)} for {pair}, forcing to empty dict.")
                        sentiment_data = {}

                    # Handle exceptions
                    if isinstance(df_15m, Exception) or df_15m is None:
                        currency_logger.error(f"Failed to fetch 15M data for {pair}")
                        logger.warning(f"Failed to fetch 15M data for {pair}")
                        return None
                    if isinstance(df_1h, Exception) or df_1h is None:
                        currency_logger.error(f"Failed to fetch 1H data for {pair}")
                        logger.warning(f"Failed to fetch 1H data for {pair}")
                        return None
                    if df_15m is not None:
                        df_15m = df_15m.reset_index()
                    if df_1h is not None:
                        df_1h = df_1h.reset_index()
                    if df_h4 is not None and not isinstance(df_h4, Exception):
                                               df_h4 = df_h4.reset_index()
                    return {
                        'pair': pair,
                        'df_15m': df_15m,
                        'df_1h': df_1h,
                        'df_h4': df_h4 if not isinstance(df_h4, Exception) else None,
                        'spread_pips': spread_pips if not isinstance(spread_pips, Exception) else None,
                        'sentiment_data': sentiment_data
                    }

                except Exception as e:
                    logger.error(f"Error fetching data for {pair}: {e}")
                    return None

            # Fetch data for all pairs in parallel
            data_tasks = [fetch_pair_data(pair) for pair in pairs]
            pair_data_results = await asyncio.gather(*data_tasks, return_exceptions=True)

            # Filter successful results
            valid_pair_data = [data for data in pair_data_results if data is not None and not isinstance(data, Exception)]

            logger.info(f"✅ Successfully fetched data for {len(valid_pair_data)}/{len(pairs)} pairs")

            # Get shared data once (not per pair)
            positions = await self.broker_manager.get_positions()
            account_info = await self.broker_manager.get_account_info()
            mpt_weights = self.correlation_analyzer.get_mpt_weights(None)
            portfolio_metrics = self.correlation_analyzer.calculate_portfolio_metrics(positions)
            await self.correlation_analyzer.update_correlation_matrix(pairs)

            # Update monitoring metrics only if metrics_collector is available
            if self.metrics_collector is not None:
                self.metrics_collector.update_account_metrics(account_info)
                self.metrics_collector.update_position_metrics(positions, sum(abs(pos.get('profit', 0)) for pos in positions if pos.get('profit', 0) < 0))

            # Process each pair with parallel signal analysis
            async def process_pair_trading_logic(pair_data: Dict) -> None:
                """Process trading logic for a single pair."""
                try:
                    pair = pair_data['pair']

                    # PHASE 1 DIAGNOSTICS: Log processing start
                    logger.info(f"🔍 PHASE 1 DIAGNOSTICS - Processing {pair}")
                    logger.info(f"   Paper Mode Status: {self.paper_mode}")

                    # Create currency-specific logger at the very beginning
                    currency_logger = get_currency_logger(pair)
                    currency_logger.info(f"🔄 Starting processing for {pair}")

                    df_15m = pair_data['df_15m']
                    df_1h = pair_data['df_1h']
                    df_h4 = pair_data['df_h4']
                    spread_data = pair_data['spread_pips']
                    spread_pips = spread_data.get('spread_pips') if spread_data and isinstance(spread_data, dict) else None
                    sentiment_data = pair_data['sentiment_data']
                    # Always force sentiment_data to dict
                    if not isinstance(sentiment_data, dict):
                        currency_logger.warning(f"[SENTIMENT] Invalid sentiment_data type {type(sentiment_data)} for {pair}, forcing to empty dict.")
                        logger.warning(f"[SENTIMENT] Invalid sentiment_data type {type(sentiment_data)} for {pair}, forcing to empty dict.")
                        sentiment_data = {}

                    self.signals_analyzed += 1

                    # PHASE 1 DIAGNOSTICS: Log data quality
                    logger.info(f"   Data validation: 15M={len(df_15m) if df_15m is not None else 0}, 1H={len(df_1h) if df_1h is not None else 0}, H4={len(df_h4) if df_h4 is not None else 0}")

                    # Validate data quality
                    if len(df_15m) < 50:
                        logger.info(f"   ❌ REJECTION: Insufficient 15M data ({len(df_15m)} < 50)")
                        return

                    # Parallel signal analysis components
                    signal_task = self.adaptive_signal_analysis(pair, df_15m, df_1h, df_h4)
                    news_check_task = self.check_news_events(pair)

                    signal, news_ok = await asyncio.gather(signal_task, news_check_task, return_exceptions=True)

                    if isinstance(signal, Exception) or signal is None:
                        logger.debug(f"   {pair}: No valid signal generated")
                        return

                    # News filter
                    if isinstance(news_ok, Exception) or not news_ok:
                        logger.info(f"   {pair}: Trade skipped - High-impact news event detected")
                        self.signals_rejected += 1
                        return

                    # PHASE 1 DIAGNOSTICS: Pre-trade validation
                    logger.info(f"   Pre-trade validation for {pair}")
                    open_positions = len(positions)
                    balance = account_info['balance']
                    total_risk = sum(abs(pos.get('profit', 0) / balance) for pos in positions if pos.get('profit', 0) < 0)

                    logger.info(f"   Account: balance=${balance:.2f}, open_positions={open_positions}, total_risk={total_risk:.3f}")

                    # Risk checks
                    if total_risk > 0.05:  # 5% total risk limit
                        logger.info(f"   ❌ REJECTION: Total open risk {total_risk:.2%} exceeds 5% limit")
                        return

                    # Currency exposure check
                    exposure = self.correlation_analyzer.get_currency_exposure(positions)
                    base, quote = pair[:3], pair[3:]
                    base_exposure = abs(exposure.get(base, 0))
                    quote_exposure = abs(exposure.get(quote, 0))

                    logger.info(f"   Exposure: {base}={base_exposure:.1f}, {quote}={quote_exposure:.1f}")

                    if base_exposure > 2.0 or quote_exposure > 2.0:
                        logger.info(f"   ❌ REJECTION: Excessive exposure to {base}/{quote} (limit: 2.0)")
                        self.signals_rejected += 1
                        return

                    # Initialize volume scaling
                    volume_scale = 1.0

                    # Sentiment-based volume scaling


                    sentiment_score = safe_get_sentiment_value(sentiment_data, 'overall_sentiment', 0.0)
                    sentiment_confidence = safe_get_sentiment_value(sentiment_data, 'overall_confidence', 0.0)

                    if sentiment_confidence > 0.3:
                        if sentiment_score < -0.3:  # Very negative sentiment
                            logger.warning(f"   {pair}: Very negative sentiment ({sentiment_score:.2f}) - reducing volume")
                            volume_scale *= 0.3
                        elif sentiment_score < -0.1:  # Moderately negative
                            logger.warning(f"   {pair}: Negative sentiment ({sentiment_score:.2f}) - reducing volume")
                            volume_scale *= 0.6
                        elif sentiment_score > 0.3:  # Positive sentiment
                            logger.info(f"   {pair}: Positive sentiment ({sentiment_score:.2f}) - maintaining volume")
                        elif sentiment_score > 0.1:  # Moderately positive
                            logger.info(f"   {pair}: Moderately positive sentiment ({sentiment_score:.2f})")
                            volume_scale *= 1.1

                    # Get MPT weight and diversification score for this pair
                    mpt_weight = 0.0
                    diversification_score = 0.0
                    if mpt_weights and pair in mpt_weights:
                        mpt_weight = mpt_weights[pair]
                    if portfolio_metrics and pair in portfolio_metrics:
                        diversification_score = portfolio_metrics[pair].get('diversification', 0.0)

                    logger.info(f"   {pair}: Sentiment {sentiment_score:.2f} (confidence: {sentiment_confidence:.2f})")
                    logger.info(f"   {pair}: MPT weight {mpt_weight:.3f}, diversification {diversification_score:.2f}")
                    logger.info(f"   {pair}: Spread {spread_pips}p")

                    # PHASE 1 DIAGNOSTICS: Gate checks with detailed logging
                    logger.info(f"   Gate checks for {pair}")
                    gate_reasons = []

                    # Spread check
                    max_spread_soft = 100  # Set very high for debugging
                    logger.info(f"   Spread check: current={spread_pips}p, limit={max_spread_soft}p")
                    if spread_pips is not None and spread_pips > max_spread_soft:
                        gate_reasons.append(f"spread {spread_pips}p > {max_spread_soft}p")
                        logger.info(f"   ❌ SPREAD GATE: {spread_pips}p > {max_spread_soft}p")

                    # Position limit check
                    max_positions = 100  # Unrealistically high for debugging
                    logger.info(f"   Position check: current={open_positions}, limit={max_positions}")
                    if open_positions >= max_positions:
                        gate_reasons.append(f"positions {open_positions}/{max_positions}")
                        logger.info(f"   ❌ POSITION GATE: {open_positions} >= {max_positions}")

                    # Sentiment confidence check (relaxed for debugging)
                    logger.info(f"   Sentiment check: confidence={sentiment_confidence:.3f}")
                    # Temporarily disabled for debugging
                    # if sentiment_confidence < 0.0:
                    #     gate_reasons.append("low confidence")
                    #     logger.info(f"   ❌ SENTIMENT GATE: confidence {sentiment_confidence:.3f} < 0.0")

                    # Risk/Reward ratio check
                    rr_ratio = abs(signal['take_profit'] - signal['entry_price']) / abs(signal['stop_loss'] - signal['entry_price'])
                    min_rr_required = self.adaptive_params['min_rr_ratio']
                    logger.info(f"   R/R check: current={rr_ratio:.2f}, required={min_rr_required:.2f}")
                    if rr_ratio < min_rr_required:
                        gate_reasons.append(f"RR ratio {rr_ratio:.2f} < {min_rr_required:.2f}")
                        logger.info(f"   ❌ RR GATE: {rr_ratio:.2f} < {min_rr_required:.2f}")

                    # Confidence threshold check
                    signal_confidence = signal.get('confidence', 0)
                    min_confidence_required = self.adaptive_params['min_confidence']
                    logger.info(f"   Confidence check: current={signal_confidence:.3f}, required={min_confidence_required:.3f}")
                    if signal_confidence < min_confidence_required:
                        gate_reasons.append(f"confidence {signal_confidence:.3f} < {min_confidence_required:.3f}")
                        logger.info(f"   ❌ CONFIDENCE GATE: {signal_confidence:.3f} < {min_confidence_required:.3f}")

                    if gate_reasons:
                        logger.info(f"   ❌ FINAL REJECTION: {pair} blocked by {len(gate_reasons)} gates - {', '.join(gate_reasons)}")
                        self.signals_rejected += 1
                        return

                    logger.info(f"   ✅ ALL GATES PASSED: {pair} ready for trade execution")

                    # 🚨 TRADE FREQUENCY PROTECTION 🚨
                    logger.info(f"   🔒 TRADE FREQUENCY CHECKS for {pair}")

                    # Reset counters if needed
                    now = datetime.now()
                    if (now - self.hourly_reset_time).total_seconds() >= 3600:
                        self.hourly_trade_count = 0
                        self.hourly_reset_time = now
                        logger.info(f"   🔄 HOURLY COUNTER RESET: {self.hourly_trade_count} trades")

                    if (now - self.daily_reset_time).total_seconds() >= 86400:
                        self.daily_trade_count = 0
                        self.daily_reset_time = now
                        logger.info(f"   🔄 DAILY COUNTER RESET: {self.daily_trade_count} trades")

                    # Check hourly trade limit
                    if self.hourly_trade_count >= self.max_trades_per_hour:
                        logger.info(f"   ❌ HOURLY LIMIT: {self.hourly_trade_count}/{self.max_trades_per_hour} trades this hour")
                        logger.info(f"   ❌ TRADE REJECTED: {pair} - Hourly limit exceeded")
                        self.signals_rejected += 1
                        return

                    # Check daily trade limit
                    if self.daily_trade_count >= self.max_trades_per_day:
                        logger.info(f"   ❌ DAILY LIMIT: {self.daily_trade_count}/{self.max_trades_per_day} trades today")
                        logger.info(f"   ❌ TRADE REJECTED: {pair} - Daily limit exceeded")
                        self.signals_rejected += 1
                        return

                    # Check pair-specific cooldown
                    last_trade_time = self.last_trade_times.get(pair)
                    if last_trade_time:
                        time_since_last_trade = (now - last_trade_time).total_seconds()
                        if time_since_last_trade < self.min_trade_interval_seconds:
                            logger.info(f"   ❌ COOLDOWN: Last trade {time_since_last_trade:.0f}s ago, need {self.min_trade_interval_seconds}s")
                            logger.info(f"   ❌ TRADE REJECTED: {pair} - Cooldown active")
                            self.signals_rejected += 1
                            return

                    # Check pair-specific hourly limit
                    pair_hourly_count = self.pair_trade_counts.get(pair, 0)
                    if pair_hourly_count >= self.max_trades_per_pair_per_hour:
                        logger.info(f"   ❌ PAIR LIMIT: {pair_hourly_count}/{self.max_trades_per_pair_per_hour} trades this hour")
                        logger.info(f"   ❌ TRADE REJECTED: {pair} - Pair hourly limit exceeded")
                        self.signals_rejected += 1
                        return

                    # 🚀 INTELLIGENT OVERRIDE EVALUATION 🚀
                    override_granted = False
                    override_reasons = []

                    if self.override_enabled:
                        logger.info(f"   🎯 EVALUATING OVERRIDE CONDITIONS for {pair}")

                        # Check if override is possible (within daily limit)
                        if self.override_trades_today < self.max_override_trades_per_day:
                            logger.info(f"   Override available: {self.override_trades_today}/{self.max_override_trades_per_day} used today")

                            # Check override cooldown
                            if self.last_override_time:
                                time_since_override = (now - self.last_override_time).total_seconds() / 60  # minutes
                                if time_since_override >= self.override_cooldown_minutes:
                                    logger.info(f"   Cooldown OK: {time_since_override:.1f}min >= {self.override_cooldown_minutes}min")
                                else:
                                    logger.info(f"   Cooldown active: {time_since_override:.1f}min < {self.override_cooldown_minutes}min")
                                    override_granted = False
                            else:
                                logger.info(f"   No previous override - cooldown OK")

                            # Evaluate override conditions
                            if not override_granted:
                                # Condition 1: Exceptional confidence
                                confidence = signal.get('confidence', 0)
                                if confidence >= self.override_conditions['min_confidence_override']:
                                    override_reasons.append(f"confidence {confidence:.1%} >= {self.override_conditions['min_confidence_override']:.1%}")
                                    logger.info(f"   ✅ CONFIDENCE OVERRIDE: {confidence:.1%} >= {self.override_conditions['min_confidence_override']:.1%}")

                                # Condition 2: Exceptional risk/reward ratio
                                rr_ratio = abs(signal['take_profit'] - signal['entry_price']) / abs(signal['stop_loss'] - signal['entry_price'])
                                if rr_ratio >= self.override_conditions['min_rr_ratio_override']:
                                    override_reasons.append(f"RR ratio {rr_ratio:.1f} >= {self.override_conditions['min_rr_ratio_override']:.1f}")
                                    logger.info(f"   ✅ RR OVERRIDE: {rr_ratio:.1f} >= {self.override_conditions['min_rr_ratio_override']:.1f}")

                                # Condition 3: Low volatility (safer conditions)
                                df_recent = await self.data_manager.get_candles(pair, "M15", 20)
                                if df_recent is not None:
                                    df_recent = df_recent.reset_index()
                                if df_recent is not None and len(df_recent) >= 10:
                                    current_volatility = df_recent['close'].pct_change().std()
                                    if current_volatility <= self.override_conditions['max_volatility_override']:
                                        override_reasons.append(f"volatility {current_volatility:.6f} <= {self.override_conditions['max_volatility_override']:.6f}")
                                        logger.info(f"   ✅ VOLATILITY OVERRIDE: {current_volatility:.6f} <= {self.override_conditions['max_volatility_override']:.6f}")

                                # Condition 4: Recent performance (if available)
                                if hasattr(self, 'daily_trade_data') and len(self.daily_trade_data) >= 5:
                                    recent_trades = list(self.daily_trade_data)[-5:]  # Last 5 trades
                                    profitable_trades = sum(1 for trade in recent_trades if trade.get('profit', 0) > 0)
                                    win_rate_recent = profitable_trades / len(recent_trades)

                                    if win_rate_recent >= self.override_conditions['min_win_rate_recent']:
                                        override_reasons.append(f"recent win rate {win_rate_recent:.1%} >= {self.override_conditions['min_win_rate_recent']:.1%}")
                                        logger.info(f"   ✅ PERFORMANCE OVERRIDE: Recent win rate {win_rate_recent:.1%} >= {self.override_conditions['min_win_rate_recent']:.1%}")

                                # Condition 5: Optimal trading session
                                if self.override_conditions['session_optimization']:
                                    current_hour = now.hour
                                    # London (8-11), NY (13-16) sessions are optimal
                                    if (8 <= current_hour <= 11) or (13 <= current_hour <= 16):
                                        override_reasons.append(f"optimal session (hour {current_hour})")
                                        logger.info(f"   ✅ SESSION OVERRIDE: Optimal trading hour {current_hour}")

                                # Grant override if we have at least 2 conditions met
                                if len(override_reasons) >= 2:
                                    override_granted = True
                                    logger.info(f"   🎯 OVERRIDE GRANTED: {len(override_reasons)} conditions met")
                                    for reason in override_reasons:
                                        logger.info(f"      - {reason}")
                                else:
                                    logger.info(f"   ❌ OVERRIDE DENIED: Only {len(override_reasons)} conditions met (need 2+)")
                        else:
                            logger.info(f"   ❌ OVERRIDE UNAVAILABLE: {self.override_trades_today}/{self.max_override_trades_per_day} used today")

                    # Handle frequency limit violations with override
                    frequency_violations = []

                    if self.hourly_trade_count >= self.max_trades_per_hour:
                        frequency_violations.append("hourly_limit")
                    if self.daily_trade_count >= self.max_trades_per_day:
                        frequency_violations.append("daily_limit")
                    if pair_hourly_count >= self.max_trades_per_pair_per_hour:
                        frequency_violations.append("pair_limit")
                    if last_trade_time and time_since_last_trade < self.min_trade_interval_seconds:
                        frequency_violations.append("cooldown")

                    if frequency_violations:
                        if override_granted:
                            logger.info(f"   🚀 OVERRIDE ACTIVATED: Bypassing {len(frequency_violations)} frequency limits")
                            logger.info(f"      Violations: {', '.join(frequency_violations)}")
                            self.override_trades_today += 1
                            self.last_override_time = now
                        else:
                            logger.info(f"   ❌ FREQUENCY LIMITS BLOCKED: {len(frequency_violations)} violations - {', '.join(frequency_violations)}")
                            logger.info(f"   ❌ TRADE REJECTED: {pair} - Frequency limits exceeded")
                            self.signals_rejected += 1
                            return

                    if not frequency_violations or override_granted:
                        logger.info(f"   ✅ FREQUENCY CHECKS PASSED: {pair}")
                        if override_granted:
                            logger.info(f"   🎯 OVERRIDE MODE: Exceptional opportunity detected")
                        logger.info(f"   Current counts: Hourly={self.hourly_trade_count}/{self.max_trades_per_hour}, Daily={self.daily_trade_count}/{self.max_trades_per_day}, Pair={pair_hourly_count}/{self.max_trades_per_pair_per_hour}")

                    # PHASE 1 DIAGNOSTICS: Trade execution section
                    logger.info(f"   🚀 TRADE EXECUTION PHASE for {pair}")
                    logger.info(f"   Signal details: direction={signal['direction'].value}, confidence={signal.get('confidence', 0):.3f}")
                    logger.info(f"   Entry: {signal['entry_price']:.5f}, SL: {signal['stop_loss']:.5f}, TP: {signal['take_profit']:.5f}")

                    # PAPER_MODE: Simulate trade instead of placing real order
                    if self.paper_mode:
                        logger.info(f"   📝 PAPER MODE: Simulating trade execution")
                        logger.info(f"   ✅ PAPER TRADE EXECUTED: {pair} {signal['direction'].value}")
                        self.trades_executed += 1

                        # 🚨 UPDATE TRADE FREQUENCY COUNTERS (PAPER MODE) 🚨
                        now = datetime.now()
                        self.last_trade_times[pair] = now
                        self.hourly_trade_count += 1
                        self.daily_trade_count += 1
                        self.pair_trade_counts[pair] = self.pair_trade_counts.get(pair, 0) + 1

                        logger.info(f"   📊 COUNTERS UPDATED: Hourly={self.hourly_trade_count}/{self.max_trades_per_hour}, Daily={self.daily_trade_count}/{self.max_trades_per_day}, Pair={self.pair_trade_counts[pair]}/{self.max_trades_per_pair_per_hour}")

                        # Simulate a trade result dict for downstream logic
                        trade_data = {
                            'timestamp': datetime.now().isoformat(),
                            'ticket': f"SIM-{self.trades_executed}",
                            'symbol': pair,
                            'pair': pair,  # Ensure compatibility with correlation analyzer
                            'direction': signal['direction'].value,
                            'entry_price': signal['entry_price'],
                            'stop_loss': signal['stop_loss'],
                            'take_profit': signal['take_profit'],
                            'volume': volume_scale,  # Notional volume
                            'confidence': signal.get('confidence', 0),
                            'adaptive_params_used': self.adaptive_params.copy(),
                            'spread_pips': spread_pips,
                            'open_positions': open_positions,
                            'exit_reason': 'entry',
                            'profit': 0,
                            'hold_duration': 0,
                            'market_conditions': {
                                'volatility_15m': df_15m['close'].pct_change().std() if len(df_15m) > 1 else 0,
                                'volatility_1h': df_1h['close'].pct_change().std() if len(df_1h) > 1 else 0,
                                'price_at_entry': df_15m['close'].iloc[-1] if len(df_15m) > 0 else 0
                            },
                            'sentiment_data': sentiment_data,
                            'correlation_data': {},
                            'volatility_level': self.get_volatility_level(
                                df_15m['close'].pct_change().std() if len(df_15m) > 1 else 0, pair
                            ),
                            'session': 'normal'
                        }
                        self.daily_trade_data.append(trade_data)
                        self.weekly_trade_data.append(trade_data)
                        return

                    # LIVE TRADING MODE
                    logger.info(f"   💰 LIVE TRADING MODE: Executing real trade")
                    logger.info(f"🚀 EXECUTING ADAPTIVE TRADE: {pair}")

                    # PHASE 1 DIAGNOSTICS: Risk-based position sizing
                    logger.info(f"   Risk calculation for {pair}")
                    sl_pips = abs(signal['entry_price'] - signal['stop_loss']) * 10000
                    pip_value = 10  # $10/pip for 1 lot on majors
                    risk_amount = balance * 0.01  # 1% risk per trade

                    logger.info(f"   SL pips: {sl_pips:.1f}, Risk amount: ${risk_amount:.2f}, Pip value: ${pip_value}")
                    volume = round((risk_amount / (sl_pips * pip_value)), 2)
                    logger.info(f"   Initial volume calculation: {volume}")

                    # Confidence and volume adjustments
                    confidence = signal.get('confidence', 0)
                    logger.info(f"   Confidence adjustment: {confidence:.3f}")
                    if confidence >= 0.9:
                        volume = min(volume * 1.2, volume)
                        logger.info(f"   High confidence boost: volume → {volume}")
                    elif confidence < 0.8:
                        volume *= 0.8
                        logger.info(f"   Low confidence reduction: volume → {volume}")

                    volume *= volume_scale
                    logger.info(f"   Volume scale applied ({volume_scale:.2f}): volume → {volume}")
                    volume = max(0.01, min(volume, 1.0))
                    logger.info(f"   Volume bounds applied: volume → {volume}")

                    # Final risk check
                    logger.info(f"   Final risk check for {pair}")
                    current_positions = await self.broker_manager.get_positions()
                    total_risk = sum([abs(pos.get('profit', 0)) for pos in current_positions if pos.get('profit', 0) < 0])
                    logger.info(f"   Current positions: {len(current_positions)}, Total risk: ${total_risk:.2f}")

                    if total_risk > balance * 0.05:
                        logger.info(f"   ❌ RISK LIMIT: Total portfolio risk ${total_risk:.2f} > ${balance * 0.05:.2f} (5% limit)")
                        volume *= 0.5
                        logger.info(f"   Risk reduction applied: volume → {volume}")

                    if volume < 0.01:
                        logger.info(f"   ❌ VOLUME LIMIT: Volume {volume} < 0.01 minimum")
                        logger.info(f"   ❌ TRADE REJECTED: {pair} - Volume too low after risk adjustments")
                        return

                    logger.info(f"   ✅ VOLUME APPROVED: {volume} lots for {pair}")

                    # PHASE 1 DIAGNOSTICS: Order placement
                    logger.info(f"   📝 Placing order for {pair}")
                    logger.info(f"   Order details: {signal['direction'].value}, volume={volume:.2f}, SL={signal['stop_loss']:.5f}, TP={signal['take_profit']:.5f}")

                    order_result = await self.broker_manager.place_order(
                        symbol=pair,
                        order_type=signal['direction'].value,
                        volume=round(volume, 2),
                        sl=signal['stop_loss'],
                        tp=signal['take_profit']
                    )

                    logger.info(f"   Order result: {order_result}")

                    if order_result and order_result.get('ticket'):
                        logger.info(f"   ✅ ORDER SUCCESSFUL: {pair} ticket #{order_result['ticket']}")
                        self.trades_executed += 1

                        # 🚨 UPDATE TRADE FREQUENCY COUNTERS 🚨
                        now = datetime.now()
                        self.last_trade_times[pair] = now
                        self.hourly_trade_count += 1
                        self.daily_trade_count += 1
                        self.pair_trade_counts[pair] = self.pair_trade_counts.get(pair, 0) + 1

                        logger.info(f"   📊 COUNTERS UPDATED: Hourly={self.hourly_trade_count}/{self.max_trades_per_hour}, Daily={self.daily_trade_count}/{self.max_trades_per_day}, Pair={self.pair_trade_counts[pair]}/{self.max_trades_per_pair_per_hour}")

                        # Record trade metrics only if metrics_collector is available
                        if self.metrics_collector is not None:
                            self.metrics_collector.record_trade(signal, round(volume, 2), sentiment_data)

                        # Record trade for attribution analysis
                        trade_data = {
                            'timestamp': datetime.now().isoformat(),
                            'ticket': order_result['ticket'],
                            'symbol': pair,
                            'direction': signal['direction'].value,
                            'entry_price': signal['entry_price'],
                            'stop_loss': signal['stop_loss'],
                            'take_profit': signal['take_profit'],
                            'volume': round(volume, 2),
                            'confidence': confidence,
                            'adaptive_params_used': self.adaptive_params.copy(),
                            'spread_pips': spread_pips,
                            'open_positions': open_positions,
                            'exit_reason': 'entry',  # Will be updated when trade closes
                            'profit': 0,  # Will be updated when trade closes
                            'hold_duration': 0,  # Will be updated when trade closes
                            'market_conditions': {
                                'volatility_15m': df_15m['close'].pct_change().std() if len(df_15m) > 1 else 0,
                                'volatility_1h': df_1h['close'].pct_change().std() if len(df_1h) > 1 else 0,
                                'price_at_entry': df_15m['close'].iloc[-1] if len(df_15m) > 0 else 0
                            },
                            'sentiment_data': sentiment_data,
                            'correlation_data': {},
                            'volatility_level': self.get_volatility_level(
                                df_15m['close'].pct_change().std() if len(df_15m) > 1 else 0, pair
                            ),
                            'session': 'normal'  # Could be enhanced with session detection
                        }

                        # Add trade to attribution analyzer if available
                        if self.attribution_analyzer is not None:
                            self.attribution_analyzer.add_trade(trade_data)

                        self.daily_trade_data.append(trade_data)
                        self.weekly_trade_data.append(trade_data)

                        logger.info(f"✅ ADAPTIVE TRADE EXECUTED: {pair} #{order_result['ticket']}")

                    else:
                        logger.info(f"   ❌ ORDER FAILED: {pair} - No ticket returned")
                        logger.info(f"   Order result details: {order_result}")

                except Exception as e:
                    logger.error(f"Error processing pair {pair_data.get('pair', 'unknown')}: {e}")

            # Process all pairs in parallel
            processing_tasks = [process_pair_trading_logic(pair_data) for pair_data in valid_pair_data]
            await asyncio.gather(*processing_tasks, return_exceptions=True)

            logger.info(f"🔄 Parallel processing completed for {len(valid_pair_data)} pairs")

            # Record scan performance only if metrics_collector is available
            scan_duration = time.time() - scan_start_time
            if self.metrics_collector is not None:
                self.metrics_collector.record_scan_performance(
                    scan_duration=scan_duration,
                    pairs_processed=len(valid_pair_data),
                    signals_generated=self.signals_analyzed - self.signals_rejected,
                    trades_executed=self.trades_executed
                )

            self.scan_count += 1

        except Exception as e:
            logger.error(f"Error in adaptive trading scan: {e}")
            if self.metrics_collector:
                self.metrics_collector.increment_error_count("scan_error")
            
    async def run_adaptive_intelligent_bot(self):
        """Main adaptive intelligent bot loop."""
        logger.info("🧠 STARTING ADAPTIVE INTELLIGENT TRADING BOT")
        logger.info("="*80)
        logger.info("🚀 ADAPTIVE FEATURES:")
        logger.info("   - Real-time trend reversal detection")
        logger.info("   - ML-driven daily analysis (US-Japan gap)")
        logger.info("   - Comprehensive weekend learning")
        logger.info("   - Adaptive parameter optimization")
        logger.info("   - Intelligent scheduling")
        logger.info("   - Enhanced profit protection")
        logger.info("="*80)
        print("[RUN] Main bot loop starting...")
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        while self.running:
            print(f"[RUN] Bot loop iteration {self.scan_count + 1}...")
            try:
                await self.adaptive_trading_scan()
                print(f"[RUN] Completed trading scan #{self.scan_count + 1}")
                self.scan_count += 1
            except Exception as e:
                print(f"[ERROR] Exception in main loop: {e}")
            await asyncio.sleep(1)  # Prevent tight loop
        
        try:
            # Initialize core components (always available)
            await self.data_manager.initialize()
            await self.broker_manager.initialize()

            # Check trading permissions and account status
            logger.info("🔍 Checking trading permissions and account status...")
            permissions_check = await self.broker_manager.check_trading_permissions()
            if 'error' in permissions_check:
                logger.error(f"❌ Trading permissions check failed: {permissions_check['error']}")
            else:
                logger.info("📊 Account Status:")
                logger.info(f"   Login: {permissions_check['account_info']['login']}")
                logger.info(f"   Balance: ${permissions_check['account_info']['balance']:,.2f}")
                logger.info(f"   Free Margin: ${permissions_check['account_info']['margin_free']:,.2f}")
                logger.info(f"   Terminal Connected: {permissions_check['terminal_info']['connected']}")
                logger.info(f"   Trading Allowed: {permissions_check['terminal_info']['trade_allowed']}")

                logger.info("📊 Symbol Status:")
                for symbol, status in permissions_check['symbol_status'].items():
                    if 'error' in status:
                        logger.error(f"   {symbol}: {status['error']}")
                    else:
                        logger.info(f"   {symbol}: Visible={status['visible']}, Selected={status['selected']}, Spread={status['spread']}p, Trading={status['trading_enabled']}")

            # Initialize advanced components lazily (only if available)
            if METRICS_AVAILABLE and self.metrics_collector:
                await self.metrics_collector.start()
                logger.info("✅ Metrics collector initialized")
            else:
                logger.info("⚠️ Metrics collector not available - monitoring disabled")

            if SCHEDULER_AVAILABLE and self.scheduler:
                self.scheduler.register_mode_change_callback(self.mode_change_handler)
                scheduler_task = asyncio.create_task(self.scheduler.monitor_mode_changes())
                logger.info("✅ Intelligent scheduler initialized")
            else:
                logger.info("⚠️ Intelligent scheduler not available - basic mode switching disabled")
                scheduler_task = None

            # Log component availability
            logger.info("🔧 COMPONENT STATUS:")
            logger.info(f"   Reversal Detector: {'✅' if REVERSAL_DETECTOR_AVAILABLE and self.reversal_detector else '❌'}")
            logger.info(f"   ML Engine: {'✅' if ML_ENGINE_AVAILABLE and self.ml_engine else '❌'}")
            logger.info(f"   Trade Analyzer: {'✅' if TRADE_ANALYZER_AVAILABLE and self.trade_analyzer else '❌'}")
            logger.info(f"   Intelligent Scheduler: {'✅' if SCHEDULER_AVAILABLE and self.scheduler else '❌'}")
            logger.info(f"   Sentiment Aggregator: {'✅' if SENTIMENT_AVAILABLE and self.sentiment_aggregator else '❌'}")
            logger.info(f"   Metrics Collector: {'✅' if METRICS_AVAILABLE and self.metrics_collector else '❌'}")
            logger.info(f"   Attribution Analyzer: {'✅' if ATTRIBUTION_AVAILABLE and self.attribution_analyzer else '❌'}")

            logger.info("✅ Adaptive intelligent bot initialized with Phase 1 improvements")
            
            # Main adaptive loop
            while self.running:
                try:
                    # Write heartbeat
                    await self.write_heartbeat()
                    
                    # Check current mode
                    if await self.scheduler.should_execute_trades(self.broker_manager):
                        await self.adaptive_trading_scan()
                        
                        # Status update every 3 scans
                        if self.scan_count % 3 == 0:
                            account_info = await self.broker_manager.get_account_info()
                            positions = await self.broker_manager.get_positions()
                            
                            logger.info(f"🤖 ADAPTIVE BOT STATUS:")
                            logger.info(f"   Mode: {self.current_mode}")
                            logger.info(f"   Scans: {self.scan_count}")
                            logger.info(f"   Trades: {self.trades_executed}")
                            logger.info(f"   Reversals Detected: {self.reversals_detected}")
                            logger.info(f"   ML Analyses: {self.ml_analyses_performed}")
                            logger.info(f"   Balance: ${account_info['balance']:,.2f}")
                            logger.info(f"   Positions: {len(positions)}")
                            logger.info(f"   Adaptive Confidence: {self.adaptive_params['min_confidence']:.1%}")
                            
                        # Wait 2 minutes
                        for i in range(12):
                            if not self.running:
                                break
                            await asyncio.sleep(10)
                            if i % 3 == 0:
                                await self.write_heartbeat()
                except Exception as e:
                    logger.error(f"Error in main loop iteration: {e}")
                    await asyncio.sleep(60)  # Wait before retrying
                    
        except Exception as e:
            logger.error(f"Critical error in bot execution: {e}")
            traceback.print_exc()
        finally:
            logger.info("🛑 Adaptive intelligent bot shutting down...")
            await self.write_heartbeat()

if __name__ == "__main__":
    bot = AdaptiveIntelligentBot()
    asyncio.run(bot.run_adaptive_intelligent_bot())
