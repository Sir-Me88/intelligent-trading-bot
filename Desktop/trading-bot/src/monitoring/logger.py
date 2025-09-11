"""Structured logging configuration."""

import logging
import logging.config
import json
import sys
from datetime import datetime
from typing import Dict, Any

from ..config.settings import settings


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class TradingLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for trading-specific context."""
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Add trading context to log messages."""
        extra = kwargs.get('extra', {})
        
        # Add default trading context
        extra.update({
            'component': 'trading_bot',
            'environment': settings.environment
        })
        
        kwargs['extra'] = extra
        return msg, kwargs


def setup_logging():
    """Setup structured logging configuration."""
    
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'json': {
                '()': JSONFormatter
            },
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': settings.log_level,
                'formatter': 'json' if settings.environment == 'production' else 'standard',
                'stream': sys.stdout
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': settings.log_level,
                'formatter': 'json',
                'filename': 'logs/forex_bot.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            }
        },
        'loggers': {
            'forex_bot': {
                'level': settings.log_level,
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'aiohttp': {
                'level': 'WARNING',
                'handlers': ['console'],
                'propagate': False
            },
            'urllib3': {
                'level': 'WARNING',
                'handlers': ['console'],
                'propagate': False
            }
        },
        'root': {
            'level': 'WARNING',
            'handlers': ['console']
        }
    }
    
    # Create logs directory if it doesn't exist
    import os
    os.makedirs('logs', exist_ok=True)
    
    logging.config.dictConfig(config)


def get_trading_logger(name: str) -> TradingLoggerAdapter:
    """Get a trading logger with context."""
    logger = logging.getLogger(f"forex_bot.{name}")
    return TradingLoggerAdapter(logger, {})


# Trade execution logger
def log_trade_execution(pair: str, direction: str, size: float, price: float, 
                       stop_loss: float = None, take_profit: float = None,
                       sentiment: float = None, pattern: str = None):
    """Log trade execution with structured data."""
    logger = get_trading_logger('trades')
    
    trade_data = {
        'action': 'trade_execution',
        'pair': pair,
        'direction': direction,
        'size': size,
        'price': price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'sentiment': sentiment,
        'pattern': pattern
    }
    
    logger.info("Trade executed", extra=trade_data)


def log_position_update(position_id: str, pair: str, unrealized_pnl: float, 
                       current_price: float):
    """Log position update with structured data."""
    logger = get_trading_logger('positions')
    
    position_data = {
        'action': 'position_update',
        'position_id': position_id,
        'pair': pair,
        'unrealized_pnl': unrealized_pnl,
        'current_price': current_price
    }
    
    logger.info("Position updated", extra=position_data)


def log_signal_generation(pair: str, direction: str, pattern: str, 
                         confidence: float, timeframe: str):
    """Log signal generation with structured data."""
    logger = get_trading_logger('signals')
    
    signal_data = {
        'action': 'signal_generation',
        'pair': pair,
        'direction': direction,
        'pattern': pattern,
        'confidence': confidence,
        'timeframe': timeframe
    }
    
    logger.info("Signal generated", extra=signal_data)


def log_risk_check(pair: str, approved: bool, reason: str, 
                  position_size: float = None, risk_amount: float = None):
    """Log risk management check with structured data."""
    logger = get_trading_logger('risk')
    
    risk_data = {
        'action': 'risk_check',
        'pair': pair,
        'approved': approved,
        'reason': reason,
        'position_size': position_size,
        'risk_amount': risk_amount
    }
    
    logger.info("Risk check completed", extra=risk_data)

