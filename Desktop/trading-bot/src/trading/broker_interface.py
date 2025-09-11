#!/usr/bin/env python3
"""Enhanced MT5 broker interface for production trading."""

import MetaTrader5 as mt5
from typing import Dict, List, Optional
import logging
import asyncio
import os
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class OrderResult(Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"

class EnhancedMT5BrokerInterface:
    """Enhanced MT5 broker interface with retry logic and robust error handling."""
    
    def __init__(self):
        self.mt5 = mt5
        self.connected = False
        self.initialized = False
        self.max_retries = 3
        self.retry_delay = 5
        self.max_spread_pips = 15
        self.max_deviation = 5
        
        self.login = os.getenv('MT5_LOGIN')
        self.password = os.getenv('MT5_PASSWORD')
        self.server = os.getenv('MT5_SERVER')
    
    async def initialize(self) -> bool:
        """Initialize the broker interface."""
        try:
            result = await self.connect()
            if result:
                self.initialized = True
            return result
        except Exception as e:
            logger.error(f"Failed to initialize broker interface: {e}")
            self.initialized = False
            return False
    
    async def connect(self) -> bool:
        """Initialize MT5 connection with retry logic."""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"🔌 MT5 connection attempt {attempt + 1}/{self.max_retries}")
                
                if not self.mt5.initialize():
                    logger.warning(f"MT5 initialize failed on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
                        continue
                    raise ConnectionError("MT5 initialization failed after all retries")
                
                if self.login and self.password and self.server:
                    if not self.mt5.login(
                        login=int(self.login),
                        password=self.password,
                        server=self.server
                    ):
                        error = self.mt5.last_error()
                        logger.warning(f"MT5 login failed on attempt {attempt + 1}: {error}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay)
                            continue
                        raise ConnectionError(f"MT5 login failed: {error}")
                
                account_info = self.mt5.account_info()
                if account_info is None:
                    logger.warning(f"Failed to get account info on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
                        continue
                    raise ConnectionError("Failed to verify account info")
                
                self.connected = True
                logger.info("✅ MT5 connection established")
                return True
                
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    self.connected = False
                    return False
                await asyncio.sleep(self.retry_delay)
        
        return False
    
    async def get_account_info(self) -> Dict:
        """Get account information."""
        try:
            account_info = self.mt5.account_info()
            if account_info is None:
                raise ValueError("Failed to get account info")
            return {
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'margin_free': account_info.margin_free
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {'balance': 0.0, 'equity': 0.0, 'margin': 0.0, 'margin_free': 0.0}
    
    async def get_positions(self) -> List[Dict]:
        """Get open positions."""
        try:
            positions = self.mt5.positions_get()
            if positions is None:
                raise ValueError("Failed to get positions")
            return [{
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': pos.type,
                'volume': pos.volume,
                'price_open': pos.price_open,
                'price_current': pos.price_current,
                'profit': pos.profit,
                'sl': pos.sl,
                'tp': pos.tp
            } for pos in positions]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    async def place_order(self, symbol: str, order_type: str, volume: float,
                         stop_loss: float = None, take_profit: float = None) -> Dict:
        """Place an order with enhanced error handling and slippage retries."""
        try:
            symbol_info = self.mt5.symbol_info(symbol)
            if symbol_info is None:
                raise ValueError(f"Invalid symbol: {symbol}")
                
            order_dict = {
                'action': self.mt5.TRADE_ACTION_DEAL,
                'symbol': symbol,
                'volume': volume,
                'type': self.mt5.ORDER_TYPE_BUY if order_type.upper() == 'BUY' else self.mt5.ORDER_TYPE_SELL,
                'price': symbol_info.bid if order_type.upper() == 'BUY' else symbol_info.ask,
                'sl': stop_loss,
                'tp': take_profit,
                'deviation': self.max_deviation,
                'type_time': self.mt5.ORDER_TIME_GTC,
                'type_filling': self.mt5.ORDER_FILLING_IOC
            }
            
            for attempt in range(self.max_retries):
                result = self.mt5.order_send(order_dict)
                if result is None:
                    raise ValueError("Order send failed")
                
                if result.retcode == self.mt5.TRADE_RETCODE_DONE:
                    return {
                        'status': OrderResult.SUCCESS.value,
                        'ticket': result.order,
                        'retcode': result.retcode,
                        'comment': result.comment
                    }
                elif result.retcode == self.mt5.TRADE_RETCODE_PRICE_CHANGED:
                    logger.warning(f"Slippage on {symbol}; retry {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
                        order_dict['price'] = self.mt5.symbol_info(symbol).bid if order_type.upper() == 'BUY' else self.mt5.symbol_info(symbol).ask
                        continue
                    return {
                        'status': OrderResult.REJECTED.value,
                        'ticket': None,
                        'retcode': result.retcode,
                        'comment': 'Slippage retries exhausted'
                    }
                else:
                    return {
                        'status': OrderResult.FAILED.value,
                        'ticket': None,
                        'retcode': result.retcode,
                        'comment': result.comment
                    }
                    
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return {
                'status': OrderResult.FAILED.value,
                'ticket': None,
                'retcode': -1,
                'comment': str(e)
            }
    
    async def validate_spread(self, symbol: str) -> bool:
        """Validate spread for the symbol."""
        try:
            symbol_info = self.mt5.symbol_info(symbol)
            if symbol_info is None or symbol_info.spread > self.max_spread_pips:
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating spread for {symbol}: {e}")
            return False
    
    async def get_spread_pips(self, symbol: str) -> Optional[Dict]:
        """Get current spread and pip value in pips for symbol."""
        try:
            symbol_info = self.mt5.symbol_info(symbol)
            if symbol_info is None:
                return None
            return {
                'spread_pips': int(symbol_info.spread),
                'pip_value': symbol_info.point * 10
            }
        except Exception as e:
            logger.error(f"get_spread_pips error for {symbol}: {e}")
            return None
    
    async def close_position(self, ticket: int) -> Dict:
        """Close a position."""
        try:
            position = self.mt5.positions_get(ticket=ticket)
            if not position:
                raise ValueError(f"Position {ticket} not found")
                
            position = position[0]
            order_dict = {
                'action': self.mt5.TRADE_ACTION_DEAL,
                'position': ticket,
                'symbol': position.symbol,
                'volume': position.volume,
                'type': self.mt5.ORDER_TYPE_SELL if position.type == self.mt5.ORDER_TYPE_BUY else self.mt5.ORDER_TYPE_BUY,
                'price': self.mt5.symbol_info(position.symbol).bid if position.type == self.mt5.ORDER_TYPE_BUY else self.mt5.symbol_info(position.symbol).ask,
                'type_time': self.mt5.ORDER_TIME_GTC,
                'type_filling': self.mt5.ORDER_FILLING_IOC
            }
            
            result = self.mt5.order_send(order_dict)
            if result is None or result.retcode != self.mt5.TRADE_RETCODE_DONE:
                raise ValueError(f"Failed to close position {ticket}: {result.comment if result else 'No result'}")
                
            return {
                'status': OrderResult.SUCCESS.value,
                'ticket': ticket,
                'retcode': result.retcode,
                'comment': result.comment
            }
        except Exception as e:
            logger.error(f"Error closing position {ticket}: {e}")
            return {
                'status': OrderResult.FAILED.value,
                'ticket': ticket,
                'retcode': -1,
                'comment': str(e)
            }

class BrokerManager:
    """Enhanced broker manager using the new MT5 interface."""
    
    def __init__(self, use_enhanced: bool = True):
        if use_enhanced:
            self.broker = EnhancedMT5BrokerInterface()
        else:
            self.broker = MT5BrokerInterface()
        self.enhanced = use_enhanced
    
    async def initialize(self) -> bool:
        """Initialize MT5 broker."""
        try:
            if self.enhanced:
                return await self.broker.connect()
            else:
                await self.broker.initialize()
                return True
        except Exception as e:
            logger.error(f"Failed to initialize broker: {e}")
            return False
    
    async def get_account_info(self) -> Dict:
        """Get account information."""
        return await self.broker.get_account_info()
    
    async def get_positions(self) -> List[Dict]:
        """Get open positions."""
        return await self.broker.get_positions()
    
    async def place_order(self, symbol: str, order_type: str, volume: float,
                         sl: float = None, tp: float = None) -> Dict:
        """Place an order with enhanced error handling."""
        if self.enhanced:
            return await self.broker.place_order(symbol, order_type, volume, sl, tp)
        else:
            return await self.broker.place_order(symbol, order_type, volume, None, sl, tp)
    
    async def validate_spread(self, symbol: str) -> bool:
        """Validate spread for the symbol."""
        if self.enhanced:
            return await self.broker.validate_spread(symbol)
        else:
            try:
                import MetaTrader5 as mt5
                symbol_info = mt5.symbol_info(symbol)
                return symbol_info is not None and symbol_info.spread < 20
            except:
                return True
    
    async def get_spread_pips(self, symbol: str) -> Optional[Dict]:
        """Get current spread and pip value."""
        return await self.broker.get_spread_pips(symbol)
    
    def is_connected(self) -> bool:
        """Check if broker is connected."""
        if self.enhanced:
            return self.broker.connected
        else:
            return getattr(self.broker, 'initialized', False)
    
    async def close_position(self, ticket: int) -> Dict:
        """Close a position."""
        return await self.broker.close_position(ticket)
