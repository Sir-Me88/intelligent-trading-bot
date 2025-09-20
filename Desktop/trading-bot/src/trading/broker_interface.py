#!/usr/bin/env python3
"""Enhanced MT5 broker interface with proper async handling."""

import MetaTrader5 as mt5
from typing import Dict, List, Optional
import logging
import asyncio
import os
import concurrent.futures
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class OrderResult(Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"

class EnhancedMT5BrokerInterface:
    """Enhanced MT5 broker interface with proper async handling using thread pools."""

    def __init__(self):
        self.mt5 = mt5
        self.connected = False
        self.initialized = False
        self.max_retries = 3
        self.retry_delay = 5
        self.max_spread_pips = 20  # Balanced spread limit for quality trades
        self.max_deviation = 5

        # Thread pool for async MT5 calls
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        # Get MT5 credentials from settings
        from src.config.settings import settings
        self.login = settings.mt5_login
        self.password = settings.mt5_password
        self.server = settings.mt5_server

    def __del__(self):
        """Cleanup thread pool on destruction."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

    def _decode_mt5_error(self, retcode: int) -> str:
        """Decode MT5 return codes to human-readable descriptions."""
        error_codes = {
            10004: "TRADE_RETCODE_REQUOTE - Requote",
            10006: "TRADE_RETCODE_REJECT - Request rejected",
            10007: "TRADE_RETCODE_CANCEL - Request canceled by trader",
            10008: "TRADE_RETCODE_PLACED - Order placed",
            10009: "TRADE_RETCODE_DONE - Request completed",
            10010: "TRADE_RETCODE_DONE_PARTIAL - Only part of the request was completed",
            10011: "TRADE_RETCODE_ERROR - Request processing error",
            10012: "TRADE_RETCODE_TIMEOUT - Request timed out",
            10013: "TRADE_RETCODE_INVALID - Invalid request",
            10014: "TRADE_RETCODE_INVALID_VOLUME - Invalid volume in the request",
            10015: "TRADE_RETCODE_INVALID_PRICE - Invalid price in the request",
            10016: "TRADE_RETCODE_INVALID_STOPS - Invalid stops in the request",
            10017: "TRADE_RETCODE_TRADE_DISABLED - Trade is disabled",
            10018: "TRADE_RETCODE_MARKET_CLOSED - Market is closed",
            10019: "TRADE_RETCODE_NO_MONEY - Not enough money to complete the request",
            10020: "TRADE_RETCODE_PRICE_CHANGED - Prices changed",
            10021: "TRADE_RETCODE_PRICE_OFF - Prices are off the current prices",
            10022: "TRADE_RETCODE_INVALID_EXPIRATION - Invalid order expiration date in the request",
            10023: "TRADE_RETCODE_ORDER_CHANGED - Order state changed",
            10024: "TRADE_RETCODE_TOO_MANY_REQUESTS - Too frequent requests",
            10025: "TRADE_RETCODE_NO_CHANGES - No changes in request",
            10026: "TRADE_RETCODE_SERVER_DISABLES_AT - Auto trading disabled by server",
            10027: "TRADE_RETCODE_CLIENT_DISABLES_AT - Auto trading disabled by client",
            10028: "TRADE_RETCODE_LOCKED - Request locked for processing",
            10029: "TRADE_RETCODE_FROZEN - Order or position frozen",
            10030: "TRADE_RETCODE_INVALID_FILL - Invalid order filling type",
            10031: "TRADE_RETCODE_CONNECTION - No connection with the trade server",
            10032: "TRADE_RETCODE_ONLY_REAL - Operation is allowed only for real accounts",
            10033: "TRADE_RETCODE_LIMIT_ORDERS - The number of pending orders has reached the limit",
            10034: "TRADE_RETCODE_LIMIT_VOLUME - The volume of orders and positions for the symbol has reached the limit",
            10035: "TRADE_RETCODE_INVALID_ORDER - Incorrect or prohibited order type",
            10036: "TRADE_RETCODE_POSITION_CLOSED - Position with the specified POSITION_IDENTIFIER has already been closed",
            10037: "TRADE_RETCODE_INVALID_CLOSE_VOLUME - A close volume exceeds the current position volume",
            10038: "TRADE_RETCODE_CLOSE_ORDER_EXIST - A close order already exists for the position",
            10039: "TRADE_RETCODE_LIMIT_POSITIONS - The number of open positions has reached the limit",
            10040: "TRADE_RETCODE_REJECT_CANCEL - The order to cancel the pending order was rejected",
            10041: "TRADE_RETCODE_LONG_ONLY - The request is rejected because the 'Only long positions are allowed' rule is set for the symbol",
            10042: "TRADE_RETCODE_SHORT_ONLY - The request is rejected because the 'Only short positions are allowed' rule is set for the symbol",
            10043: "TRADE_RETCODE_CLOSE_ONLY - The request is rejected because the 'Only position closing is allowed' rule is set for the symbol",
            10044: "TRADE_RETCODE_FIFO_CLOSE - The request is rejected because 'Position closing is allowed only by FIFO rule' is set for the trading account"
        }
        return error_codes.get(retcode, f"Unknown error code: {retcode}")
    
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
        """Get account information using thread pool."""
        try:
            loop = asyncio.get_event_loop()
            account_info = await loop.run_in_executor(self.executor, self.mt5.account_info)

            if account_info is None:
                raise ValueError("Failed to get account info")

            # Log account details for debugging
            logger.info(f"Account Info: Login={account_info.login}, Balance={account_info.balance}, "
                       f"Margin={account_info.margin}, Free Margin={account_info.margin_free}")

            return {
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'margin_free': account_info.margin_free
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {'balance': 0.0, 'equity': 0.0, 'margin': 0.0, 'margin_free': 0.0}

    async def check_trading_permissions(self) -> Dict:
        """Check account trading permissions and symbol availability."""
        try:
            loop = asyncio.get_event_loop()

            # Get account info
            account_info = await loop.run_in_executor(self.executor, self.mt5.account_info)

            # Get terminal info
            terminal_info = await loop.run_in_executor(self.executor, self.mt5.terminal_info)

            # Check some common symbols
            symbols_to_check = ['EURUSD', 'GBPUSD', 'USDJPY']
            symbol_status = {}

            for symbol in symbols_to_check:
                symbol_info = await loop.run_in_executor(self.executor, self.mt5.symbol_info, symbol)
                if symbol_info:
                    symbol_status[symbol] = {
                        'visible': symbol_info.visible,
                        'selected': symbol_info.select,
                        'spread': symbol_info.spread,
                        'volume_min': getattr(symbol_info, 'volume_min', 0.01),
                        'trading_enabled': symbol_info.visible and symbol_info.select
                    }
                else:
                    symbol_status[symbol] = {'error': 'Symbol not found'}

            return {
                'account_info': {
                    'login': account_info.login if account_info else None,
                    'balance': account_info.balance if account_info else 0,
                    'margin': account_info.margin if account_info else 0,
                    'margin_free': account_info.margin_free if account_info else 0
                },
                'terminal_info': {
                    'connected': terminal_info.connected if terminal_info else False,
                    'trade_allowed': terminal_info.trade_allowed if terminal_info else False,
                    'name': terminal_info.name if terminal_info else None
                },
                'symbol_status': symbol_status
            }
        except Exception as e:
            logger.error(f"Error checking trading permissions: {e}")
            return {'error': str(e)}

    async def get_positions(self) -> List[Dict]:
        """Get open positions using thread pool."""
        try:
            loop = asyncio.get_event_loop()
            positions = await loop.run_in_executor(self.executor, self.mt5.positions_get)

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
        """Place an order with enhanced error handling and slippage retries using thread pool, with detailed logging."""
        try:
            logger.info(
                f"Attempting to place order: symbol={symbol}, volume={volume}, type={order_type}, sl={stop_loss}, tp={take_profit}"
            )
            loop = asyncio.get_event_loop()

            # Get symbol info and tick in thread pool
            symbol_info = await loop.run_in_executor(self.executor, self.mt5.symbol_info, symbol)
            tick = await loop.run_in_executor(self.executor, self.mt5.symbol_info_tick, symbol)
            if symbol_info is None or tick is None:
                logger.error(f"Invalid symbol or tick: {symbol}")
                # Log additional debugging info
                logger.error(f"Symbol info: {symbol_info}")
                logger.error(f"Tick info: {tick}")
                logger.error(f"MT5 last error: {self.mt5.last_error()}")
                raise ValueError(f"Invalid symbol or tick: {symbol}")

            # Check if symbol is available for trading
            if not symbol_info.visible or not symbol_info.select:
                logger.error(f"Symbol {symbol} is not available for trading")
                logger.error(f"Symbol visible: {symbol_info.visible}, selected: {symbol_info.select}")
                raise ValueError(f"Symbol {symbol} is not available for trading")

            # Determine order type and price
            if order_type.upper() == 'BUY':
                price = tick.ask
                order_type_mt5 = self.mt5.ORDER_TYPE_BUY
            else:
                price = tick.bid
                order_type_mt5 = self.mt5.ORDER_TYPE_SELL

            # Ensure volume is above broker minimum
            min_lot = symbol_info.volume_min if hasattr(symbol_info, 'volume_min') else 0.01
            if volume < min_lot:
                logger.error(f"Volume {volume} is below broker minimum {min_lot} for {symbol}")
                raise ValueError(f"Volume {volume} is below broker minimum {min_lot}")

            # Try different filling types in order of preference
            # Note: This broker supports IOC but not FOK
            filling_types = [
                self.mt5.ORDER_FILLING_IOC,  # Immediate or Cancel (works with this broker)
                self.mt5.ORDER_FILLING_RETURN,  # Return remaining
                self.mt5.ORDER_FILLING_FOK  # Fill or Kill (not supported by this broker)
            ]

            order_dict = {
                'action': self.mt5.TRADE_ACTION_DEAL,
                'symbol': symbol,
                'volume': volume,
                'type': order_type_mt5,
                'price': price,
                'sl': stop_loss if stop_loss else 0.0,
                'tp': take_profit if take_profit else 0.0,
                'deviation': self.max_deviation,
                'magic': 0,
                'comment': "AI Bot",
                'type_time': self.mt5.ORDER_TIME_GTC,
                'type_filling': filling_types[0]  # Start with IOC (this broker doesn't support FOK)
            }

            for attempt in range(self.max_retries):
                logger.info(f"Order send attempt {attempt+1} for {symbol}: {order_dict}")

                # Use direct MT5 call (thread pool was causing issues)
                result = self.mt5.order_send(order_dict)
                logger.info(f"Order send result for {symbol}: {result}")

                if result is None:
                    last_error = self.mt5.last_error()
                    logger.error(f"Order send returned None for {symbol}. MT5 last_error: {last_error}")
                    raise ValueError("Order send failed")

                if result.retcode == self.mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Order placed successfully for {symbol}: ticket={result.order}")
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
                        # Refresh price in thread pool
                        fresh_tick = await loop.run_in_executor(self.executor, self.mt5.symbol_info_tick, symbol)
                        if fresh_tick:
                            order_dict['price'] = fresh_tick.ask if order_type.upper() == 'BUY' else fresh_tick.bid
                        continue
                    logger.error(f"Slippage retries exhausted for {symbol}")
                    return {
                        'status': OrderResult.REJECTED.value,
                        'ticket': None,
                        'retcode': result.retcode,
                        'comment': 'Slippage retries exhausted'
                    }
                else:
                    # Decode MT5 error codes for better debugging
                    error_description = self._decode_mt5_error(result.retcode)
                    logger.error(
                        f"Order failed for {symbol}: retcode={result.retcode} ({error_description}), comment={result.comment}"
                    )
                    return {
                        'status': OrderResult.FAILED.value,
                        'ticket': None,
                        'retcode': result.retcode,
                        'comment': f"{result.comment} ({error_description})"
                    }

        except Exception as e:
            logger.error(f"Exception during order placement for {symbol}: {e}")
            return {
                'status': OrderResult.FAILED.value,
                'ticket': None,
                'retcode': -1,
                'comment': str(e)
            }
    
    async def validate_spread(self, symbol: str) -> bool:
        """Validate spread for the symbol using thread pool."""
        try:
            loop = asyncio.get_event_loop()
            symbol_info = await loop.run_in_executor(self.executor, self.mt5.symbol_info, symbol)
            if symbol_info is None or symbol_info.spread > self.max_spread_pips:
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating spread for {symbol}: {e}")
            return False

    async def get_spread_pips(self, symbol: str) -> Optional[Dict]:
        """Get current spread and pip value in pips for symbol using thread pool."""
        try:
            loop = asyncio.get_event_loop()
            symbol_info = await loop.run_in_executor(self.executor, self.mt5.symbol_info, symbol)
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
        """Close a position using thread pool."""
        try:
            loop = asyncio.get_event_loop()

            # Get position info in thread pool
            positions = await loop.run_in_executor(self.executor, self.mt5.positions_get, ticket)
            if not positions:
                raise ValueError(f"Position {ticket} not found")

            position = positions[0]

            # Get symbol info for closing price
            symbol_info = await loop.run_in_executor(self.executor, self.mt5.symbol_info, position.symbol)
            if symbol_info is None:
                raise ValueError(f"Symbol info not available for {position.symbol}")

            order_dict = {
                'action': self.mt5.TRADE_ACTION_DEAL,
                'position': ticket,
                'symbol': position.symbol,
                'volume': position.volume,
                'type': self.mt5.ORDER_TYPE_SELL if position.type == self.mt5.ORDER_TYPE_BUY else self.mt5.ORDER_TYPE_BUY,
                'price': symbol_info.bid if position.type == self.mt5.ORDER_TYPE_BUY else symbol_info.ask,
                'type_time': self.mt5.ORDER_TIME_GTC,
                'type_filling': self.mt5.ORDER_FILLING_IOC
            }

            # Send close order in thread pool
            result = await loop.run_in_executor(self.executor, self.mt5.order_send, order_dict)
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
            # Fallback to enhanced interface (old MT5BrokerInterface removed)
            self.broker = EnhancedMT5BrokerInterface()
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

    async def check_trading_permissions(self) -> Dict:
        """Check account trading permissions and symbol availability."""
        return await self.broker.check_trading_permissions()
