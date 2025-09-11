#!/usr/bin/env python3
"""Advanced Trend Reversal Detection System."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

# LSTM Dependencies (with fallback)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import MinMaxScaler
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available - LSTM features disabled")
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class TrendReversalDetector:
    """Advanced trend reversal detection using multiple indicators."""
    
    def __init__(self):
        # Reversal detection parameters
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.macd_divergence_threshold = 0.0001
        self.volume_spike_threshold = 1.5
        self.price_action_threshold = 0.0005
        
        # Multi-timeframe analysis
        self.timeframes = ['M1', 'M5', 'M15', 'H1']
        self.reversal_confidence_threshold = 0.75
        
        # Chandelier Exit parameters
        self.chandelier_period = 22
        self.chandelier_multiplier = 3.0
        
        # Historical data for pattern recognition
        self.reversal_patterns = []

        # LSTM Components for volatility prediction
        self.lstm_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
        self.scaler = None
        
    def detect_trend_reversal(self, symbol: str, current_data: Dict) -> Dict:
        """Detect potential trend reversal with confidence score."""
        try:
            reversal_signals = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'reversal_detected': False,
                'confidence': 0.0,
                'direction': 'NONE',  # BULLISH_REVERSAL, BEARISH_REVERSAL
                'signals': [],
                'immediate_action': 'HOLD'  # HOLD, EXIT_LONG, EXIT_SHORT
            }
            
            # Get multi-timeframe data
            signals = []
            
            # 1. RSI Divergence Analysis
            rsi_signal = self._analyze_rsi_divergence(current_data)
            if rsi_signal['strength'] > 0:
                signals.append(rsi_signal)
                
            # 2. MACD Divergence
            macd_signal = self._analyze_macd_divergence(current_data)
            if macd_signal['strength'] > 0:
                signals.append(macd_signal)
                
            # 3. Price Action Patterns
            price_action_signal = self._analyze_price_action_reversal(current_data)
            if price_action_signal['strength'] > 0:
                signals.append(price_action_signal)
                
            # 4. Volume Analysis
            volume_signal = self._analyze_volume_reversal(current_data)
            if volume_signal['strength'] > 0:
                signals.append(volume_signal)
                
            # 5. Support/Resistance Breaks
            sr_signal = self._analyze_support_resistance_break(current_data)
            if sr_signal['strength'] > 0:
                signals.append(sr_signal)
                
            # 6. Multi-timeframe Confirmation
            mtf_signal = self._analyze_multi_timeframe_reversal(current_data)
            if mtf_signal['strength'] > 0:
                signals.append(mtf_signal)
                
            # Calculate overall confidence
            if signals:
                total_strength = sum(signal['strength'] for signal in signals)
                weighted_confidence = total_strength / len(signals)
                
                reversal_signals['confidence'] = min(weighted_confidence, 1.0)
                reversal_signals['signals'] = signals
                
                # Determine reversal direction
                bullish_signals = [s for s in signals if s['direction'] == 'BULLISH']
                bearish_signals = [s for s in signals if s['direction'] == 'BEARISH']
                
                if len(bullish_signals) > len(bearish_signals):
                    reversal_signals['direction'] = 'BULLISH_REVERSAL'
                elif len(bearish_signals) > len(bullish_signals):
                    reversal_signals['direction'] = 'BEARISH_REVERSAL'
                    
                # Determine if reversal is strong enough
                if reversal_signals['confidence'] >= self.reversal_confidence_threshold:
                    reversal_signals['reversal_detected'] = True
                    
                    # Determine immediate action based on current positions
                    current_position = current_data.get('current_position', 'NONE')
                    if current_position == 'LONG' and reversal_signals['direction'] == 'BEARISH_REVERSAL':
                        reversal_signals['immediate_action'] = 'EXIT_LONG'
                    elif current_position == 'SHORT' and reversal_signals['direction'] == 'BULLISH_REVERSAL':
                        reversal_signals['immediate_action'] = 'EXIT_SHORT'
                        
            logger.info(f"Reversal Analysis {symbol}: Confidence {reversal_signals['confidence']:.2f}, "
                       f"Direction: {reversal_signals['direction']}, Action: {reversal_signals['immediate_action']}")
                       
            return reversal_signals
            
        except Exception as e:
            logger.error(f"Error in trend reversal detection for {symbol}: {e}")
            return {'reversal_detected': False, 'confidence': 0.0, 'error': str(e)}
            
    def _analyze_rsi_divergence(self, data: Dict) -> Dict:
        """Analyze RSI divergence patterns."""
        try:
            df = data.get('df_15m')
            if df is None or len(df) < 20:
                return {'strength': 0, 'type': 'RSI', 'direction': 'NONE'}
                
            # Calculate RSI
            rsi = self._calculate_rsi(df['close'], period=14)
            current_rsi = rsi.iloc[-1]
            
            # Check for overbought/oversold conditions
            if current_rsi > self.rsi_overbought:
                # Look for bearish divergence
                price_highs = df['high'].rolling(window=5).max()
                rsi_highs = rsi.rolling(window=5).max()
                
                if price_highs.iloc[-1] > price_highs.iloc[-6] and rsi_highs.iloc[-1] < rsi_highs.iloc[-6]:
                    return {
                        'strength': 0.8,
                        'type': 'RSI_BEARISH_DIVERGENCE',
                        'direction': 'BEARISH',
                        'details': f'RSI: {current_rsi:.2f} (Overbought + Divergence)'
                    }
                    
            elif current_rsi < self.rsi_oversold:
                # Look for bullish divergence
                price_lows = df['low'].rolling(window=5).min()
                rsi_lows = rsi.rolling(window=5).min()
                
                if price_lows.iloc[-1] < price_lows.iloc[-6] and rsi_lows.iloc[-1] > rsi_lows.iloc[-6]:
                    return {
                        'strength': 0.8,
                        'type': 'RSI_BULLISH_DIVERGENCE',
                        'direction': 'BULLISH',
                        'details': f'RSI: {current_rsi:.2f} (Oversold + Divergence)'
                    }
                    
            return {'strength': 0, 'type': 'RSI', 'direction': 'NONE'}
            
        except Exception as e:
            logger.error(f"Error in RSI divergence analysis: {e}")
            return {'strength': 0, 'type': 'RSI', 'direction': 'NONE'}
            
    def _analyze_macd_divergence(self, data: Dict) -> Dict:
        """Analyze MACD divergence patterns."""
        try:
            df = data.get('df_15m')
            if df is None or len(df) < 30:
                return {'strength': 0, 'type': 'MACD', 'direction': 'NONE'}
                
            # Calculate MACD
            macd_line, macd_signal, macd_histogram = self._calculate_macd(df['close'])
            
            # Look for MACD divergence
            recent_price_high = df['high'].rolling(window=10).max().iloc[-1]
            recent_price_low = df['low'].rolling(window=10).min().iloc[-1]
            recent_macd_high = macd_line.rolling(window=10).max().iloc[-1]
            recent_macd_low = macd_line.rolling(window=10).min().iloc[-1]
            
            # Check for bearish divergence (price higher highs, MACD lower highs)
            if (recent_price_high > df['high'].rolling(window=10).max().iloc[-11] and
                recent_macd_high < macd_line.rolling(window=10).max().iloc[-11]):
                return {
                    'strength': 0.7,
                    'type': 'MACD_BEARISH_DIVERGENCE',
                    'direction': 'BEARISH',
                    'details': f'MACD: {macd_line.iloc[-1]:.5f} (Bearish Divergence)'
                }
                
            # Check for bullish divergence (price lower lows, MACD higher lows)
            if (recent_price_low < df['low'].rolling(window=10).min().iloc[-11] and
                recent_macd_low > macd_line.rolling(window=10).min().iloc[-11]):
                return {
                    'strength': 0.7,
                    'type': 'MACD_BULLISH_DIVERGENCE',
                    'direction': 'BULLISH',
                    'details': f'MACD: {macd_line.iloc[-1]:.5f} (Bullish Divergence)'
                }
                
            return {'strength': 0, 'type': 'MACD', 'direction': 'NONE'}
            
        except Exception as e:
            logger.error(f"Error in MACD divergence analysis: {e}")
            return {'strength': 0, 'type': 'MACD', 'direction': 'NONE'}
            
    def _analyze_price_action_reversal(self, data: Dict) -> Dict:
        """Analyze price action reversal patterns."""
        try:
            df = data.get('df_15m')
            if df is None or len(df) < 10:
                return {'strength': 0, 'type': 'PRICE_ACTION', 'direction': 'NONE'}
                
            # Look for reversal candlestick patterns
            recent_candles = df.tail(5)
            
            # Doji pattern detection
            for i in range(len(recent_candles)):
                candle = recent_candles.iloc[i]
                body_size = abs(candle['close'] - candle['open'])
                total_range = candle['high'] - candle['low']
                
                if total_range > 0 and body_size / total_range < 0.1:  # Doji
                    if i == len(recent_candles) - 1:  # Most recent candle
                        return {
                            'strength': 0.6,
                            'type': 'DOJI_REVERSAL',
                            'direction': 'REVERSAL',
                            'details': 'Doji pattern detected - potential reversal'
                        }
                        
            # Hammer/Shooting star patterns
            last_candle = recent_candles.iloc[-1]
            body_size = abs(last_candle['close'] - last_candle['open'])
            upper_shadow = last_candle['high'] - max(last_candle['open'], last_candle['close'])
            lower_shadow = min(last_candle['open'], last_candle['close']) - last_candle['low']
            
            # Hammer pattern (bullish reversal)
            if lower_shadow > 2 * body_size and upper_shadow < body_size:
                return {
                    'strength': 0.7,
                    'type': 'HAMMER_PATTERN',
                    'direction': 'BULLISH',
                    'details': 'Hammer pattern - bullish reversal signal'
                }
                
            # Shooting star pattern (bearish reversal)
            if upper_shadow > 2 * body_size and lower_shadow < body_size:
                return {
                    'strength': 0.7,
                    'type': 'SHOOTING_STAR_PATTERN',
                    'direction': 'BEARISH',
                    'details': 'Shooting star pattern - bearish reversal signal'
                }
                
            return {'strength': 0, 'type': 'PRICE_ACTION', 'direction': 'NONE'}
            
        except Exception as e:
            logger.error(f"Error in price action analysis: {e}")
            return {'strength': 0, 'type': 'PRICE_ACTION', 'direction': 'NONE'}
            
    def _analyze_volume_reversal(self, data: Dict) -> Dict:
        """Analyze volume-based reversal signals."""
        try:
            df = data.get('df_15m')
            if df is None or len(df) < 20:
                return {'strength': 0, 'type': 'VOLUME', 'direction': 'NONE'}
            
            # Volume fallback: return 0 if volume column missing
            if 'volume' not in df.columns:
                return {'strength': 0, 'type': 'VOLUME', 'direction': 'NONE'}
                
            # Calculate average volume
            avg_volume = df['volume'].rolling(window=20).mean()
            current_volume = df['volume'].iloc[-1]
            
            # Volume spike detection
            if current_volume > avg_volume.iloc[-1] * self.volume_spike_threshold:
                # Determine direction based on price movement
                price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
                
                if abs(price_change) > self.price_action_threshold:
                    direction = 'BULLISH' if price_change > 0 else 'BEARISH'
                    return {
                        'strength': 0.6,
                        'type': 'VOLUME_SPIKE',
                        'direction': direction,
                        'details': f'Volume spike: {current_volume/avg_volume.iloc[-1]:.2f}x average'
                    }
                    
            return {'strength': 0, 'type': 'VOLUME', 'direction': 'NONE'}
            
        except Exception as e:
            logger.error(f"Error in volume analysis: {e}")
            return {'strength': 0, 'type': 'VOLUME', 'direction': 'NONE'}
            
    def _analyze_support_resistance_break(self, data: Dict) -> Dict:
        """Analyze support/resistance level breaks."""
        try:
            df = data.get('df_15m')
            if df is None or len(df) < 50:
                return {'strength': 0, 'type': 'SUPPORT_RESISTANCE', 'direction': 'NONE'}
                
            current_price = df['close'].iloc[-1]
            
            # Calculate support and resistance levels
            recent_highs = df['high'].rolling(window=20).max()
            recent_lows = df['low'].rolling(window=20).min()
            
            resistance_level = recent_highs.iloc[-1]
            support_level = recent_lows.iloc[-1]
            
            # Check for resistance break (bullish)
            if current_price > resistance_level * 1.001:  # 0.1% buffer
                return {
                    'strength': 0.8,
                    'type': 'RESISTANCE_BREAK',
                    'direction': 'BULLISH',
                    'details': f'Price broke resistance at {resistance_level:.5f}'
                }
                
            # Check for support break (bearish)
            if current_price < support_level * 0.999:  # 0.1% buffer
                return {
                    'strength': 0.8,
                    'type': 'SUPPORT_BREAK',
                    'direction': 'BEARISH',
                    'details': f'Price broke support at {support_level:.5f}'
                }
                
            return {'strength': 0, 'type': 'SUPPORT_RESISTANCE', 'direction': 'NONE'}
            
        except Exception as e:
            logger.error(f"Error in support/resistance analysis: {e}")
            return {'strength': 0, 'type': 'SUPPORT_RESISTANCE', 'direction': 'NONE'}
            
    def _analyze_multi_timeframe_reversal(self, data: Dict) -> Dict:
        """Analyze reversal signals across multiple timeframes."""
        try:
            df_15m = data.get('df_15m')
            df_1h = data.get('df_1h')
            
            if df_15m is None or df_1h is None:
                return {'strength': 0, 'type': 'MULTI_TIMEFRAME', 'direction': 'NONE'}
            
            reversal_signals = []
            
            # Analyze 15M timeframe
            if len(df_15m) >= 20:
                rsi_15m = self._calculate_rsi(df_15m['close']).iloc[-1]
                if rsi_15m > 75 or rsi_15m < 25:  # Strong overbought/oversold
                    direction = 'BEARISH' if rsi_15m > 75 else 'BULLISH'
                    reversal_signals.append({'tf': '15M', 'strength': 0.6, 'direction': direction})
            
            # Analyze 1H timeframe
            if len(df_1h) >= 20:
                rsi_1h = self._calculate_rsi(df_1h['close']).iloc[-1]
                if rsi_1h > 70 or rsi_1h < 30:  # Overbought/oversold
                    direction = 'BEARISH' if rsi_1h > 70 else 'BULLISH'
                    reversal_signals.append({'tf': '1H', 'strength': 0.8, 'direction': direction})
            
            if not reversal_signals:
                return {'strength': 0, 'type': 'MULTI_TIMEFRAME', 'direction': 'NONE'}
            
            # Calculate weighted strength
            total_strength = sum(signal['strength'] for signal in reversal_signals)
            avg_strength = total_strength / len(reversal_signals)
            
            # Determine dominant direction
            bullish_count = sum(1 for s in reversal_signals if s['direction'] == 'BULLISH')
            bearish_count = sum(1 for s in reversal_signals if s['direction'] == 'BEARISH')
            
            if bullish_count > bearish_count:
                direction = 'BULLISH'
            elif bearish_count > bullish_count:
                direction = 'BEARISH'
            else:
                direction = 'NONE'
            
            return {
                'strength': avg_strength,
                'type': 'MULTI_TIMEFRAME',
                'direction': direction,
                'details': f'Signals: {len(reversal_signals)}, Strength: {avg_strength:.2f}'
            }
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis: {e}")
            return {'strength': 0, 'type': 'MULTI_TIMEFRAME', 'direction': 'NONE'}
            
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
    
    def calculate_chandelier_exit(self, df: pd.DataFrame, position_type: str = 'LONG') -> Dict:
        """Calculate Chandelier Exit levels for dynamic trailing stops."""
        try:
            if df is None or len(df) < self.chandelier_period:
                return {'chandelier_exit': None, 'atr': None, 'signal': 'HOLD'}
            
            # Calculate ATR (Average True Range)
            atr = self._calculate_atr(df, period=self.chandelier_period)
            current_atr = atr.iloc[-1]
            
            if position_type.upper() == 'LONG':
                # For long positions: Chandelier Exit = Highest High - (ATR * Multiplier)
                highest_high = df['high'].rolling(window=self.chandelier_period).max().iloc[-1]
                chandelier_exit = highest_high - (current_atr * self.chandelier_multiplier)
                current_price = df['close'].iloc[-1]
                
                # Signal to exit if price closes below Chandelier Exit
                signal = 'EXIT_LONG' if current_price < chandelier_exit else 'HOLD'
                
            else:  # SHORT position
                # For short positions: Chandelier Exit = Lowest Low + (ATR * Multiplier)
                lowest_low = df['low'].rolling(window=self.chandelier_period).min().iloc[-1]
                chandelier_exit = lowest_low + (current_atr * self.chandelier_multiplier)
                current_price = df['close'].iloc[-1]
                
                # Signal to exit if price closes above Chandelier Exit
                signal = 'EXIT_SHORT' if current_price > chandelier_exit else 'HOLD'
            
            return {
                'chandelier_exit': chandelier_exit,
                'atr': current_atr,
                'signal': signal,
                'current_price': current_price,
                'distance_pips': abs(current_price - chandelier_exit) * 10000,
                'strength': 0.9 if signal.startswith('EXIT') else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating Chandelier Exit: {e}")
            return {'chandelier_exit': None, 'atr': None, 'signal': 'HOLD', 'strength': 0.0}
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        try:
            # True Range calculation
            high_low = df['high'] - df['low']
            high_close_prev = abs(df['high'] - df['close'].shift(1))
            low_close_prev = abs(df['low'] - df['close'].shift(1))
            
            true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series([0] * len(df), index=df.index)
    
    def detect_chandelier_exit_signal(self, symbol: str, current_data: Dict, position_type: str) -> Dict:
        """Detect Chandelier Exit signals for position management."""
        try:
            df_15m = current_data.get('df_15m')
            df_1h = current_data.get('df_1h')
            
            if df_15m is None:
                return {'chandelier_signal': 'HOLD', 'confidence': 0.0}
            
            # Calculate Chandelier Exit on 15M timeframe
            chandelier_15m = self.calculate_chandelier_exit(df_15m, position_type)
            
            # Calculate Chandelier Exit on 1H timeframe for confirmation
            chandelier_1h = None
            if df_1h is not None and len(df_1h) >= self.chandelier_period:
                chandelier_1h = self.calculate_chandelier_exit(df_1h, position_type)
            
            # Determine final signal
            signal_15m = chandelier_15m.get('signal', 'HOLD')
            signal_1h = chandelier_1h.get('signal', 'HOLD') if chandelier_1h else 'HOLD'
            
            # Multi-timeframe confirmation
            if signal_15m.startswith('EXIT') and signal_1h.startswith('EXIT'):
                confidence = 0.95  # High confidence with both timeframes confirming
                final_signal = signal_15m
            elif signal_15m.startswith('EXIT'):
                confidence = 0.75  # Medium confidence with only 15M signal
                final_signal = signal_15m
            else:
                confidence = 0.0
                final_signal = 'HOLD'
            
            return {
                'chandelier_signal': final_signal,
                'confidence': confidence,
                'chandelier_15m': chandelier_15m,
                'chandelier_1h': chandelier_1h,
                'details': {
                    'exit_level_15m': chandelier_15m.get('chandelier_exit'),
                    'exit_level_1h': chandelier_1h.get('chandelier_exit') if chandelier_1h else None,
                    'atr_15m': chandelier_15m.get('atr'),
                    'distance_pips_15m': chandelier_15m.get('distance_pips')
                }
            }
            
        except Exception as e:
            logger.error(f"Error in Chandelier Exit detection for {symbol}: {e}")
            return {'chandelier_signal': 'HOLD', 'confidence': 0.0, 'error': str(e)}

    # LSTM Methods for Volatility Prediction
    def train_vol_predictor(self, symbol: str, periods: int = 1000):
        """Train LSTM model for volatility prediction."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - skipping LSTM training")
            return

        try:
            logger.info(f"🧠 Training LSTM volatility predictor for {symbol}...")

            # Get historical data
            from ..data.market_data import MarketDataManager
            dm = MarketDataManager()
            await dm.initialize()
            df = await dm.get_candles(symbol, 'M15', periods)

            if df is None or len(df) < 100:
                logger.warning(f"Insufficient data for {symbol} LSTM training")
                return

            # Prepare features
            df['rsi'] = self._calculate_rsi(df['close'])
            features = df[['open', 'high', 'low', 'close', 'volume', 'rsi']].values
            vol_target = df['close'].pct_change().rolling(1).std().shift(-1).dropna().values

            # Normalize features
            self.scaler = MinMaxScaler()
            features_scaled = self.scaler.fit_transform(features[:-len(vol_target)])
            targets = vol_target.reshape(-1, 1)

            # Create sequences (20 time steps)
            X, y = [], []
            seq_length = 20
            for i in range(seq_length, len(features_scaled)):
                X.append(features_scaled[i-seq_length:i])
                y.append(targets[i])

            X, y = torch.tensor(np.array(X), dtype=torch.float32).to(self.device), \
                   torch.tensor(np.array(y), dtype=torch.float32).to(self.device)

            # Create LSTM model
            self.lstm_model = VolLSTM(input_size=6, hidden_size=64, num_layers=2).to(self.device)
            optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            # Training loop
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            for epoch in range(20):  # Quick training
                epoch_loss = 0
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    pred = self.lstm_model(batch_x)
                    loss = criterion(pred, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                if epoch % 5 == 0:
                    logger.info(f"LSTM Epoch {epoch}, Loss: {epoch_loss/len(loader):.6f}")

            logger.info(f"✅ LSTM volatility predictor trained for {symbol}")

        except Exception as e:
            logger.error(f"Error training LSTM for {symbol}: {e}")

    def predict_volatility(self, current_data: Dict) -> float:
        """Predict next volatility using trained LSTM model."""
        if not TORCH_AVAILABLE or self.lstm_model is None or self.scaler is None:
            # Fallback to ATR calculation
            df = current_data.get('df_15m')
            if df is not None and len(df) >= 10:
                return df['close'].pct_change().std()
            return 0.001

        try:
            df = current_data.get('df_15m')
            if df is None or len(df) < 20:
                return 0.001

            # Prepare input sequence
            df['rsi'] = self._calculate_rsi(df['close'])
            features = df[['open', 'high', 'low', 'close', 'volume', 'rsi']].tail(20).values
            features_scaled = self.scaler.transform(features)

            input_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                pred_vol = self.lstm_model(input_tensor).item()

            return max(pred_vol, 0.0001)  # Ensure positive volatility

        except Exception as e:
            logger.error(f"Error predicting volatility: {e}")
            # Fallback
            df = current_data.get('df_15m')
            if df is not None and len(df) >= 10:
                return df['close'].pct_change().std()
            return 0.001


class VolLSTM(nn.Module):
    """LSTM model for volatility prediction."""

    def __init__(self, input_size=6, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Predict next volatility

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])
