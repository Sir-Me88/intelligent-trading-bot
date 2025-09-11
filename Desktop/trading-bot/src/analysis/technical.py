#!/usr/bin/env python3
"""Technical analysis indicators and pattern recognition."""
import logging
from typing import Dict, List, Optional
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Candlestick pattern types."""
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"


class SignalDirection(Enum):
    NONE = "none"
    BUY = "buy"
    SELL = "sell"


class TechnicalAnalyzer:
    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period

        # SuperTrend parameters
        self.supertrend_period = 10
        self.supertrend_multiplier = 3.0

        # Ichimoku parameters
        self.ichimoku_tenkan_period = 9
        self.ichimoku_kijun_period = 26
        self.ichimoku_senkou_span_b_period = 52
        self.ichimoku_displacement = 26

    def calculate_atr(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Calculate ATR series (safe manual implementation)."""
        try:
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(self.atr_period, min_periods=1).mean()
            return atr
        except Exception as e:
            logger.debug("ATR calculation failed: %s", e)
            return None

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Return latest ATR scalar (used by tests)."""
        atr_series = self.calculate_atr(df)
        if atr_series is None or len(atr_series) == 0 or pd.isna(atr_series.iloc[-1]):
            return 0.0
        return float(atr_series.iloc[-1])

    def _calculate_support_resistance(self, df: pd.DataFrame, lookback: int = 100) -> Dict[str, List[float]]:
        """Return support < resistance as lists (test-friendly)."""
        highs = df['high'].tail(lookback).astype(float)
        lows = df['low'].tail(lookback).astype(float)
        support_level = float(lows.quantile(0.1))
        resistance_level = float(highs.quantile(0.9))
        if support_level >= resistance_level:
            support_level = float(lows.min())
            resistance_level = float(highs.max())
            if support_level >= resistance_level:
                resistance_level = support_level + abs(support_level) * 0.01 + 1e-6
        return {"support": [support_level], "resistance": [resistance_level]}

    def _detect_bullish_engulfing(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect simple bullish engulfing on last two candles, return dict when found."""
        try:
            if len(df) < 2:
                return None
            prev = df.iloc[-2]
            last = df.iloc[-1]

            prev_open = float(prev['open'])
            prev_close = float(prev['close'])
            last_open = float(last['open'])
            last_close = float(last['close'])

            if (prev_close < prev_open) and (last_close > last_open) and (last_close - last_open > prev_open - prev_close):
                prev_body = abs(prev_close - prev_open)
                last_body = abs(last_close - last_open)
                confidence = min(1.0, last_body / (prev_body + 1e-9))
                return {
                    "pattern": PatternType.BULLISH_ENGULFING,
                    "confidence": float(confidence),
                    "entry_price": float(last_close),
                    "stop_loss": float(min(prev['low'], last['low'])),
                    "take_profit": float(last_close + (last_body * 3))
                }
        except Exception:
            logger.debug("Bullish engulfing detection error", exc_info=True)
        return None

    def _detect_bearish_engulfing(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect simple bearish engulfing on last two candles, return dict when found."""
        try:
            if len(df) < 2:
                return None
            prev = df.iloc[-2]
            last = df.iloc[-1]

            prev_open = float(prev['open'])
            prev_close = float(prev['close'])
            last_open = float(last['open'])
            last_close = float(last['close'])

            if (prev_close > prev_open) and (last_close < last_open) and (last_open - last_close > prev_close - prev_open):
                prev_body = abs(prev_close - prev_open)
                last_body = abs(last_close - last_open)
                confidence = min(1.0, last_body / (prev_body + 1e-9))
                return {
                    "pattern": PatternType.BEARISH_ENGULFING,
                    "confidence": float(confidence),
                    "entry_price": float(last_close),
                    "stop_loss": float(max(prev['high'], last['high'])),
                    "take_profit": float(last_close - (last_body * 3))
                }
        except Exception:
            logger.debug("Bearish engulfing detection error", exc_info=True)
        return None

    def _check_confluence(self, price: float, levels: Dict[str, List[float]], threshold: float = 0.0005) -> bool:
        """Check if price is within threshold of any support/resistance level."""
        for lvl in levels.get('support', []) + levels.get('resistance', []):
            try:
                if abs(price - float(lvl)) <= threshold:
                    return True
            except Exception:
                continue
        return False

    def _calculate_stop_loss(self, entry_price, direction, atr):
        """Calculate stop loss based on ATR and direction (test-friendly)."""
        try:
            if str(direction).lower() in ("buy", "long"):
                return float(entry_price) - float(atr) * 2.0
            return float(entry_price) + float(atr) * 2.0
        except Exception:
            return float(entry_price)

    def calculate_supertrend(self, df: pd.DataFrame, period: int = None, multiplier: float = None) -> pd.DataFrame:
        """Calculate SuperTrend indicator."""
        try:
            if period is None:
                period = self.supertrend_period
            if multiplier is None:
                multiplier = self.supertrend_multiplier

            if len(df) < period:
                return df

            # Calculate ATR
            atr = self.calculate_atr(df)
            if atr is None:
                return df

            # Calculate basic upper and lower bands
            hl2 = (df['high'] + df['low']) / 2
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)

            # Initialize SuperTrend
            supertrend = pd.Series(index=df.index, dtype=float)
            trend = pd.Series(index=df.index, dtype=int)

            # First value
            if df['close'].iloc[period-1] <= upper_band.iloc[period-1]:
                supertrend.iloc[period-1] = upper_band.iloc[period-1]
                trend.iloc[period-1] = 1  # Uptrend
            else:
                supertrend.iloc[period-1] = lower_band.iloc[period-1]
                trend.iloc[period-1] = -1  # Downtrend

            # Calculate SuperTrend for remaining values
            for i in range(period, len(df)):
                curr_close = df['close'].iloc[i]
                prev_supertrend = supertrend.iloc[i-1]
                prev_trend = trend.iloc[i-1]

                # Calculate new bands
                curr_upper = upper_band.iloc[i]
                curr_lower = lower_band.iloc[i]

                if prev_trend == 1:  # Previous uptrend
                    if curr_close > prev_supertrend:
                        supertrend.iloc[i] = min(curr_upper, prev_supertrend)
                        trend.iloc[i] = 1
                    else:
                        supertrend.iloc[i] = curr_upper
                        trend.iloc[i] = -1
                else:  # Previous downtrend
                    if curr_close < prev_supertrend:
                        supertrend.iloc[i] = max(curr_lower, prev_supertrend)
                        trend.iloc[i] = -1
                    else:
                        supertrend.iloc[i] = curr_lower
                        trend.iloc[i] = 1

            # Add to dataframe
            df = df.copy()
            df['supertrend'] = supertrend
            df['supertrend_trend'] = trend

            return df

        except Exception as e:
            logger.error(f"Error calculating SuperTrend: {e}")
            return df

    def calculate_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud indicator."""
        try:
            if len(df) < self.ichimoku_senkou_span_b_period:
                return df

            # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
            tenkan_high = df['high'].rolling(window=self.ichimoku_tenkan_period).max()
            tenkan_low = df['low'].rolling(window=self.ichimoku_tenkan_period).min()
            tenkan_sen = (tenkan_high + tenkan_low) / 2

            # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
            kijun_high = df['high'].rolling(window=self.ichimoku_kijun_period).max()
            kijun_low = df['low'].rolling(window=self.ichimoku_kijun_period).min()
            kijun_sen = (kijun_high + kijun_low) / 2

            # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, shifted forward 26 periods
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(self.ichimoku_displacement)

            # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted forward 26 periods
            senkou_high = df['high'].rolling(window=self.ichimoku_senkou_span_b_period).max()
            senkou_low = df['low'].rolling(window=self.ichimoku_senkou_span_b_period).min()
            senkou_span_b = ((senkou_high + senkou_low) / 2).shift(self.ichimoku_displacement)

            # Chikou Span (Lagging Span): Close shifted backward 26 periods
            chikou_span = df['close'].shift(-self.ichimoku_displacement)

            # Add to dataframe
            df = df.copy()
            df['tenkan_sen'] = tenkan_sen
            df['kijun_sen'] = kijun_sen
            df['senkou_span_a'] = senkou_span_a
            df['senkou_span_b'] = senkou_span_b
            df['chikou_span'] = chikou_span

            return df

        except Exception as e:
            logger.error(f"Error calculating Ichimoku: {e}")
            return df

    def generate_supertrend_signal(self, df: pd.DataFrame) -> Dict:
        """Generate signal based on SuperTrend indicator."""
        try:
            if 'supertrend_trend' not in df.columns or len(df) < 2:
                return {"direction": SignalDirection.NONE, "confidence": 0.0}

            current_trend = df['supertrend_trend'].iloc[-1]
            prev_trend = df['supertrend_trend'].iloc[-2]

            # Trend change detection
            if current_trend != prev_trend:
                if current_trend == 1:  # Bullish trend
                    confidence = 0.7
                    direction = SignalDirection.BUY
                else:  # Bearish trend
                    confidence = 0.7
                    direction = SignalDirection.SELL
            else:
                # Continuation signals
                if current_trend == 1:
                    confidence = 0.5
                    direction = SignalDirection.BUY
                else:
                    confidence = 0.5
                    direction = SignalDirection.SELL

            return {
                "direction": direction,
                "confidence": confidence,
                "indicator": "SuperTrend",
                "trend": "bullish" if current_trend == 1 else "bearish"
            }

        except Exception as e:
            logger.error(f"Error generating SuperTrend signal: {e}")
            return {"direction": SignalDirection.NONE, "confidence": 0.0}

    def generate_ichimoku_signal(self, df: pd.DataFrame) -> Dict:
        """Generate signal based on Ichimoku Cloud indicator."""
        try:
            required_cols = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']
            if not all(col in df.columns for col in required_cols) or len(df) < 2:
                return {"direction": SignalDirection.NONE, "confidence": 0.0}

            current = df.iloc[-1]
            prev = df.iloc[-2]

            signals = []
            confidence = 0.0

            # Tenkan-sen vs Kijun-sen cross
            if prev['tenkan_sen'] <= prev['kijun_sen'] and current['tenkan_sen'] > current['kijun_sen']:
                signals.append("TK_BULLISH_CROSS")
                confidence += 0.3
            elif prev['tenkan_sen'] >= prev['kijun_sen'] and current['tenkan_sen'] < current['kijun_sen']:
                signals.append("TK_BEARISH_CROSS")
                confidence += 0.3

            # Price vs Cloud
            current_price = current['close']
            cloud_top = max(current['senkou_span_a'], current['senkou_span_b'])
            cloud_bottom = min(current['senkou_span_a'], current['senkou_span_b'])

            if current_price > cloud_top:
                signals.append("ABOVE_CLOUD")
                confidence += 0.2
            elif current_price < cloud_bottom:
                signals.append("BELOW_CLOUD")
                confidence += 0.2

            # Chikou Span vs Price
            if current['chikou_span'] > current_price:
                signals.append("CHIKOU_BULLISH")
                confidence += 0.2
            elif current['chikou_span'] < current_price:
                signals.append("CHIKOU_BEARISH")
                confidence += 0.2

            # Determine direction
            bullish_signals = [s for s in signals if 'BULLISH' in s or s == 'ABOVE_CLOUD']
            bearish_signals = [s for s in signals if 'BEARISH' in s or s == 'BELOW_CLOUD']

            if len(bullish_signals) > len(bearish_signals):
                direction = SignalDirection.BUY
            elif len(bearish_signals) > len(bullish_signals):
                direction = SignalDirection.SELL
            else:
                direction = SignalDirection.NONE
                confidence = 0.0

            return {
                "direction": direction,
                "confidence": min(confidence, 0.9),
                "indicator": "Ichimoku",
                "signals": signals
            }

        except Exception as e:
            logger.error(f"Error generating Ichimoku signal: {e}")
            return {"direction": SignalDirection.NONE, "confidence": 0.0}

    def generate_ensemble_signal(self, df_recent: pd.DataFrame, df_full: pd.DataFrame,
                               adaptive_params: Dict = None) -> Dict:
        """Generate ensemble signal combining SuperTrend and Ichimoku."""
        try:
            # Calculate indicators
            df_with_supertrend = self.calculate_supertrend(df_full.copy())
            df_with_ichimoku = self.calculate_ichimoku(df_full.copy())

            # Generate individual signals
            supertrend_signal = self.generate_supertrend_signal(df_with_supertrend)
            ichimoku_signal = self.generate_ichimoku_signal(df_with_ichimoku)

            # Ensemble logic
            signals = []
            total_confidence = 0.0
            direction_votes = {"BUY": 0, "SELL": 0}

            # Add SuperTrend signal
            if supertrend_signal["direction"] != SignalDirection.NONE:
                signals.append(supertrend_signal)
                total_confidence += supertrend_signal["confidence"]
                direction_votes[supertrend_signal["direction"].value.upper()] += 1

            # Add Ichimoku signal
            if ichimoku_signal["direction"] != SignalDirection.NONE:
                signals.append(ichimoku_signal)
                total_confidence += ichimoku_signal["confidence"]
                direction_votes[ichimoku_signal["direction"].value.upper()] += 1

            # Determine ensemble direction
            if direction_votes["BUY"] > direction_votes["SELL"]:
                ensemble_direction = SignalDirection.BUY
            elif direction_votes["SELL"] > direction_votes["BUY"]:
                ensemble_direction = SignalDirection.SELL
            else:
                ensemble_direction = SignalDirection.NONE

            # Calculate ensemble confidence
            if len(signals) > 0:
                ensemble_confidence = total_confidence / len(signals)
                # Boost confidence if both indicators agree
                if direction_votes["BUY"] == 2 or direction_votes["SELL"] == 2:
                    ensemble_confidence = min(ensemble_confidence * 1.2, 0.95)
            else:
                ensemble_confidence = 0.0

            # Generate entry/exit levels
            if ensemble_direction != SignalDirection.NONE and len(df_recent) > 0:
                current_price = df_recent['close'].iloc[-1]
                atr_value = self._calculate_atr(df_recent)

                if ensemble_direction == SignalDirection.BUY:
                    entry_price = current_price
                    stop_loss = current_price - (atr_value * 2.0)
                    take_profit = current_price + (atr_value * 3.0)
                else:  # SELL
                    entry_price = current_price
                    stop_loss = current_price + (atr_value * 2.0)
                    take_profit = current_price - (atr_value * 3.0)

                return {
                    "direction": ensemble_direction,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "confidence": ensemble_confidence,
                    "indicator": "SuperTrend/Ichimoku Ensemble",
                    "signals": signals,
                    "ensemble_strength": len(signals)
                }

            return {"direction": SignalDirection.NONE, "confidence": 0.0}

        except Exception as e:
            logger.error(f"Error generating ensemble signal: {e}")
            return {"direction": SignalDirection.NONE, "confidence": 0.0}

    def generate_signal(self, df_recent: pd.DataFrame, df_full: pd.DataFrame,
                       adaptive_params: Dict = None, pair: str = None,
                       correlation_analyzer=None, economic_calendar_filter=None) -> Dict:
        """Enhanced signal generation with SuperTrend/Ichimoku ensemble."""
        try:
            # First try ensemble signal
            ensemble_signal = self.generate_ensemble_signal(df_recent, df_full, adaptive_params)

            if ensemble_signal["direction"] != SignalDirection.NONE and ensemble_signal["confidence"] >= 0.6:
                logger.info(f"🎯 ENSEMBLE SIGNAL: {ensemble_signal['direction'].value} "
                           f"(confidence: {ensemble_signal['confidence']:.2f})")
                return ensemble_signal

            # Fallback to traditional engulfing patterns
            logger.debug("Ensemble signal weak, falling back to engulfing patterns")
            pattern = self._detect_bullish_engulfing(df_recent)
            if pattern and isinstance(pattern, dict) and pattern.get("pattern") == PatternType.BULLISH_ENGULFING:
                return {
                    "direction": SignalDirection.BUY,
                    "entry_price": pattern.get("entry_price"),
                    "stop_loss": pattern.get("stop_loss"),
                    "take_profit": pattern.get("take_profit"),
                    "confidence": pattern.get("confidence", 1.0)
                }

            pattern = self._detect_bearish_engulfing(df_recent)
            if pattern and isinstance(pattern, dict) and pattern.get("pattern") == PatternType.BEARISH_ENGULFING:
                return {
                    "direction": SignalDirection.SELL,
                    "entry_price": pattern.get("entry_price"),
                    "stop_loss": pattern.get("stop_loss"),
                    "take_profit": pattern.get("take_profit"),
                    "confidence": pattern.get("confidence", 1.0)
                }

            return {"direction": SignalDirection.NONE}

        except Exception as e:
            logger.error(f"Error in enhanced signal generation: {e}")
            return {"direction": SignalDirection.NONE}
