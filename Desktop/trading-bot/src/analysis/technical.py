def safe_get_sentiment_value(sentiment_data, key, default=0.0):
    if isinstance(sentiment_data, dict):
        return sentiment_data.get(key, default)
    return default
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
    def _get_dynamic_thresholds(self, df: pd.DataFrame, adaptive_params: Dict) -> (float, float):
        """Return (min_rr_ratio, min_confidence) dynamically based on volatility."""
        if adaptive_params is None:
            return 2.0, 0.75  # fallback defaults

        # Calculate volatility as std of close returns
        if df is not None and len(df) > 5:
            vol = df['close'].pct_change().std()
        else:
            vol = 0.001

        low = adaptive_params.get('volatility_threshold_low', 0.001)
        high = adaptive_params.get('volatility_threshold_high', 0.003)

        # BALANCED APPROACH: Conservative enough for profitability, relaxed enough for signals
        if vol < low:
            # Conservative reduction for low volatility - maintain quality
            min_rr = adaptive_params.get('min_rr_ratio', 2.0) * 0.95  # Require 95% of base RR
            min_conf = adaptive_params.get('min_confidence', 0.75) * 0.98  # Require 98% of base confidence
        elif vol > high:
            # Moderate increase for high volatility - safety first
            min_rr = adaptive_params.get('min_rr_ratio', 2.0) * 1.05  # Require 105% of base RR
            min_conf = adaptive_params.get('min_confidence', 0.75) * 1.02  # Require 102% of base confidence
        else:
            min_rr = adaptive_params.get('min_rr_ratio', 2.0)
            min_conf = adaptive_params.get('min_confidence', 0.75)

        logger.debug(f"Dynamic thresholds: vol={vol:.6f}, low={low:.6f}, high={high:.6f}, min_rr={min_rr:.2f}, min_conf={min_conf:.2f}")
        return float(min_rr), float(min_conf)
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

            # FIX: Handle NaN values properly by using the last valid trends
            trend_series = df['supertrend_trend'].dropna()
            if len(trend_series) < 2:
                return {"direction": SignalDirection.NONE, "confidence": 0.0}

            current_trend = trend_series.iloc[-1]
            prev_trend = trend_series.iloc[-2]

            # Ensure we have valid numeric values
            if pd.isna(current_trend) or pd.isna(prev_trend):
                return {"direction": SignalDirection.NONE, "confidence": 0.0}

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
                               adaptive_params: Dict = None, sentiment: dict = None) -> Dict:
        """Generate ensemble signal combining SuperTrend and Ichimoku."""
        try:
            if sentiment is None or not isinstance(sentiment, dict):
                sentiment = {}
            # Calculate indicators
            df_with_supertrend = self.calculate_supertrend(df_full.copy())
            df_with_ichimoku = self.calculate_ichimoku(df_full.copy())

            # Generate individual signals with error handling
            supertrend_signal = self.generate_supertrend_signal(df_with_supertrend)
            ichimoku_signal = self.generate_ichimoku_signal(df_with_ichimoku)

            # IMPROVED ENSEMBLE LOGIC: Accept signals from working indicators only
            signals = []
            total_confidence = 0.0
            direction_votes = {"BUY": 0, "SELL": 0}

            # Add SuperTrend signal if valid
            if supertrend_signal["direction"] != SignalDirection.NONE:
                signals.append(supertrend_signal)
                total_confidence += supertrend_signal["confidence"]
                direction_votes[supertrend_signal["direction"].value.upper()] += 1
                logger.debug(f"SuperTrend signal added: {supertrend_signal['direction'].value} (conf: {supertrend_signal['confidence']:.3f})")

            # Add Ichimoku signal if valid
            if ichimoku_signal["direction"] != SignalDirection.NONE:
                signals.append(ichimoku_signal)
                total_confidence += ichimoku_signal["confidence"]
                direction_votes[ichimoku_signal["direction"].value.upper()] += 1
                logger.debug(f"Ichimoku signal added: {ichimoku_signal['direction'].value} (conf: {ichimoku_signal['confidence']:.3f})")

            # If no signals work, return NONE (but this is now less likely due to fixes)
            if len(signals) == 0:
                logger.debug("No valid signals from any indicator")
                return {"direction": SignalDirection.NONE, "confidence": 0.0}

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
                # Add sentiment boost (e.g., bullish sentiment adds up to 0.1)
                bull_boost = safe_get_sentiment_value(sentiment, 'bullish', 0.0) * 0.1
                bear_boost = safe_get_sentiment_value(sentiment, 'bearish', 0.0) * 0.1
                if ensemble_direction == SignalDirection.BUY:
                    ensemble_confidence += bull_boost
                elif ensemble_direction == SignalDirection.SELL:
                    ensemble_confidence += bear_boost
                ensemble_confidence = min(ensemble_confidence, 0.99)
            else:
                ensemble_confidence = 0.0

            logger.debug(f"Ensemble result: direction={ensemble_direction.value if ensemble_direction != SignalDirection.NONE else 'NONE'}, confidence={ensemble_confidence:.3f}, signals={len(signals)}")

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
                       correlation_analyzer=None, economic_calendar_filter=None, sentiment: dict = None) -> Dict:
        """Enhanced signal generation with SuperTrend/Ichimoku ensemble."""
        try:
            if sentiment is None or not isinstance(sentiment, dict):
                sentiment = {}

            # FIX: Add comprehensive debug logging
            logger.debug(f"🔍 SIGNAL GENERATION DEBUG for {pair}")
            logger.debug(f"   Data shapes: df_recent={df_recent.shape if df_recent is not None else None}, df_full={df_full.shape if df_full is not None else None}")

            # Get dynamic thresholds
            min_rr_ratio, min_confidence = self._get_dynamic_thresholds(df_recent, adaptive_params)
            logger.debug(f"   Dynamic thresholds: min_rr={min_rr_ratio:.2f}, min_conf={min_confidence:.2f}")

            # First try ensemble signal
            ensemble_signal = self.generate_ensemble_signal(df_recent, df_full, adaptive_params, sentiment)
            logger.debug(f"   Ensemble result: direction={ensemble_signal.get('direction', 'NONE')}, confidence={ensemble_signal.get('confidence', 0):.3f}")

            # Calculate R/R ratio for ensemble
            rr_ratio = 0.0
            if ensemble_signal["direction"] != SignalDirection.NONE:
                entry = ensemble_signal.get("entry_price", 0)
                stop = ensemble_signal.get("stop_loss", 0)
                take = ensemble_signal.get("take_profit", 0)
                if entry and stop and take and ((entry > stop and take > entry) or (entry < stop and take < entry)):
                    risk = abs(entry - stop)
                    reward = abs(take - entry)
                    rr_ratio = reward / risk if risk > 0 else 0.0
                    logger.debug(f"   R/R calculation: entry={entry:.5f}, stop={stop:.5f}, take={take:.5f}, ratio={rr_ratio:.2f}")

            # Check threshold conditions
            confidence_ok = ensemble_signal["confidence"] >= min_confidence
            rr_ok = rr_ratio >= min_rr_ratio

            logger.debug(f"   Threshold checks: confidence_ok={confidence_ok} ({ensemble_signal['confidence']:.3f} >= {min_confidence:.3f}), rr_ok={rr_ok} ({rr_ratio:.2f} >= {min_rr_ratio:.2f})")

            if (
                ensemble_signal["direction"] != SignalDirection.NONE
                and confidence_ok
                and rr_ok
            ):
                logger.info(f"🎯 ENSEMBLE SIGNAL: {ensemble_signal['direction'].value} "
                            f"(conf: {ensemble_signal['confidence']:.2f}, R/R: {rr_ratio:.2f}, min_conf: {min_confidence:.2f}, min_rr: {min_rr_ratio:.2f})")
                return ensemble_signal

            # Fallback to traditional engulfing patterns
            logger.debug(f"Ensemble signal weak (conf: {ensemble_signal['confidence']:.2f}, R/R: {rr_ratio:.2f}), falling back to engulfing patterns")

            pattern = self._detect_bullish_engulfing(df_recent)
            if pattern and isinstance(pattern, dict) and pattern.get("pattern") == PatternType.BULLISH_ENGULFING:
                # Calculate R/R for pattern
                entry = pattern.get("entry_price", 0)
                stop = pattern.get("stop_loss", 0)
                take = pattern.get("take_profit", 0)
                risk = abs(entry - stop)
                reward = abs(take - entry)
                rr = reward / risk if risk > 0 else 0.0
                conf = pattern.get("confidence", 1.0)
                logger.debug(f"   Bullish engulfing: conf={conf:.3f}, rr={rr:.2f}")
                if conf >= min_confidence and rr >= min_rr_ratio:
                    logger.info(f"🎯 BULLISH ENGULFING SIGNAL: conf={conf:.3f}, rr={rr:.2f}")
                    return {
                        "direction": SignalDirection.BUY,
                        "entry_price": entry,
                        "stop_loss": stop,
                        "take_profit": take,
                        "confidence": conf
                    }

            pattern = self._detect_bearish_engulfing(df_recent)
            if pattern and isinstance(pattern, dict) and pattern.get("pattern") == PatternType.BEARISH_ENGULFING:
                entry = pattern.get("entry_price", 0)
                stop = pattern.get("stop_loss", 0)
                take = pattern.get("take_profit", 0)
                risk = abs(entry - stop)
                reward = abs(take - entry)
                rr = reward / risk if risk > 0 else 0.0
                conf = pattern.get("confidence", 1.0)
                logger.debug(f"   Bearish engulfing: conf={conf:.3f}, rr={rr:.2f}")
                if conf >= min_confidence and rr >= min_rr_ratio:
                    logger.info(f"🎯 BEARISH ENGULFING SIGNAL: conf={conf:.3f}, rr={rr:.2f}")
                    return {
                        "direction": SignalDirection.SELL,
                        "entry_price": entry,
                        "stop_loss": stop,
                        "take_profit": take,
                        "confidence": conf
                    }

            logger.debug(f"   No valid signals found for {pair}")
            return {"direction": SignalDirection.NONE, "confidence": 0.0}

        except Exception as e:
            logger.error(f"Error in enhanced signal generation for {pair}: {e}")
            return {"direction": SignalDirection.NONE, "confidence": 0.0}
