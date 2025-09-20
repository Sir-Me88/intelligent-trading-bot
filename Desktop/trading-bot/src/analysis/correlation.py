#!/usr/bin/env python3
"""Currency correlation analysis for hedging decisions."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import asyncio
import logging
from datetime import datetime, timedelta

from ..data.market_data import MarketDataManager

logger = logging.getLogger(__name__)


def calculate_correlation_matrix(returns_dict):
    """
    Calculate the correlation matrix for a dictionary of returns.
    returns_dict: {pair: [returns]}
    """
    # Diagnostic: Log available data for each pair
    for pair, returns in returns_dict.items():
        logger.info(f"[CORRELATION] {pair}: {len(returns)} return points, sample: {returns[:5]}")

    # Convert to DataFrame if possible
    import pandas as pd
    try:
        returns_df = pd.DataFrame(returns_dict)
        logger.info(f"[CORRELATION] Returns DataFrame shape: {returns_df.shape}")
        logger.info(f"[CORRELATION] Returns DataFrame head:\n{returns_df.head()}")
    except Exception as e:
        logger.error(f"[CORRELATION] Failed to create returns DataFrame: {e}")
        return None

    # Calculate correlation matrix
    try:
        correlation_matrix = returns_df.corr()
        if correlation_matrix is None or correlation_matrix.empty:
            logger.warning("[CORRELATION] Correlation matrix is empty or None after calculation.")
        else:
            logger.info(f"[CORRELATION] Correlation matrix calculated:\n{correlation_matrix}")
        return correlation_matrix
    except Exception as e:
        logger.error(f"[CORRELATION] Error calculating correlation matrix: {e}")
        return None


class CorrelationAnalyzer:
    """Analyzes currency pair correlations for hedging opportunities."""

    def __init__(self, data_manager: MarketDataManager = None, lookback_days: int = 60):
        self.data_manager = data_manager
        self.lookback_days = lookback_days
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.last_update: Optional[datetime] = None
        self.pair_volatilities: Dict[str, float] = {}

    async def update_correlation_matrix(self, currency_pairs: List[str]) -> pd.DataFrame:
        """Update the correlation matrix for all currency pairs (async)."""
        try:
            periods_needed = max(1, self.lookback_days * 24)

            async def _fetch_pair(pair: str):
                try:
                    df = await self.data_manager.get_candles(pair, "H1", periods_needed)
                    if df is None or len(df) == 0:
                        return pair, None
                    # ensure timestamp column is present
                    if "time" in df.columns:
                        df = df.reset_index(drop=True)
                        df["timestamp"] = df["time"]
                    elif "timestamp" not in df.columns and df.index.dtype != object:
                        df = df.reset_index().rename(columns={"index": "timestamp"})
                    elif df.index.name == "time":
                        # Handle case where time is the index
                        df = df.reset_index()
                        df["timestamp"] = df["time"]

                    # Create series with proper datetime index
                    if "timestamp" in df.columns:
                        series = pd.Series(df["close"].values, index=pd.to_datetime(df["timestamp"]))
                    else:
                        # Fallback: use the index if it's datetime-like
                        series = pd.Series(df["close"].values, index=pd.to_datetime(df.index))
                    series.name = pair
                    return pair, series
                except Exception as e:
                    logger.warning("Failed to fetch %s: %s", pair, e)
                    return pair, None

            tasks = [_fetch_pair(p) for p in currency_pairs]
            results = await asyncio.gather(*tasks, return_exceptions=False)

            price_data = {p: s for p, s in results if s is not None and not s.empty}
            if len(price_data) < 2:
                self.correlation_matrix = pd.DataFrame()
                self.last_update = datetime.utcnow()
                return self.correlation_matrix

            prices_df = pd.DataFrame(price_data)
            returns_df = prices_df.pct_change().dropna()
            corr = returns_df.corr()
            self.correlation_matrix = corr
            self.last_update = datetime.utcnow()

            # store volatilities
            for pair in price_data.keys():
                self.pair_volatilities[pair] = float(returns_df[pair].std()) if pair in returns_df.columns else 0.0

            logger.info("Updated correlation matrix for %d pairs", len(price_data))
            return corr

        except Exception as e:
            logger.error("Error updating correlation matrix: %s", e)
            raise

    def _calculate_correlation_matrix(self, currency_pairs: List[str], price_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """Synchronous helper used by tests to build a correlation matrix from provided series."""
        df = pd.DataFrame({p: price_data[p] for p in currency_pairs if p in price_data})
        if df.empty:
            self.correlation_matrix = pd.DataFrame()
            self.last_update = datetime.utcnow()
            return self.correlation_matrix
        returns = df.pct_change().dropna()
        corr = returns.corr()
        self.correlation_matrix = corr
        self.last_update = datetime.utcnow()
        for p in df.columns:
            self.pair_volatilities[p] = float(returns[p].std()) if p in returns.columns else 0.0
        return corr

    def find_hedging_opportunities(self, target_pair: str, correlation_threshold: float = 0.8) -> List[Dict]:
        """Find pairs suitable for hedging against the target pair."""
        if self.correlation_matrix is None or target_pair not in self.correlation_matrix.columns:
            return []

        opportunities = []
        target_correlations = self.correlation_matrix[target_pair]

        for pair, correlation in target_correlations.items():
            if pair == target_pair:
                continue

            abs_correlation = abs(correlation)
            if abs_correlation >= correlation_threshold:
                opportunities.append({
                    "pair": pair,
                    "correlation": float(correlation),
                    "abs_correlation": float(abs_correlation),
                    "hedge_ratio": float(min(abs_correlation * 1.2, 1.0))
                })

        return sorted(opportunities, key=lambda x: x["abs_correlation"], reverse=True)

    def should_hedge_position(self, open_positions: List[Dict], new_signal: Dict, correlation_threshold: float = 0.8) -> Dict:
        """Determine if a new position needs hedging based on correlations."""
        result = {
            "should_hedge": False,
            "hedge_pairs": [],
            "correlation_risk": 0.0,
            "net_exposure": 0.0,
        }

        if self.correlation_matrix is None or new_signal is None:
            return result

        new_pair = new_signal.get("pair")
        new_direction = str(new_signal.get("direction", "")).upper()

        if not new_pair or not new_direction:
            return result

        max_correlation = 0.0

        for position in open_positions:
            try:
                existing_pair = position.get("symbol") or position.get("pair")
                existing_type = position.get("type", None)  # 0=BUY, 1=SELL or 'buy'/'sell'
                existing_size = position.get("volume", position.get("size", 0.0))

                if existing_pair not in self.correlation_matrix.columns or new_pair not in self.correlation_matrix.columns:
                    continue

                correlation = float(self.correlation_matrix.loc[new_pair, existing_pair])
                abs_correlation = abs(correlation)
                max_correlation = max(max_correlation, abs_correlation)

                if abs_correlation < correlation_threshold:
                    continue

                # determine direction strings
                if existing_type in (0, "0", "BUY", "buy"):
                    existing_direction = "BUY"
                elif existing_type in (1, "1", "SELL", "sell"):
                    existing_direction = "SELL"
                else:
                    existing_direction = position.get("direction", "BUY").upper()

                same_direction = (new_direction == existing_direction)

                # if correlation positive and same direction => increases exposure
                # if correlation negative and opposite direction => increases exposure
                if (correlation > 0 and same_direction) or (correlation < 0 and not same_direction):
                    hedge_ratio = float(min(abs_correlation * 1.2, 1.0))
                    result["should_hedge"] = True
                    result["hedge_pairs"].append({
                        "pair": existing_pair,
                        "correlation": correlation,
                        "existing_size": float(existing_size or 0.0),
                        "hedge_ratio": hedge_ratio
                    })

            except Exception as e:
                logger.debug("Error accessing correlation for %s/%s: %s", new_pair, existing_pair if 'existing_pair' in locals() else None, e)
                continue

        result["correlation_risk"] = float(max_correlation)
        return result

    def get_currency_exposure(self, positions: List[Dict]) -> Dict[str, float]:
        """Calculate net exposure for each currency."""
        exposure: Dict[str, float] = {}

        for position in positions:
            try:
                symbol = position.get("symbol", position.get("pair", ""))
                if not symbol or len(symbol) < 6:
                    continue

                base_currency = symbol[:3]
                quote_currency = symbol[3:6]
                volume = float(position.get("volume", position.get("size", 0.0) or 0.0))
                position_type = position.get("type", position.get("direction", 0))

                # normalize type into 0=BUY, 1=SELL
                is_buy = position_type in (0, "0", "BUY", "buy")
                if is_buy:
                    exposure[base_currency] = exposure.get(base_currency, 0.0) + volume
                    exposure[quote_currency] = exposure.get(quote_currency, 0.0) - volume
                else:
                    exposure[base_currency] = exposure.get(base_currency, 0.0) - volume
                    exposure[quote_currency] = exposure.get(quote_currency, 0.0) + volume

            except Exception as e:
                logger.debug("Error calculating exposure for %s: %s", position, e)
                continue

        return exposure

    def calculate_hedge_ratio(self, pair1: str, pair2: str) -> float:
        """Calculate optimal hedge ratio between two pairs using linear regression fallback to correlation-based estimate."""
        if self.correlation_matrix is None:
            return 0.0

        try:
            periods_needed = min(self.lookback_days * 24, 500)

            df1 = asyncio.run(self.data_manager.get_candles(pair1, "H1", periods_needed))
            df2 = asyncio.run(self.data_manager.get_candles(pair2, "H1", periods_needed))

            if df1 is None or df2 is None or len(df1) < 30 or len(df2) < 30:
                correlation = float(self.correlation_matrix.loc[pair1, pair2]) if pair1 in self.correlation_matrix.columns and pair2 in self.correlation_matrix.columns else 0.0
                return float(min(abs(correlation) * 1.2, 1.0))

            returns1 = df1["close"].pct_change().dropna()
            returns2 = df2["close"].pct_change().dropna()

            min_length = min(len(returns1), len(returns2))
            returns1 = returns1.iloc[-min_length:]
            returns2 = returns2.iloc[-min_length:]

            X = np.column_stack([np.ones(len(returns2)), returns2.values])
            beta, alpha = np.linalg.lstsq(X, returns1.values, rcond=None)[0]

            hedge_ratio = float(min(abs(beta), 1.0))
            return hedge_ratio

        except Exception as e:
            logger.error("Error calculating hedge ratio for %s/%s: %s", pair1, pair2, e)
            try:
                correlation = float(self.correlation_matrix.loc[pair1, pair2])
                return float(min(abs(correlation) * 1.2, 1.0))
            except Exception:
                return 0.0

    def get_correlation_summary(self) -> Dict:
        """Get a summary of current correlations."""
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            return {"status": "No correlation data available"}

        try:
            correlations = []
            cols = list(self.correlation_matrix.columns)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    pair1 = cols[i]
                    pair2 = cols[j]
                    corr = float(self.correlation_matrix.iloc[i, j])
                    correlations.append({
                        "pair1": pair1,
                        "pair2": pair2,
                        "correlation": corr,
                        "abs_correlation": abs(corr)
                    })

            correlations.sort(key=lambda x: x["abs_correlation"], reverse=True)
            avg_corr = float(np.mean([c["abs_correlation"] for c in correlations])) if correlations else 0.0

            return {
                "status": "Active",
                "last_update": self.last_update.isoformat() if self.last_update else None,
                "total_pairs": len(cols),
                "highest_correlations": correlations[:10],
                "average_correlation": avg_corr
            }

        except Exception as e:
            logger.error("Error generating correlation summary: %s", e)
            return {"status": "Error generating summary", "error": str(e)}

    def get_mpt_weights(self, expected_returns: np.ndarray, target_return: float = None) -> Dict[str, float]:
        """Calculate Modern Portfolio Theory (MPT) optimal weights using mean-variance optimization."""
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            logger.warning("No correlation matrix available for MPT optimization")
            return {}

        try:
            from scipy.optimize import minimize

            pairs = list(self.correlation_matrix.columns)
            n_assets = len(pairs)

            if n_assets < 2:
                logger.warning("Need at least 2 assets for MPT optimization")
                return {pairs[0]: 1.0} if pairs else {}

            # Use historical volatilities as risk estimates
            volatilities = np.array([self.pair_volatilities.get(pair, 0.02) for pair in pairs])

            # Create covariance matrix from correlations and volatilities
            corr_matrix = self.correlation_matrix.values
            cov_matrix = np.outer(volatilities, volatilities) * corr_matrix

            # If no expected returns provided, use volatility-based estimates
            if expected_returns is None:
                # Higher volatility pairs get higher expected returns (simplified)
                expected_returns = volatilities * 0.1  # 10% annualized per unit volatility

            # Define optimization functions
            def portfolio_volatility(weights):
                """Calculate portfolio volatility."""
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            def portfolio_return(weights):
                """Calculate portfolio expected return."""
                return np.dot(weights, expected_returns)

            def negative_sharpe_ratio(weights):
                """Negative Sharpe ratio for minimization."""
                port_return = portfolio_return(weights)
                port_vol = portfolio_volatility(weights)
                return -port_return / port_vol if port_vol > 0 else 0

            def portfolio_variance(weights):
                """Minimize variance for minimum variance portfolio."""
                return portfolio_volatility(weights) ** 2

            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            ]

            # Add target return constraint if specified
            if target_return is not None:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x: portfolio_return(x) - target_return
                })

            # Bounds: weights between 0 and 1
            bounds = tuple((0, 1) for _ in range(n_assets))

            # Initial guess: equal weights
            initial_weights = np.array([1/n_assets] * n_assets)

            # Optimize for maximum Sharpe ratio (minimum variance if no target return)
            if target_return is None:
                # Minimum variance portfolio
                result = minimize(portfolio_variance, initial_weights,
                                method='SLSQP', bounds=bounds, constraints=constraints)
            else:
                # Maximum Sharpe ratio portfolio
                result = minimize(negative_sharpe_ratio, initial_weights,
                                method='SLSQP', bounds=bounds, constraints=constraints)

            if result.success:
                optimal_weights = result.x
                # Create weights dictionary
                weights_dict = {pairs[i]: float(optimal_weights[i]) for i in range(n_assets)}

                # Remove zero weights for cleaner output
                weights_dict = {k: v for k, v in weights_dict.items() if v > 0.001}

                # Renormalize
                total_weight = sum(weights_dict.values())
                if total_weight > 0:
                    weights_dict = {k: v/total_weight for k, v in weights_dict.items()}

                logger.info(f"MPT optimization successful: {len(weights_dict)} assets with non-zero weights")
                return weights_dict
            else:
                logger.warning(f"MPT optimization failed: {result.message}")
                # Fallback to equal weights
                return {pair: 1.0/n_assets for pair in pairs}

        except Exception as e:
            logger.error(f"Error in MPT optimization: {e}")
            # Fallback to equal weights
            pairs = list(self.correlation_matrix.columns) if self.correlation_matrix is not None else []
            return {pair: 1.0/len(pairs) for pair in pairs} if pairs else {}

    def optimize_portfolio_risk(self, current_positions: List[Dict], max_risk_per_pair: float = 0.05) -> Dict:
        """Optimize portfolio to minimize risk while maintaining diversification."""
        try:
            if not current_positions:
                return {"status": "No positions to optimize", "adjustments": {}}

            # Get current exposure
            current_exposure = self.get_currency_exposure(current_positions)

            # Calculate position sizes based on correlation
            pair_weights = {}
            total_exposure = sum(abs(v) for v in current_exposure.values())

            if total_exposure == 0:
                return {"status": "No exposure to optimize", "adjustments": {}}

            # Adjust weights based on correlation and risk
            for position in current_positions:
                pair = position.get("symbol", position.get("pair", ""))
                if not pair:
                    continue

                # Find correlated pairs for risk assessment
                correlated_pairs = self.find_hedging_opportunities(pair, correlation_threshold=0.6)

                # Calculate risk multiplier based on correlation
                risk_multiplier = 1.0
                if correlated_pairs:
                    avg_correlation = np.mean([abs(p["correlation"]) for p in correlated_pairs])
                    risk_multiplier = 1.0 + (avg_correlation * 0.5)  # Increase risk weight for correlated pairs

                # Calculate optimal size
                current_size = position.get("volume", position.get("size", 0))
                optimal_size = min(current_size, max_risk_per_pair / risk_multiplier)

                pair_weights[pair] = {
                    "current_size": current_size,
                    "optimal_size": optimal_size,
                    "risk_multiplier": risk_multiplier,
                    "correlated_pairs": len(correlated_pairs)
                }

            return {
                "status": "Optimization complete",
                "adjustments": pair_weights,
                "total_positions": len(current_positions),
                "risk_reduction": sum(pw["risk_multiplier"] for pw in pair_weights.values()) / len(pair_weights) if pair_weights else 1.0
            }

        except Exception as e:
            logger.error(f"Error in portfolio risk optimization: {e}")
            return {"status": "Optimization failed", "error": str(e), "adjustments": {}}

    def calculate_portfolio_metrics(self, positions: List[Dict]) -> Dict:
        """Calculate comprehensive portfolio metrics including diversification and risk."""
        try:
            if not positions:
                return {"status": "No positions", "metrics": {}}

            # Get exposure
            exposure = self.get_currency_exposure(positions)

            # Calculate diversification metrics
            total_exposure = sum(abs(v) for v in exposure.values())
            max_exposure = max(abs(v) for v in exposure.values()) if exposure else 0
            concentration_ratio = max_exposure / total_exposure if total_exposure > 0 else 0

            # Calculate correlation-based risk
            pair_correlations = []
            position_pairs = [p.get("symbol", p.get("pair", "")) for p in positions if p.get("symbol") or p.get("pair")]

            for i, pair1 in enumerate(position_pairs):
                for pair2 in position_pairs[i+1:]:
                    if (self.correlation_matrix is not None and
                        pair1 in self.correlation_matrix.columns and
                        pair2 in self.correlation_matrix.columns):
                        corr = abs(self.correlation_matrix.loc[pair1, pair2])
                        pair_correlations.append(corr)

            avg_correlation = np.mean(pair_correlations) if pair_correlations else 0.0

            # Calculate portfolio volatility estimate
            portfolio_volatility = 0.0
            if self.correlation_matrix is not None and position_pairs:
                # Simplified portfolio volatility calculation
                weights = np.array([1.0/len(position_pairs)] * len(position_pairs))
                volatilities = np.array([self.pair_volatilities.get(pair, 0.02) for pair in position_pairs])

                # Portfolio variance = w^T * Σ * w where Σ is covariance matrix
                corr_subset = self.correlation_matrix.loc[position_pairs, position_pairs].values
                cov_matrix = np.outer(volatilities, volatilities) * corr_subset
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            return {
                "status": "Metrics calculated",
                "metrics": {
                    "total_exposure": total_exposure,
                    "concentration_ratio": concentration_ratio,
                    "currency_count": len(exposure),
                    "average_correlation": avg_correlation,
                    "portfolio_volatility": portfolio_volatility,
                    "diversification_score": 1.0 - concentration_ratio,  # Higher is better
                    "correlation_risk": avg_correlation  # Lower is better
                },
                "exposure_breakdown": exposure,
                "position_count": len(positions)
            }

        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {"status": "Calculation failed", "error": str(e), "metrics": {}}

    def get_correlation(self, pair1: str, pair2: str) -> float:
        """Get correlation between two pairs."""
        if self.correlation_matrix is None:
            return 0.0
        try:
            return float(self.correlation_matrix.loc[pair1, pair2])
        except (KeyError, TypeError):
            return 0.0

    def find_correlated_pairs(self, target_pair: str, threshold: float = 0.7) -> List[Dict]:
        """Find pairs correlated with target pair above threshold."""
        if self.correlation_matrix is None or target_pair not in self.correlation_matrix.columns:
            return []

        correlated = []
        for pair in self.correlation_matrix.columns:
            if pair == target_pair:
                continue
            corr = abs(self.get_correlation(target_pair, pair))
            if corr >= threshold:
                correlated.append({
                    "pair": pair,
                    "correlation": self.get_correlation(target_pair, pair),
                    "abs_correlation": corr
                })

        return sorted(correlated, key=lambda x: x["abs_correlation"], reverse=True)

    def is_correlation_matrix_stale(self) -> bool:
        """Check if correlation matrix needs updating."""
        if self.last_update is None:
            return True
        # Consider stale if older than 1 hour
        return (datetime.utcnow() - self.last_update) > timedelta(hours=1)

    def _calculate_hedge_ratio(self, pair1: str, pair2: str, correlation: float = None) -> float:
        """Calculate hedge ratio based on correlation and volatility."""
        if correlation is None:
            correlation = self.get_correlation(pair1, pair2)

        vol1 = self.pair_volatilities.get(pair1, 0.02)
        vol2 = self.pair_volatilities.get(pair2, 0.02)

        if vol2 > 0:
            ratio = abs(correlation) * (vol1 / vol2)
        else:
            ratio = abs(correlation)

        return float(min(ratio, 1.0))
