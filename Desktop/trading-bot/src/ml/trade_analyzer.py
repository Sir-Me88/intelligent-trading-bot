#!/usr/bin/env python3
"""Trade analyzer for ML-driven trading insights."""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging
import requests
from datetime import datetime
from src.config.settings import settings

logger = logging.getLogger(__name__)

class TradeAnalyzer:
    """Analyzes trades for ML-driven insights and model training."""
    
    def __init__(self):
        self.trades: List[Dict] = []
        self.model = None  # Placeholder for ML model (e.g., scikit-learn)
    
    def record_trade(self, trade: Dict):
        """Record a trade for analysis."""
        try:
            self.trades.append(trade)
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def train_models(self) -> Dict:
        """Train ML models on recorded trades."""
        try:
            if not self.trades:
                return {'status': 'NO_DATA', 'metrics': {}}
                
            df = pd.DataFrame(self.trades)
            if len(df) < 5:
                return {'status': 'INSUFFICIENT_DATA', 'metrics': {}}
                
            # Example: Train a simple decision tree (replace with your model)
            from sklearn.tree import DecisionTreeClassifier
            features = ['rsi_15m', 'ma_cross_signal', 'volatility', 'confidence']
            X = df[features]
            y = df['is_profitable']
            self.model = DecisionTreeClassifier(max_depth=5)
            self.model.fit(X, y)
            
            return {
                'status': 'TRAINED',
                'metrics': {
                    'n_trades': len(df),
                    'feature_importance': dict(zip(features, self.model.feature_importances_))
                }
            }
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {'status': 'FAILED', 'metrics': {}}
    
    def generate_insights(self) -> Dict:
        """Generate actionable insights from trade data."""
        try:
            if not self.trades:
                return {}
                
            df = pd.DataFrame(self.trades)
            insights = {}
            
            # Win rate and profit factor
            insights['win_rate'] = df['is_profitable'].mean()
            gross_profit = df[df['final_profit'] > 0]['final_profit'].sum()
            gross_loss = abs(df[df['final_profit'] < 0]['final_profit'].sum())
            insights['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Hourly performance
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            hourly_perf = df.groupby('hour')['final_profit'].mean().to_dict()
            insights['hourly_performance'] = hourly_perf
            
            # Pair performance
            pair_perf = df.groupby('symbol')['final_profit'].agg(['mean', 'count']).to_dict()
            insights['pair_performance'] = pair_perf
            
            # Confidence correlation
            insights['confidence_correlation'] = df['confidence'].corr(df['final_profit'])
            
            # Volatility impact
            insights['volatility_impact'] = df['volatility'].mean()
            
            # News impact analysis
            news_impact = self._analyze_news_impact(df)
            insights['news_impact'] = news_impact
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {}
    
    def analyze_trade_performance(self, trades: List[Dict]) -> Dict:
        """Analyze trade performance and return comprehensive metrics."""
        try:
            if not trades:
                return {'total_trades': 0, 'message': 'No trades to analyze'}

            df = pd.DataFrame(trades)

            # Basic metrics
            total_trades = len(df)
            winning_trades = len(df[df['profit'] > 0])
            losing_trades = len(df[df['profit'] < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            # Profit metrics
            total_profit = df['profit'].sum()
            avg_profit = df['profit'].mean()
            max_profit = df['profit'].max()
            max_loss = df['profit'].min()

            # Risk metrics
            profit_factor = abs(df[df['profit'] > 0]['profit'].sum() / df[df['profit'] < 0]['profit'].sum()) if len(df[df['profit'] < 0]) > 0 else float('inf')

            # Performance by pair
            pair_performance = df.groupby('symbol')['profit'].agg(['sum', 'mean', 'count']).to_dict('index')

            analysis = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'avg_profit': avg_profit,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'profit_factor': profit_factor,
                'pair_performance': pair_performance,
                'analysis_timestamp': datetime.now().isoformat()
            }

            logger.info(f"Trade performance analysis complete: {total_trades} trades, {win_rate:.1%} win rate")
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing trade performance: {e}")
            return {'error': str(e), 'total_trades': 0}

    def _analyze_news_impact(self, df: pd.DataFrame) -> Dict:
        """Analyze impact of news events on trade performance."""
        try:
            news_impacts = {}
            for _, trade in df.iterrows():
                try:
                    trade_time = datetime.fromisoformat(trade['timestamp'])
                    url = f"{settings.news_api_url}?pair={trade['symbol'][:3]}&date={trade_time.date()}"
                    response = requests.get(url, timeout=5)
                    events = response.json()
                    for event in events:
                        event_time = datetime.fromisoformat(event['time'])
                        if event.get('impact') == 'High' and abs((trade_time - event_time).total_seconds()) < 3600:
                            news_impacts[trade['ticket']] = {
                                'impact': 'High',
                                'profit_loss': trade['final_profit'],
                                'event_time': event['time']
                            }
                except Exception as e:
                    logger.debug(f"News impact analysis failed for trade {trade['ticket']}: {e}")
            return news_impacts
        except Exception as e:
            logger.error(f"Error in news impact analysis: {e}")
            return {}
