# Forex Trading Bot

A production-ready, 24/7 automated forex trading bot implementing candlestick pattern recognition, risk management, sentiment analysis, and correlation-based hedging.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Analysis Layer │    │ Trading Engine  │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • MT5 API       │───▶│ • Technical     │───▶│ • Position Mgmt │
│ • ForexFactory  │    │ • Correlation   │    │ • Risk Mgmt     │
│ • ForexFactory  │    │ • Sentiment     │    │ • Order Exec    │
│ • Twitter API   │    │ • Economic Cal  │    │ • Hedging       │
│ • EventRegistry │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │    Monitoring Layer     │
                    ├─────────────────────────┤
                    │ • Prometheus Metrics    │
                    │ • Grafana Dashboard     │
                    │ • Structured Logging    │
                    │ • Health Checks         │
                    └─────────────────────────┘
```

## Features

### 🎯 Trading Strategy
- **Candlestick Patterns**: Bullish/bearish engulfing, pin bars, inside bar breakouts
- **Confluence Trading**: Patterns must occur at support/resistance levels
- **Multi-timeframe**: 15-minute and 1-hour analysis
- **Dynamic Stops**: ATR-based stop losses and take profits
- **Trailing Stops**: Chandelier exit system

### 🛡️ Risk Management
- **Position Sizing**: 1% risk per trade, max 6% total exposure
- **Correlation Analysis**: 60-day rolling correlation matrix
- **Hedging**: Automatic hedge trades for correlated pairs (>0.8 correlation)
- **Economic Calendar**: Skip trading during high-impact news events

### 📊 Sentiment Analysis
- **Twitter Monitoring**: Real-time forex sentiment from Twitter
- **News Analysis**: Breaking news sentiment via EventRegistry
- **Multi-model**: VADER + FinBERT sentiment scoring
- **Position Adjustment**: Reduce size by 50% on negative sentiment

### 🔄 24/7 Operation
- **Docker Deployment**: Containerized for reliable operation
- **Auto-restart**: Automatic recovery from crashes
- **Health Checks**: Built-in monitoring and alerting
- **Graceful Shutdown**: Clean resource cleanup

### 📈 Monitoring
- **Prometheus Metrics**: Real-time performance tracking
- **Grafana Dashboard**: Visual monitoring interface
- **Structured Logging**: JSON logs for analysis
- **Performance Metrics**: Latency, equity, PnL tracking

## Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone <repository-url>
cd forex_bot

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env
```

### 2. Configure API Keys
Add your credentials to `.env`:
```env
MT5_LOGIN=your_mt5_login
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_mt5_server
TWITTER_BEARER_TOKEN=your_twitter_token
EVENTREGISTRY_API_KEY=your_eventregistry_key
```

### 3. Deploy with Docker
```bash
# Build and start all services
docker compose up --build -d

# View logs
docker compose logs -f forex-bot

# Check status
docker compose ps
```

### 4. Access Monitoring
- **Grafana Dashboard**: http://localhost:3000 (admin