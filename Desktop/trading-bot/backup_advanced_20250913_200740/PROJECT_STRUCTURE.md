# Forex Trading Bot - Project Structure

```
forex_bot/
├── src/                              # Source code
│   ├── __init__.py
│   ├── analysis/                     # Analysis modules
│   │   ├── __init__.py
│   │   ├── correlation.py           # Correlation analysis
│   │   └── technical.py             # Technical analysis
│   ├── bot/                         # Main bot logic
│   │   ├── __init__.py
│   │   └── trading_bot.py           # Main trading bot
│   ├── config/                      # Configuration
│   │   ├── __init__.py
│   │   └── settings.py              # Settings management
│   ├── data/                        # Data providers
│   │   ├── __init__.py
│   │   └── market_data.py           # Market data management
│   ├── monitoring/                  # Monitoring and logging
│   │   ├── __init__.py
│   │   ├── logger.py                # Structured logging
│   │   └── metrics.py               # Prometheus metrics
│   ├── news/                        # News and sentiment
│   │   ├── __init__.py
│   │   ├── economic_calendar.py     # Economic events
│   │   └── sentiment.py             # Sentiment analysis
│   └── trading/                     # Trading logic
│       ├── __init__.py
│       ├── broker_interface.py      # Broker connections
│       └── position_manager.py      # Position management
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── conftest.py                  # Test configuration
│   ├── test_correlation_analysis.py
│   ├── test_position_manager.py
│   └── test_technical_analysis.py
├── config/                          # Configuration files
│   ├── grafana/                     # Grafana configuration
│   │   ├── dashboards/
│   │   │   └── forex-bot-dashboard.json
│   │   └── provisioning/
│   │       ├── dashboards/
│   │       │   └── dashboard.yml
│   │       └── datasources/
│   │           └── prometheus.yml
│   └── prometheus/
│       └── prometheus.yml
├── scripts/                         # Utility scripts
│   ├── backup.sh                    # Backup data
│   ├── clean.sh                     # Clean up
│   ├── restore.sh                   # Restore backup
│   ├── setup.sh                     # Initial setup
│   ├── start.sh                     # Start services
│   ├── stop.sh                      # Stop services
│   └── test.sh                      # Run tests
├── .github/                         # GitHub workflows
│   └── workflows/
│       └── ci.yml                   # CI/CD pipeline
├── logs/                            # Log files (created at runtime)
├── data/                            # Data storage (created at runtime)
├── backups/                         # Backup storage (created at runtime)
├── .env.example                     # Environment template
├── .gitignore                       # Git ignore rules
├── docker-compose.yml               # Docker Compose configuration
├── Dockerfile                       # Docker image definition
├── Makefile                         # Build automation
├── PROJECT_STRUCTURE.md             # This file
├── README.md                        # Project documentation
└── requirements.txt                 # Python dependencies
```

## Key Components

### 🎯 Trading Engine (`src/trading/`)
- **position_manager.py**: Position lifecycle, risk management, position sizing
- **broker_interface.py**: MT5 broker integration for trade execution

### 📊 Analysis Engine (`src/analysis/`)
- **technical.py**: Candlestick patterns, support/resistance, confluence
- **correlation.py**: Correlation matrix, hedging recommendations

### 📰 News & Sentiment (`src/news/`)
- **economic_calendar.py**: ForexFactory economic events filtering
- **sentiment.py**: Twitter and news sentiment analysis

### 📈 Data Management (`src/data/`)
- **market_data.py**: Multi-provider market data with caching

### 🤖 Bot Core (`src/bot/`)
- **trading_bot.py**: Main orchestrator, signal processing, execution

### 📊 Monitoring (`src/monitoring/`)
- **metrics.py**: Prometheus metrics collection
- **logger.py**: Structured JSON logging

### ⚙️ Configuration (`src/config/`)
- **settings.py**: Pydantic-based configuration management

## Deployment Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Trading Bot   │    │   Prometheus    │    │    Grafana      │
│   (Port 8000)   │───▶│   (Port 9090)   │───▶│   (Port 3000)   │
│                 │    │                 │    │                 │
│ • Signal Gen    │    │ • Metrics Store │    │ • Dashboards    │
│ • Risk Mgmt     │    │ • Alerting      │    │ • Visualization │
│ • Order Exec    │    │ • Time Series   │    │ • Monitoring    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│     Brokers     │
│                 │
│ • MT5 API       │
│                 │
└─────────────────┘
```

## Data Flow

1. **Market Data** → Technical Analysis → Signal Generation
2. **Economic Calendar** → News Filter → Trade Validation
3. **Sentiment Data** → Position Sizing Adjustment
4. **Correlation Matrix** → Hedging Decisions
5. **Risk Management** → Position Approval
6. **Broker Interface** → Order Execution
7. **Metrics Collection** → Monitoring Dashboard

## Getting Started

1. **Setup**: `make setup` or `./scripts/setup.sh`
2. **Configure**: Edit `.env` with API credentials
3. **Start**: `make start` or `./scripts/start.sh`
4. **Monitor**: Access Grafana at http://localhost:3000
5. **Stop**: `make stop` or `./scripts/stop.sh`

## Development

- **Tests**: `make test`
- **Linting**: `make lint`
- **Type Check**: `make type-check`
- **Dev Environment**: `make dev-setup`

## Production Deployment

- Containerized with Docker
- Health checks and auto-restart
- Prometheus metrics and Grafana dashboards
- Structured logging for analysis
- Automated CI/CD pipeline