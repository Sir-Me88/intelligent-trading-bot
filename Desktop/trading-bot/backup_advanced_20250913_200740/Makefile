.PHONY: help setup start stop clean test lint type-check build backup restore logs

# Default target
help:
    @echo "Forex Trading Bot - Available Commands:"
    @echo ""
    @echo "  setup      - Initial setup (create directories, .env file)"
    @echo "  start      - Start the trading bot and monitoring stack"
    @echo "  stop       - Stop all services"
    @echo "  restart    - Restart all services"
    @echo "  clean      - Remove all containers, volumes, and data"
    @echo "  test       - Run unit tests with coverage"
    @echo "  lint       - Run code linting"
    @echo "  type-check - Run type checking"
    @echo "  build      - Build Docker images"
    @echo "  logs       - Show bot logs"
    @echo "  backup     - Create backup of data and configuration"
    @echo "  restore    - Restore from backup (usage: make restore BACKUP=path)"
    @echo "  status     - Show service status"
    @echo ""

setup:
    @./scripts/setup.sh

start:
    @./scripts/start.sh

stop:
    @./scripts/stop.sh

restart: stop start

clean:
    @./scripts/clean.sh

test:
    @./scripts/test.sh

lint:
    @echo "🔍 Running linting..."
    @ruff check src/ tests/

type-check:
    @echo "🔍 Running type checking..."
    @mypy src/ --ignore-missing-imports

build:
    @echo "🐳 Building Docker images..."
    @docker-compose build

logs:
    @docker-compose logs -f forex-bot

backup:
    @./scripts/backup.sh

restore:
    @if [ -z "$(BACKUP)" ]; then \
        echo "❌ Usage: make restore BACKUP=path/to/backup"; \
        exit 1; \
    fi
    @./scripts/restore.sh $(BACKUP)

status:
    @echo "📊 Service Status:"
    @docker-compose ps

# Development targets
dev-setup:
    @echo "🛠️  Setting up development environment..."
    @python3 -m venv venv
    @./venv/bin/pip install -r requirements.txt
    @echo "✅ Development environment ready!"
    @echo "💡 Activate with: source venv/bin/activate"

dev-test:
    @echo "🧪 Running development tests..."
    @source venv/bin/activate && pytest tests/ -v

dev-run:
    @echo "🚀 Running bot in development mode..."
    @source venv/bin/activate && python -m src.bot.trading_bot

# Docker targets
docker-build:
    @docker build -t forex-trading-bot .

docker-run:
    @docker run -d --name forex-bot --env-file .env -p 8000:8000 forex-trading-bot

docker-shell:
    @docker exec -it forex-bot /bin/bash

# Monitoring targets
grafana:
    @echo "🌐 Opening Grafana dashboard..."
    @open http://localhost:3000 || xdg-open http://localhost:3000

prometheus:
    @echo "🌐 Opening Prometheus..."
    @open http://localhost:9090 || xdg-open http://localhost:9090

metrics:
    @echo "📊 Bot metrics:"
    @curl -s http://localhost:8000/metrics | grep forex_bot | head -20