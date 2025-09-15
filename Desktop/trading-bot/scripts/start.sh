#!/bin/bash

# Start Forex Trading Bot

set -e

echo "🚀 Starting Forex Trading Bot..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please run ./scripts/setup.sh first."
    exit 1
fi

# Check if required environment variables are set
source .env

if [ -z "$MT5_LOGIN" ] || [ "$MT5_LOGIN" = "your_mt5_login_here" ]; then
    echo "❌ MT5_LOGIN not set in .env file. Please configure your MT5 credentials."
    exit 1
fi

if [ -z "$MT5_PASSWORD" ] || [ "$MT5_PASSWORD" = "your_mt5_password_here" ]; then
    echo "❌ MT5_PASSWORD not set in .env file. Please configure your MT5 credentials."
    exit 1
fi

if [ -z "$MT5_SERVER" ] || [ "$MT5_SERVER" = "your_mt5_server_here" ]; then
    echo "❌ MT5_SERVER not set in .env file. Please configure your MT5 credentials."
    exit 1
fi

# Start services
echo "🐳 Starting Docker services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🔍 Checking service health..."

# Check if forex-bot is running
if docker-compose ps forex-bot | grep -q "Up"; then
    echo "✅ Forex Bot is running"
else
    echo "❌ Forex Bot failed to start"
    docker-compose logs forex-bot
    exit 1
fi

# Check if Prometheus is running
if docker-compose ps prometheus | grep -q "Up"; then
    echo "✅ Prometheus is running"
else
    echo "❌ Prometheus failed to start"
fi

# Check if Grafana is running
if docker-compose ps grafana | grep -q "Up"; then
    echo "✅ Grafana is running"
else
    echo "❌ Grafana failed to start"
fi

echo ""
echo "🎉 Forex Trading Bot started successfully!"
echo ""
echo "📊 Monitoring URLs:"
echo "   Grafana Dashboard: http://localhost:3000 (admin/admin123)"
echo "   Prometheus: http://localhost:9090"
echo "   Bot Metrics: http://localhost:8000/metrics"
echo ""
echo "📝 View logs with: docker-compose logs -f forex-bot"
echo "🛑 Stop with: ./scripts/stop.sh"