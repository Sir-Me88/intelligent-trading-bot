#!/bin/bash

# Clean up Forex Trading Bot data and containers

set -e

echo "🗑️  Cleaning up Forex Trading Bot..."

# Stop and remove containers
echo "🐳 Stopping and removing containers..."
docker-compose down -v

# Remove Docker images
echo "🖼️  Removing Docker images..."
docker-compose down --rmi all

# Remove volumes
echo "💾 Removing Docker volumes..."
docker volume rm forex_bot_prometheus_data 2>/dev/null || true
docker volume rm forex_bot_grafana_data 2>/dev/null || true

# Clean up logs
echo "📝 Cleaning up logs..."
rm -rf logs/*

# Clean up data
echo "📊 Cleaning up data..."
rm -rf data/*

echo "✅ Cleanup completed successfully!"
echo ""
echo "💡 To set up again, run: ./scripts/setup.sh"