#!/bin/bash

# Stop Forex Trading Bot

set -e

echo "🛑 Stopping Forex Trading Bot..."

# Stop Docker services
docker-compose down

echo "✅ Forex Trading Bot stopped successfully!"
echo ""
echo "💡 To start again, run: ./scripts/start.sh"
echo "🗑️  To remove all data, run: ./scripts/clean.sh"