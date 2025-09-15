#!/bin/bash

# Forex Trading Bot Setup Script

set -e

echo "🚀 Setting up Forex Trading Bot..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p data
mkdir -p config/prometheus
mkdir -p config/grafana/provisioning/datasources
mkdir -p config/grafana/provisioning/dashboards
mkdir -p config/grafana/dashboards

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📋 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys before running the bot!"
fi

# Set permissions
echo "🔐 Setting permissions..."
chmod +x scripts/*.sh

# Build Docker images
echo "🐳 Building Docker images..."
docker-compose build

# Create Docker volumes
echo "💾 Creating Docker volumes..."
docker volume create forex_bot_prometheus_data
docker volume create forex_bot_grafana_data

echo "✅ Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API credentials"
echo "2. Run: ./scripts/start.sh"
echo "3. Access Grafana at http://localhost:3000 (admin/admin123)"
echo "4. Access Prometheus at http://localhost:9090"
echo ""
echo "For more information, see README.md"