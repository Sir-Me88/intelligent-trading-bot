#!/bin/bash

# Restore Forex Trading Bot data from backup

set -e

if [ $# -eq 0 ]; then
    echo "❌ Usage: $0 <backup_directory>"
    echo "Available backups:"
    ls -la backups/ 2>/dev/null || echo "No backups found"
    exit 1
fi

BACKUP_DIR="$1"

if [ ! -d "$BACKUP_DIR" ]; then
    echo "❌ Backup directory not found: $BACKUP_DIR"
    exit 1
fi

echo "🔄 Restoring from backup: $BACKUP_DIR"

# Stop services first
echo "🛑 Stopping services..."
docker-compose down

# Restore logs
if [ -d "$BACKUP_DIR/logs" ]; then
    echo "📝 Restoring logs..."
    rm -rf logs
    cp -r "$BACKUP_DIR/logs" .
fi

# Restore data
if [ -d "$BACKUP_DIR/data" ]; then
    echo "📊 Restoring data..."
    rm -rf data
    cp -r "$BACKUP_DIR/data" .
fi

# Restore configuration
echo "⚙️  Restoring configuration..."
if [ -f "$BACKUP_DIR/.env" ]; then
    cp "$BACKUP_DIR/.env" .
fi
cp -r "$BACKUP_DIR/config" . 2>/dev/null || true

# Restore Docker volumes
if [ -f "$BACKUP_DIR/prometheus_data.tar.gz" ]; then
    echo "🐳 Restoring Prometheus data..."
    docker volume create forex_bot_prometheus_data
    docker run --rm -v forex_bot_prometheus_data:/data -v "$(pwd)/$BACKUP_DIR":/backup alpine tar xzf /backup/prometheus_data.tar.gz -C /data
fi

if [ -f "$BACKUP_DIR/grafana_data.tar.gz" ]; then
    echo "🐳 Restoring Grafana data..."
    docker volume create forex_bot_grafana_data
    docker run --rm -v forex_bot_grafana_data:/data -v "$(pwd)/$BACKUP_DIR":/backup alpine tar xzf /backup/grafana_data.tar.gz -C /data
fi

echo "✅ Restore completed!"
echo ""
echo "💡 Start services with: ./scripts/start.sh"