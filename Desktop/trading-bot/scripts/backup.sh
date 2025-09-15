#!/bin/bash

# Backup Forex Trading Bot data

set -e

BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"

echo "💾 Creating backup in $BACKUP_DIR..."

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup logs
if [ -d "logs" ]; then
    echo "📝 Backing up logs..."
    cp -r logs "$BACKUP_DIR/"
fi

# Backup data
if [ -d "data" ]; then
    echo "📊 Backing up data..."
    cp -r data "$BACKUP_DIR/"
fi

# Backup configuration
echo "⚙️  Backing up configuration..."
cp .env "$BACKUP_DIR/" 2>/dev/null || echo "No .env file found"
cp -r config "$BACKUP_DIR/"

# Backup Docker volumes
echo "🐳 Backing up Docker volumes..."
docker run --rm -v forex_bot_prometheus_data:/data -v "$(pwd)/$BACKUP_DIR":/backup alpine tar czf /backup/prometheus_data.tar.gz -C /data . 2>/dev/null || echo "Prometheus volume not found"
docker run --rm -v forex_bot_grafana_data:/data -v "$(pwd)/$BACKUP_DIR":/backup alpine tar czf /backup/grafana_data.tar.gz -C /data . 2>/dev/null || echo "Grafana volume not found"

# Create backup info
echo "📋 Creating backup info..."
cat > "$BACKUP_DIR/backup_info.txt" << EOF
Forex Trading Bot Backup
========================
Date: $(date)
Host: $(hostname)
User: $(whoami)
Git Commit: $(git rev-parse HEAD 2>/dev/null || echo "Not a git repository")
Docker Images:
$(docker images | grep forex)
EOF

echo "✅ Backup completed: $BACKUP_DIR"
echo ""
echo "💡 To restore, run: ./scripts/restore.sh $BACKUP_DIR"