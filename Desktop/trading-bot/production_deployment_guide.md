# 🚀 Production Deployment Guide - Trading Bot

## Overview
This guide provides complete instructions for deploying the trading bot to production on a VPS (Virtual Private Server).

## 📋 Prerequisites

### System Requirements
- **Ubuntu 20.04+** or **CentOS 7+**
- **4GB RAM minimum** (8GB recommended)
- **2 CPU cores minimum**
- **50GB SSD storage**
- **Python 3.10+**
- **Docker & Docker Compose**

### Network Requirements
- **Static IP address** (recommended)
- **Open ports**: 22 (SSH), 3000 (Grafana), 9090 (Prometheus)
- **MT5 broker connectivity** (whitelist VPS IP)

---

## 🏗️ Step 1: Server Setup

### 1.1 Update System
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl wget git htop ufw
```

### 1.2 Install Docker
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker
```

### 1.3 Configure Firewall
```bash
# Allow SSH
sudo ufw allow ssh
sudo ufw allow 22

# Allow monitoring ports
sudo ufw allow 3000  # Grafana
sudo ufw allow 9090  # Prometheus

# Enable firewall
sudo ufw --force enable
```

### 1.4 Install Python & Dependencies
```bash
# Install Python 3.10+
sudo apt install -y python3.10 python3.10-venv python3-pip

# Install system dependencies
sudo apt install -y build-essential libssl-dev libffi-dev python3-dev
```

---

## 📦 Step 2: Deploy Trading Bot

### 2.1 Clone Repository
```bash
cd /opt
sudo mkdir trading-bot
sudo chown $USER:$USER trading-bot
cd trading-bot

# Clone your repository
git clone https://github.com/your-username/intelligent-trading-bot.git .
```

### 2.2 Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit with your credentials
nano .env
```

**Required .env configuration:**
```env
# MT5 Credentials (REQUIRED)
MT5_LOGIN=your_mt5_account_number
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_mt5_broker_server

# Economic Calendar (REQUIRED)
FMP_API_KEY=your_fmp_api_key

# Optional APIs
TWELVE_DATA_KEY=your_twelve_data_key
TELEGRAM_TOKEN=your_telegram_token
TELEGRAM_AUTHORIZED_USERS=your_chat_id
EVENTREGISTRY_API_KEY=your_eventregistry_key
TWITTER_BEARER_TOKEN=your_twitter_token

# Production Settings
PAPER_MODE=false
LOG_LEVEL=INFO
```

### 2.3 Install Python Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python -c "import MetaTrader5 as mt5; print('MT5 installed successfully')"
```

### 2.4 Test Configuration
```bash
# Test MT5 connection
python test_mt5_connection.py

# Test API configuration
python setup_api_config.py --test

# Test ML components
python test_ml_components.py
```

---

## 🐳 Step 3: Docker Deployment (Recommended)

### 3.1 Build Docker Image
```bash
# Build the image
docker build -t trading-bot:latest .

# Verify build
docker images | grep trading-bot
```

### 3.2 Configure Docker Compose
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  trading-bot:
    image: trading-bot:latest
    container_name: trading-bot-prod
    restart: unless-stopped
    environment:
      - MT5_LOGIN=${MT5_LOGIN}
      - MT5_PASSWORD=${MT5_PASSWORD}
      - MT5_SERVER=${MT5_SERVER}
      - FMP_API_KEY=${FMP_API_KEY}
      - PAPER_MODE=false
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - ./ml_data:/app/ml_data
    networks:
      - trading-network

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    networks:
      - trading-network

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=your_secure_password
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana/provisioning:/etc/grafana/provisioning
    networks:
      - trading-network

networks:
  trading-network:
    driver: bridge

volumes:
  prometheus_data:
  grafana_data:
```

### 3.3 Deploy with Docker Compose
```bash
# Start all services
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f trading-bot
```

---

## 📊 Step 4: Monitoring Setup

### 4.1 Access Grafana
1. Open browser: `http://your-vps-ip:3000`
2. Login with admin/your_secure_password
3. Import trading bot dashboard

### 4.2 Configure Alerts
```yaml
# Alert rules for Prometheus
groups:
  - name: trading_bot_alerts
    rules:
      - alert: BotDown
        expr: up{job="trading_bot"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Trading bot is down"
          description: "Trading bot has been down for more than 5 minutes"

      - alert: HighDrawdown
        expr: trading_bot_drawdown > 0.05
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High drawdown detected"
          description: "Portfolio drawdown exceeds 5%"
```

### 4.3 Log Monitoring
```bash
# View real-time logs
docker-compose -f docker-compose.prod.yml logs -f trading-bot

# Monitor system resources
htop

# Check disk usage
df -h

# Monitor network
sudo netstat -tlnp | grep :9090
```

---

## 🔄 Step 5: Automation & Maintenance

### 5.1 Create Systemd Service
```bash
# Create service file
sudo nano /etc/systemd/system/trading-bot.service
```

```ini
[Unit]
Description=Trading Bot Service
After=docker.service
Requires=docker.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/trading-bot
ExecStart=/usr/local/bin/docker-compose -f docker-compose.prod.yml up
ExecStop=/usr/local/bin/docker-compose -f docker-compose.prod.yml down
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot

# Check status
sudo systemctl status trading-bot
```

### 5.2 Backup Automation
```bash
# Create backup script
sudo nano /opt/trading-bot/backup.sh
```

```bash
#!/bin/bash
BACKUP_DIR="/opt/trading-bot/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup logs and data
tar -czf $BACKUP_DIR/trading_bot_$DATE.tar.gz \
    /opt/trading-bot/logs \
    /opt/trading-bot/data \
    /opt/trading-bot/ml_data \
    /opt/trading-bot/.env

# Keep only last 7 backups
cd $BACKUP_DIR
ls -t *.tar.gz | tail -n +8 | xargs -r rm

echo "Backup completed: trading_bot_$DATE.tar.gz"
```

```bash
# Make executable and schedule
chmod +x /opt/trading-bot/backup.sh
sudo crontab -e

# Add to crontab (daily at 2 AM)
0 2 * * * /opt/trading-bot/backup.sh
```

### 5.3 Log Rotation
```bash
# Configure logrotate
sudo nano /etc/logrotate.d/trading-bot
```

```
/opt/trading-bot/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    create 644 ubuntu ubuntu
    postrotate
        docker-compose -f /opt/trading-bot/docker-compose.prod.yml restart trading-bot
    endscript
}
```

---

## 🔧 Step 6: Maintenance & Updates

### 6.1 Update Procedure
```bash
# Stop the bot
docker-compose -f docker-compose.prod.yml down

# Pull latest changes
git pull origin main

# Update dependencies
source venv/bin/activate
pip install -r requirements.txt

# Rebuild Docker image
docker build -t trading-bot:latest .

# Restart services
docker-compose -f docker-compose.prod.yml up -d
```

### 6.2 Health Checks
```bash
# Create health check script
nano /opt/trading-bot/health_check.sh
```

```bash
#!/bin/bash

# Check if containers are running
if ! docker-compose -f docker-compose.prod.yml ps | grep -q "Up"; then
    echo "Containers are not running"
    exit 1
fi

# Check bot heartbeat
if [ ! -f "logs/core_bot_heartbeat.json" ]; then
    echo "Heartbeat file missing"
    exit 1
fi

# Check file age (should be updated within last 5 minutes)
if [[ $(find "logs/core_bot_heartbeat.json" -mmin +5) ]]; then
    echo "Heartbeat file is stale"
    exit 1
fi

echo "Health check passed"
exit 0
```

### 6.3 Emergency Procedures

#### Stop Trading Immediately
```bash
# Emergency stop
docker-compose -f docker-compose.prod.yml exec trading-bot python -c "
from run_core_trading_bot import CoreTradingBot
import asyncio
bot = CoreTradingBot()
asyncio.run(bot.emergency_stop())
"
```

#### Restart Services
```bash
# Full restart
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml up -d
```

#### Access Emergency Console
```bash
# Enter container
docker-compose -f docker-compose.prod.yml exec trading-bot bash

# Manual intervention
python -c "from src.trading.broker_interface import BrokerManager; bm = BrokerManager(); print('Broker connected:', bm.is_connected())"
```

---

## 📈 Step 7: Performance Optimization

### 7.1 System Tuning
```bash
# Increase file watchers
echo "fs.inotify.max_user_watches=524288" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Optimize Docker
sudo nano /etc/docker/daemon.json
```

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2"
}
```

### 7.2 Memory Management
```bash
# Check memory usage
docker stats

# Limit container memory
docker-compose -f docker-compose.prod.yml up -d --scale trading-bot=1
```

### 7.3 Network Optimization
```bash
# Optimize network settings
sudo nano /etc/sysctl.conf
```

```
net.core.somaxconn=65536
net.ipv4.tcp_tw_reuse=1
net.ipv4.ip_local_port_range=1024 65535
```

---

## 🔒 Step 8: Security Hardening

### 8.1 SSH Security
```bash
# Disable password authentication
sudo nano /etc/ssh/sshd_config
```

```
PasswordAuthentication no
PermitRootLogin no
```

```bash
sudo systemctl restart sshd
```

### 8.2 Firewall Rules
```bash
# Additional security rules
sudo ufw allow from your_ip_address to any port 22
sudo ufw default deny incoming
sudo ufw default allow outgoing
```

### 8.3 Container Security
```bash
# Run security scan
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    clair-scanner --ip localhost trading-bot:latest
```

---

## 📞 Step 9: Monitoring & Support

### 9.1 Log Analysis
```bash
# View recent errors
docker-compose -f docker-compose.prod.yml logs --tail=100 trading-bot | grep ERROR

# Monitor performance
docker stats --no-stream
```

### 9.2 Remote Access
```bash
# SSH tunnel for Grafana
ssh -L 3000:localhost:3000 user@your-vps-ip

# Access logs remotely
ssh user@your-vps-ip "docker-compose -f /opt/trading-bot/docker-compose.prod.yml logs -f trading-bot"
```

### 9.3 Troubleshooting Commands
```bash
# Check system resources
htop
df -h
free -h

# Network diagnostics
ping -c 4 8.8.8.8
traceroute your_mt5_server

# Docker diagnostics
docker system df
docker container ls -a
```

---

## 🎯 Step 10: Go-Live Checklist

- [ ] Server provisioned with required specifications
- [ ] Docker and Docker Compose installed
- [ ] Firewall configured
- [ ] Repository cloned and configured
- [ ] Environment variables set
- [ ] MT5 connection tested
- [ ] API keys configured and tested
- [ ] Docker containers built and running
- [ ] Grafana dashboard accessible
- [ ] Backup automation configured
- [ ] Log rotation configured
- [ ] Systemd service created
- [ ] Health checks passing
- [ ] Emergency procedures documented

---

## 🚨 Emergency Contacts

**Critical Issues:**
- MT5 Connection: Check broker status
- System Down: Restart Docker services
- High Drawdown: Emergency stop procedure

**Support Resources:**
- Documentation: `/opt/trading-bot/README.md`
- Logs: `/opt/trading-bot/logs/`
- Backups: `/opt/trading-bot/backups/`

---

## 📈 Scaling Considerations

### Horizontal Scaling
```bash
# Run multiple bot instances
docker-compose -f docker-compose.prod.yml up -d --scale trading-bot=3
```

### Load Balancing
```bash
# Use nginx for load balancing
sudo apt install nginx
sudo nano /etc/nginx/sites-available/trading-bot
```

---

**Your trading bot is now deployed to production! 🎉**

Monitor performance through Grafana and ensure regular backups are maintained.
