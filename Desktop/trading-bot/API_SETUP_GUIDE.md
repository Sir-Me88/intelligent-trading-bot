# 🔑 API Configuration Guide - Trading Bot

## Overview
This guide helps you configure all the API keys and credentials needed for the trading bot's advanced features.

## 📋 Required vs Optional APIs

### 🔴 **REQUIRED (for basic trading)**
- **MT5 Credentials**: Essential for live trading
- **FMP API Key**: Economic calendar data

### 🟡 **OPTIONAL (enhance functionality)**
- **Twelve Data API**: Additional market data
- **Telegram Bot**: Mobile notifications and control
- **EventRegistry API**: News sentiment analysis
- **Twitter API**: Social media sentiment

---

## 🚀 Quick Setup Script

Run this script to configure all APIs:

```bash
python setup_api_config.py
```

Or configure manually following the steps below.

---

## 🔧 Manual Configuration

### 1. Create/Edit .env File

Create a `.env` file in the project root:

```bash
cp .env.example .env
nano .env
```

### 2. Configure Each API

#### **MT5 Broker Credentials (REQUIRED)**
```env
# MT5 Trading Account
MT5_LOGIN=your_mt5_account_number
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_mt5_broker_server
```

**How to get MT5 credentials:**
1. Download and install MetaTrader 5
2. Create a live/demo account with a broker
3. Note your login number, password, and server

#### **Financial Modeling Prep API (REQUIRED)**
```env
# Economic Calendar API
FMP_API_KEY=your_fmp_api_key
```

**How to get FMP API key:**
1. Go to [Financial Modeling Prep](https://financialmodelingprep.com/)
2. Sign up for free account
3. Get your API key from dashboard

#### **Twelve Data API (OPTIONAL)**
```env
# Additional Market Data
TWELVE_DATA_KEY=your_twelve_data_key
```

**How to get Twelve Data API key:**
1. Go to [Twelve Data](https://twelvedata.com/)
2. Sign up and get free API key
3. 800 requests/day free tier

#### **Telegram Bot (OPTIONAL)**
```env
# Telegram Notifications & Control
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_AUTHORIZED_USERS=123456789,987654321
```

**How to set up Telegram bot:**
1. Message @BotFather on Telegram
2. Send `/newbot` and follow instructions
3. Copy the bot token
4. Get your user ID from @userinfobot
5. Add user IDs to `TELEGRAM_AUTHORIZED_USERS` (comma-separated)

#### **EventRegistry News API (OPTIONAL)**
```env
# News Sentiment Analysis
EVENTREGISTRY_API_KEY=your_eventregistry_key
```

**How to get EventRegistry API key:**
1. Go to [EventRegistry](https://eventregistry.org/)
2. Sign up for free account
3. Get your API key

#### **Twitter API (OPTIONAL)**
```env
# Social Media Sentiment
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
```

**How to get Twitter Bearer Token:**
1. Go to [Twitter Developer Portal](https://developer.twitter.com/)
2. Create a project/app
3. Get Bearer Token from app settings

---

## 📄 Complete .env Example

```env
# ===========================================
# TRADING BOT ENVIRONMENT CONFIGURATION
# ===========================================

# MT5 Broker Credentials (REQUIRED)
MT5_LOGIN=12345678
MT5_PASSWORD=your_secure_password
MT5_SERVER=YourBroker-Server01

# Economic Calendar (REQUIRED)
FMP_API_KEY=your_fmp_api_key_here

# Additional Market Data (OPTIONAL)
TWELVE_DATA_KEY=your_twelve_data_key_here

# Telegram Bot Control (OPTIONAL)
TELEGRAM_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_AUTHORIZED_USERS=123456789,987654321

# News Sentiment (OPTIONAL)
EVENTREGISTRY_API_KEY=your_eventregistry_key_here

# Social Media Sentiment (OPTIONAL)
TWITTER_BEARER_TOKEN=AAAAAAAAAAAAAAAAAAAAA

# ===========================================
# ADVANCED CONFIGURATION
# ===========================================

# Paper Trading Mode (set to 'true' for testing)
PAPER_MODE=false

# Logging Level
LOG_LEVEL=INFO

# Database Configuration (if using)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_bot
DB_USER=postgres
DB_PASSWORD=your_db_password
```

---

## 🧪 Testing API Configuration

### Test MT5 Connection
```bash
python test_mt5_connection.py
```

### Test All APIs
```bash
python setup_api_config.py --test
```

### Test Individual Components
```bash
# Test economic calendar
python -c "from src.config.settings import settings; print('FMP Key:', bool(settings.news_api_key))"

# Test Telegram bot
python -c "import os; print('Telegram configured:', bool(os.getenv('TELEGRAM_TOKEN')))"
```

---

## 🔍 Troubleshooting

### Common Issues

#### **MT5 Connection Failed**
- Verify credentials are correct
- Check if MT5 is running
- Ensure broker server is accessible
- Try demo account first

#### **API Key Invalid**
- Check for typos in .env file
- Verify API key hasn't expired
- Confirm API key has correct permissions

#### **Telegram Bot Not Working**
- Verify bot token is correct
- Check user IDs are numeric
- Ensure bot is not blocked

#### **News APIs Not Working**
- Check API key validity
- Verify internet connection
- Check API rate limits

---

## 📊 API Usage & Limits

| API | Free Tier | Paid Tier | Usage |
|-----|-----------|-----------|--------|
| FMP | 250/day | $29/month | Economic calendar |
| Twelve Data | 800/day | $9.99/month | Market data |
| EventRegistry | 10K/month | Custom | News sentiment |
| Twitter | 1.5K/month | Custom | Social sentiment |
| Telegram | Unlimited | N/A | Notifications |

---

## 🔒 Security Best Practices

1. **Never commit .env file** to version control
2. **Use strong passwords** for MT5 accounts
3. **Limit Telegram authorized users** to trusted individuals
4. **Rotate API keys** periodically
5. **Use environment variables** instead of hardcoding
6. **Enable 2FA** on all accounts where possible

---

## 🚀 Next Steps

After configuring APIs:

1. **Test basic functionality**: `python test_mt5_connection.py`
2. **Run core bot**: `python run_core_trading_bot.py`
3. **Enable advanced features**: `python run_adaptive_intelligent_bot.py`
4. **Monitor performance**: Check logs and Grafana dashboard

---

## 📞 Support

If you encounter issues:
1. Check the logs folder for error messages
2. Verify all API keys are correctly formatted
3. Test individual components
4. Review this guide for troubleshooting steps

**Your trading bot is now configured with all available APIs! 🎉**
