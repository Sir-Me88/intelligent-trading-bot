# 🚀 QUICK START GUIDE - Trading Bot

## ✅ System Status: OPERATIONAL
All core systems tested and working! Ready for live trading.

## 🔧 Setup Steps (5 minutes)

### Step 1: Configure MT5 Connection
```bash
# 1. Install MetaTrader 5 if not already installed
# 2. Open MT5 and create/login to account
# 3. Note your credentials

# 4. Edit .env file with your MT5 credentials:
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password  
MT5_SERVER=your_broker_server
```

### Step 2: Test Connection
```bash
python setup_mt5_connection.py
```

### Step 3: Run Live Test
```bash
python run_simple_live_test.py
```

### Step 4: Start Trading Bot
```bash
python start_bot.py
```

## 📊 What's Working

✅ **Core Systems (100% Functional)**
- Market data processing
- Technical analysis (RSI, MACD, Bollinger Bands)
- Signal generation
- Risk management
- Position tracking
- Multi-timeframe analysis
- Correlation analysis
- Performance monitoring

✅ **Ready for Live Trading**
- MT5 broker integration
- Real-time market data
- Automated signal generation
- Risk validation
- Position management

## ⚠️ Optional Enhancements

These are NOT required for basic trading:
- News sentiment analysis (needs API keys)
- Advanced ML features (needs torch fix)
- Telegram notifications (needs bot token)

## 🎯 Trading Strategy

The bot uses:
- **Multi-timeframe analysis** (15M + 1H)
- **Technical indicators**: RSI, MACD, Bollinger Bands, ATR
- **Risk management**: 1% risk per trade, max 5 positions
- **Adaptive parameters**: Adjusts to market volatility
- **Correlation filtering**: Avoids correlated trades

## 🛡️ Safety Features

- **Risk validation** before every trade
- **Position size calculation** based on account equity
- **Stop loss** on every trade
- **Maximum drawdown protection**
- **Circuit breaker** for unusual market conditions

## 📈 Monitoring

- Real-time performance metrics
- Trade logging and analysis
- Heartbeat monitoring
- Error tracking and recovery

## 🚀 Next Steps

1. **Configure MT5** (required)
2. **Test with paper trading** (recommended)
3. **Start with small position sizes** (recommended)
4. **Monitor performance** (essential)
5. **Add API keys for news** (optional)

## 💡 Tips

- Start with demo account first
- Use small position sizes initially
- Monitor the logs/adaptive_intelligent_bot.log file
- Check heartbeat file for system status
- The bot runs 24/7 during market hours

## 🆘 Support

If you encounter issues:
1. Check the logs folder for error messages
2. Verify MT5 connection with setup script
3. Ensure market is open for testing
4. Run test_working_systems.py to verify components

**Your trading bot is ready to trade! 🎉**