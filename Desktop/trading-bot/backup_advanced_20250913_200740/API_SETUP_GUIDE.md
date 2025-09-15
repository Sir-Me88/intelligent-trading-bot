# 2025 TRADING BOT - API SETUP GUIDE
# ======================================

## Required APIs for Full Functionality

### 1. MetaTrader 5 (Required for Live Trading)
- **Purpose**: Live forex trading execution
- **Setup**: https://www.metatrader5.com/en/terminal/help/start_advanced/start
- **Demo Account**: Available for testing
- **Configuration**:
  ```
  MT5_LOGIN=your_demo_login
  MT5_PASSWORD=your_demo_password
  MT5_SERVER=MetaQuotes-Demo
  ```

### 2. FinGPT v3.1 (Required for Advanced Sentiment)
- **Purpose**: Latest financial language model for sentiment analysis
- **Setup**: https://github.com/AI4Finance-Foundation/FinGPT
- **Benefits**: 5-7% accuracy improvement on volatile pairs
- **Configuration**:
  ```
  FINGPT_API_KEY=your_api_key
  FINGPT_MODEL=AI4Finance-Foundation/FinGPT-v3.1-forex-sentiment
  ```

### 3. News APIs (Recommended for Sentiment)
- **Financial Modeling Prep**: https://financialmodelingprep.com/
- **Alpha Vantage**: https://www.alphavantage.co/
- **Twitter API**: https://developer.twitter.com/

### 4. Monitoring (Optional but Recommended)
- **Prometheus**: https://prometheus.io/
- **Grafana**: https://grafana.com/
- **Telegram Bot**: For alerts and notifications

## Quick Start Configuration

### For Testing (Demo Mode):
```bash
# Copy template
cp .env.template .env

# Edit with your demo credentials
nano .env

# Test configuration
python test_api_config.py
```

### For Production:
```bash
# Use live credentials
# Enable all monitoring
# Configure backup systems
# Set up alerts
```

## API Key Security Best Practices

1. **Never commit .env to version control**
2. **Use environment-specific keys** (dev/staging/prod)
3. **Rotate keys quarterly**
4. **Monitor API usage** to prevent rate limits
5. **Use encrypted storage** for sensitive keys

## Testing Your Configuration

```bash
# Test all APIs
python test_api_config.py

# Test specific components
python test_fingpt_connection.py
python test_mt5_connection.py
python test_news_apis.py
```

## Troubleshooting

### Common Issues:
- **MT5 Connection Failed**: Check demo server credentials
- **FinGPT API Error**: Verify API key and model name
- **Rate Limits**: Implement exponential backoff
- **Network Issues**: Configure proxy settings if needed

### Support:
- Check logs in `logs/` directory
- Monitor Grafana dashboard
- Review API documentation links above

---
*Generated for 2025 Advanced Trading Bot*
*Last updated: 2025-09-12*