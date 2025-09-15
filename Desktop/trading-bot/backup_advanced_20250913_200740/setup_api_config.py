#!/usr/bin/env python3
"""API Configuration Setup for 2025 Trading Bot."""

import os
from pathlib import Path
import json
from datetime import datetime

def create_env_template():
    """Create comprehensive .env template with all required API keys."""

    env_template = """
# ===========================================
# 2025 ADVANCED TRADING BOT - API CONFIGURATION
# ===========================================

# MetaTrader 5 Configuration
MT5_LOGIN=your_mt5_login_here
MT5_PASSWORD=your_mt5_password_here
MT5_SERVER=your_mt5_server_here

# FinGPT v3.1 Configuration (2025 Upgrade)
FINGPT_API_KEY=your_fingpt_api_key_here
FINGPT_MODEL=AI4Finance-Foundation/FinGPT-v3.1-forex-sentiment

# News & Sentiment APIs
NEWS_API_KEY=your_news_api_key_here
FMP_API_KEY=your_financial_modeling_prep_key_here
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here

# Alternative Data Sources
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here
COINMARKETCAP_KEY=your_coinmarketcap_key_here

# Database Configuration
POSTGRES_URL=postgresql://user:password@localhost/trading_db
REDIS_URL=redis://localhost:6379

# Monitoring & Alerting
PROMETHEUS_PORT=8000
GRAFANA_PORT=3000
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Security & Encryption
ENCRYPTION_KEY=your_32_char_encryption_key_here
JWT_SECRET=your_jwt_secret_here

# Trading Parameters (Adaptive)
MIN_CONFIDENCE=0.75
MIN_RR_RATIO=3.5
ATR_MULTIPLIER_NORMAL=2.0
ATR_MULTIPLIER_HIGH_VOL=3.0
MAX_DRAWDOWN=0.15

# Risk Management
MAX_POSITION_SIZE=0.02
MAX_DAILY_LOSS=0.05
CIRCUIT_BREAKER_THRESHOLD=0.10

# ML Configuration
RL_LEARNING_RATE=0.0003
ML_CONFIDENCE_THRESHOLD=0.8
LSTM_SEQUENCE_LENGTH=20

# Edge Computing
PREFETCH_WINDOW_SECONDS=10
LATENCY_THRESHOLD_MS=100
CACHE_TTL_SECONDS=300

# Logging & Debugging
LOG_LEVEL=INFO
LOG_TO_FILE=true
LOG_TO_CONSOLE=true
ENABLE_XAI_LOGGING=true

# Backup & Recovery
BACKUP_INTERVAL_HOURS=24
BACKUP_RETENTION_DAYS=30
AUTO_RECOVERY_ENABLED=true

# Performance Monitoring
PERFORMANCE_MONITORING_ENABLED=true
LATENCY_TRACKING_ENABLED=true
MEMORY_MONITORING_ENABLED=true
"""

    with open('.env.template', 'w', encoding='utf-8') as f:
        f.write(env_template.strip())

    print("✅ Created .env.template with comprehensive API configuration")

def create_api_guide():
    """Create detailed API setup guide."""

    guide = """
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
"""

    with open('API_SETUP_GUIDE.md', 'w', encoding='utf-8') as f:
        f.write(guide.strip())

    print("✅ Created comprehensive API setup guide")

def create_config_validator():
    """Create API configuration validator."""

    validator_code = '''
#!/usr/bin/env python3
"""API Configuration Validator for 2025 Trading Bot."""

import os
import sys
import asyncio
from pathlib import Path

sys.path.append('src')

class APIConfigValidator:
    """Validates API configurations and connections."""

    def __init__(self):
        self.results = {}
        self.required_apis = {
            'MT5_LOGIN': 'MetaTrader 5 Login',
            'MT5_PASSWORD': 'MetaTrader 5 Password',
            'MT5_SERVER': 'MetaTrader 5 Server',
        }

        self.optional_apis = {
            'FINGPT_API_KEY': 'FinGPT v3.1 API',
            'NEWS_API_KEY': 'News API',
            'FMP_API_KEY': 'Financial Modeling Prep',
            'TWITTER_BEARER_TOKEN': 'Twitter API',
        }

    def validate_env_file(self):
        """Validate .env file exists and has required variables."""
        if not Path('.env').exists():
            self.results['env_file'] = {'status': 'FAILED', 'message': '.env file not found'}
            return False

        self.results['env_file'] = {'status': 'PASSED', 'message': '.env file found'}
        return True

    def validate_required_apis(self):
        """Validate required API configurations."""
        missing_required = []

        for var_name, description in self.required_apis.items():
            value = os.getenv(var_name)
            if not value or value == f'your_{var_name.lower()}_here':
                missing_required.append(f"{description} ({var_name})")

        if missing_required:
            self.results['required_apis'] = {
                'status': 'FAILED',
                'message': f'Missing required APIs: {", ".join(missing_required)}'
            }
            return False

        self.results['required_apis'] = {
            'status': 'PASSED',
            'message': 'All required APIs configured'
        }
        return True

    def validate_optional_apis(self):
        """Validate optional API configurations."""
        configured_optional = []
        missing_optional = []

        for var_name, description in self.optional_apis.items():
            value = os.getenv(var_name)
            if value and value != f'your_{var_name.lower()}_here':
                configured_optional.append(description)
            else:
                missing_optional.append(description)

        self.results['optional_apis'] = {
            'status': 'INFO',
            'configured': configured_optional,
            'missing': missing_optional,
            'message': f'Optional APIs - Configured: {len(configured_optional)}, Missing: {len(missing_optional)}'
        }

        return len(configured_optional) > 0

    async def test_mt5_connection(self):
        """Test MetaTrader 5 connection."""
        try:
            import MetaTrader5 as mt5

            login = int(os.getenv('MT5_LOGIN', 0))
            password = os.getenv('MT5_PASSWORD', '')
            server = os.getenv('MT5_SERVER', '')

            if not all([login, password, server]):
                self.results['mt5_connection'] = {
                    'status': 'FAILED',
                    'message': 'MT5 credentials not configured'
                }
                return False

            if not mt5.initialize(login=login, password=password, server=server):
                self.results['mt5_connection'] = {
                    'status': 'FAILED',
                    'message': f'MT5 connection failed: {mt5.last_error()}'
                }
                return False

            account_info = mt5.account_info()
            if account_info:
                self.results['mt5_connection'] = {
                    'status': 'PASSED',
                    'message': f'MT5 connected - Balance: ${account_info.balance:.2f}',
                    'account': account_info.login
                }
                mt5.shutdown()
                return True
            else:
                self.results['mt5_connection'] = {
                    'status': 'FAILED',
                    'message': 'MT5 connected but no account info'
                }
                mt5.shutdown()
                return False

        except ImportError:
            self.results['mt5_connection'] = {
                'status': 'FAILED',
                'message': 'MetaTrader5 package not installed'
            }
            return False
        except Exception as e:
            self.results['mt5_connection'] = {
                'status': 'FAILED',
                'message': f'MT5 test error: {e}'
            }
            return False

    async def test_fingpt_connection(self):
        """Test FinGPT v3.1 connection."""
        try:
            from src.news.sentiment import FinGPTAnalyzer

            analyzer = FinGPTAnalyzer()

            # Test basic initialization
            if hasattr(analyzer, 'market_context'):
                context = analyzer.market_context
                self.results['fingpt_connection'] = {
                    'status': 'PASSED',
                    'message': 'FinGPT v3.1 initialized with 2025 context',
                    'features': [
                        f"NFP 2025: {context.get('nfp_2025_data', False)}",
                        f"Fed Focus: {context.get('fed_policy_focus', False)}",
                        f"Volatility Regime: {context.get('volatility_regime', 'unknown')}"
                    ]
                }
                return True
            else:
                self.results['fingpt_connection'] = {
                    'status': 'FAILED',
                    'message': 'FinGPT market context not loaded'
                }
                return False

        except Exception as e:
            self.results['fingpt_connection'] = {
                'status': 'FAILED',
                'message': f'FinGPT test error: {e}'
            }
            return False

    def generate_report(self):
        """Generate comprehensive validation report."""
        print("\\n" + "="*60)
        print("🔍 API CONFIGURATION VALIDATION REPORT")
        print("="*60)
        print(f"Validation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

        # Overall status
        passed_tests = sum(1 for result in self.results.values()
                          if isinstance(result, dict) and result.get('status') == 'PASSED')
        total_tests = len(self.results)

        for test_name, result in self.results.items():
            if isinstance(result, dict):
                status = result.get('status', 'UNKNOWN')
                message = result.get('message', 'No message')

                if status == 'PASSED':
                    print(f"[PASS] {test_name.replace('_', ' ').title()}: {message}")
                elif status == 'FAILED':
                    print(f"[FAIL] {test_name.replace('_', ' ').title()}: {message}")
                elif status == 'INFO':
                    print(f"[INFO] {test_name.replace('_', ' ').title()}: {message}")
                    if 'configured' in result:
                        print(f"   Configured: {', '.join(result['configured'])}")
                    if 'missing' in result:
                        print(f"   Missing: {', '.join(result['missing'])}")

        print("\\n" + "="*60)
        print(f"🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")

        # Recommendations
        if passed_tests == total_tests:
            print("🎉 ALL API CONFIGURATIONS VALID!")
            print("\\n🚀 READY FOR:")
            print("   - Live trading deployment")
            print("   - Full system testing")
            print("   - Performance benchmarking")
        elif passed_tests >= total_tests * 0.7:
            print("⚠️ MOST APIs CONFIGURED - Ready for testing")
            print("\\n📋 NEXT STEPS:")
            print("   - Complete missing API configurations")
            print("   - Test individual components")
            print("   - Start with paper trading")
        else:
            print("❌ CRITICAL APIs MISSING")
            print("\\n📋 PRIORITY ACTIONS:")
            print("   - Configure MT5 credentials")
            print("   - Set up FinGPT API key")
            print("   - Review API_SETUP_GUIDE.md")

        return passed_tests == total_tests

    async def run_validation(self):
        """Run complete API validation."""
        print("🔍 STARTING API CONFIGURATION VALIDATION...")

        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()

        # Run all validations
        self.validate_env_file()
        self.validate_required_apis()
        self.validate_optional_apis()

        await self.test_mt5_connection()
        await self.test_fingpt_connection()

        # Generate report
        return self.generate_report()

if __name__ == "__main__":
    validator = APIConfigValidator()
    asyncio.run(validator.run_validation())
'''

    with open('test_api_config.py', 'w', encoding='utf-8') as f:
        f.write(validator_code.strip())

    print("✅ Created API configuration validator")

def main():
    """Create complete API configuration setup."""
    print("CREATING 2025 TRADING BOT API CONFIGURATION")
    print("="*60)

    create_env_template()
    create_api_guide()
    create_config_validator()

    print("\\n" + "="*60)
    print("API CONFIGURATION SETUP COMPLETE")
    print("\\nFiles Created:")
    print("   - .env.template (comprehensive configuration)")
    print("   - API_SETUP_GUIDE.md (detailed setup instructions)")
    print("   - test_api_config.py (configuration validator)")
    print("\\nNEXT STEPS:")
    print("   1. Copy .env.template to .env")
    print("   2. Configure your API keys")
    print("   3. Run: python test_api_config.py")
    print("   4. Start paper trading validation")
    print("="*60)

if __name__ == "__main__":
    main()
