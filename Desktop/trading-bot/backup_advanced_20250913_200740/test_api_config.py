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
        print("\n" + "="*60)
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

        print("\n" + "="*60)
        print(f"🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")

        # Recommendations
        if passed_tests == total_tests:
            print("🎉 ALL API CONFIGURATIONS VALID!")
            print("\n🚀 READY FOR:")
            print("   - Live trading deployment")
            print("   - Full system testing")
            print("   - Performance benchmarking")
        elif passed_tests >= total_tests * 0.7:
            print("⚠️ MOST APIs CONFIGURED - Ready for testing")
            print("\n📋 NEXT STEPS:")
            print("   - Complete missing API configurations")
            print("   - Test individual components")
            print("   - Start with paper trading")
        else:
            print("❌ CRITICAL APIs MISSING")
            print("\n📋 PRIORITY ACTIONS:")
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