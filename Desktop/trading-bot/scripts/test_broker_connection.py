#!/usr/bin/env python3
"""Test MT5 broker connection for Phase 3 deployment."""

import asyncio
import os
from src.trading.broker_interface import BrokerManager

async def test_broker_connection():
    """Test broker connection."""
    print("🔗 Testing MT5 Broker Connection...")
    print("=" * 50)

    try:
        # Test broker connection
        broker = BrokerManager()
        success = await broker.initialize()

        if success:
            print("✅ MT5 Broker connection successful")
            account_info = await broker.get_account_info()
            balance = account_info.get('balance', 0)
            equity = account_info.get('equity', 0)
            print(f"   Account Balance: ${balance:.2f}")
            print(f"   Account Equity: ${equity:.2f}")
            print("✅ Ready for live deployment")
            return True
        else:
            print("❌ MT5 Broker connection failed")
            print("   Please check MT5 credentials in .env file")
            return False

    except Exception as e:
        print(f"❌ Error testing broker connection: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_broker_connection())
    exit(0 if result else 1)
