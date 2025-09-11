class RiskManager:
    def __init__(self, market_data_manager=None):
        self.market_data_manager = market_data_manager

    async def validate_trade(self, signal, account_info, current_positions):
        """Validate a trade signal based on risk management rules."""
        # Basic validation - check account equity
        equity = account_info.get('equity', 0)
        if equity < 1000:
            return {
                'approved': False,
                'reason': 'Invalid account equity'
            }

        # Check maximum positions
        if len(current_positions) >= 5:
            return {
                'approved': False,
                'reason': 'Maximum positions reached'
            }

        # Basic position size calculation (simplified)
        position_size = 0.1  # Fixed size for testing
        risk_amount = 50  # Fixed risk for testing

        return {
            'approved': True,
            'position_size': position_size,
            'risk_amount': risk_amount
        }
