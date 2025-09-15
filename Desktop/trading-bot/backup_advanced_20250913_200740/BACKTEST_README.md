# Adaptive Intelligent Bot Backtesting Framework

This document explains how to use the backtesting framework for the Adaptive Intelligent Trading Bot.

## Overview

The backtesting framework allows you to simulate the Adaptive Intelligent Bot's performance on historical data before deploying it live. It includes:

- **Historical Data Loading**: Fetches historical market data from MT5
- **Simulated Broker**: Simulates trading operations without real money
- **Performance Analysis**: Comprehensive statistics and reporting
- **Risk Management**: Proper position sizing and risk controls

## Prerequisites

1. **MT5 Terminal**: Must be installed and running with market data access
2. **Python Dependencies**: All required packages must be installed
3. **Historical Data**: MT5 must have sufficient historical data for the backtest period

## Quick Start

### Basic Backtest (30 days)

```bash
python backtest_adaptive_bot.py
```

### Custom Backtest Parameters

```bash
# Backtest with $50,000 initial balance for 90 days
python backtest_adaptive_bot.py --balance 50000 --days 90

# Backtest specific date range
python backtest_adaptive_bot.py --start-date 2024-01-01 --days 60

# Full command line options
python backtest_adaptive_bot.py --balance 10000 --days 30 --start-date 2024-06-01
```

## Command Line Options

- `--balance`: Initial account balance (default: $10,000)
- `--days`: Number of days to backtest (default: 30)
- `--start-date`: Start date in YYYY-MM-DD format (optional)

## What the Backtest Does

1. **Data Loading**: Downloads historical M15, H1, and H4 data for all currency pairs
2. **Simulation**: Runs the bot's trading logic minute-by-minute through historical data
3. **Trade Execution**: Simulates opening/closing positions based on signals
4. **Risk Management**: Applies proper position sizing and stop losses
5. **Reporting**: Generates detailed performance statistics

## Output Files

### Console Output
- Real-time backtest progress
- Trade execution logs
- Final performance summary

### Log Files
- `logs/backtest_adaptive_bot.log`: Detailed execution logs

### Report Files
- `backtest_reports/backtest_report_YYYYMMDD_HHMMSS.json`: Comprehensive JSON report

## Performance Metrics

The backtest generates detailed statistics including:

### Financial Metrics
- Initial vs Final Balance
- Net Profit/Loss
- Return Percentage
- Maximum Drawdown

### Trading Statistics
- Total Trades Executed
- Win Rate
- Average Win/Loss
- Profit Factor
- Average Trades per Day

### Risk Metrics
- Maximum Drawdown
- Risk-Adjusted Returns
- Sharpe Ratio (if applicable)

## Sample Output

```
🧪 BACKTEST RESULTS SUMMARY
============================================================
Period: 2024-08-01 to 2024-08-31 (30 days)
Initial Balance: $10,000.00
Final Balance: $10,850.75
Net Profit: $850.75
Return: 8.51%
Max Drawdown: 3.25%

Trading Statistics:
Total Trades: 45
Winning Trades: 28
Losing Trades: 17
Win Rate: 62.22%
Average Win: $125.50
Average Loss: -$85.30
Profit Factor: 1.85
Avg Trades/Day: 1.5

Backtest Info:
Total Scans: 1,440
Signals Analyzed: 43,200
Trades Executed: 45
============================================================
```

## Understanding the Results

### Key Metrics to Focus On

1. **Return Percentage**: Overall profitability
2. **Win Rate**: Percentage of winning trades
3. **Profit Factor**: Gross profit / Gross loss (>1.0 is profitable)
4. **Max Drawdown**: Largest peak-to-valley decline
5. **Average Trades/Day**: Trading frequency

### Interpreting Results

- **Win Rate > 50%**: Generally positive
- **Profit Factor > 1.5**: Good risk-adjusted performance
- **Max Drawdown < 10%**: Reasonable risk control
- **Return > 5% per month**: Strong performance

## Troubleshooting

### Common Issues

1. **MT5 Connection Failed**
   - Ensure MT5 terminal is running
   - Check login credentials in environment variables
   - Verify market data availability

2. **No Historical Data**
   - Check MT5 data availability for the period
   - Try a shorter backtest period
   - Verify symbol names match MT5

3. **Memory Issues**
   - Reduce backtest period
   - Close other applications
   - Use shorter date ranges

### Error Messages

- `"Failed to initialize data manager"`: MT5 connection issue
- `"No historical data found"`: Insufficient data in MT5
- `"Reached end of historical data"`: Backtest period too long

## Advanced Usage

### Custom Analysis

The JSON report contains detailed trade history that can be analyzed further:

```python
import json
with open('backtest_reports/backtest_report_20240101_120000.json', 'r') as f:
    report = json.load(f)

# Analyze trade timing
trades = report['trades_history']
# Your analysis code here
```

### Parameter Optimization

Use the backtest results to optimize bot parameters:

- Adjust confidence thresholds
- Modify risk per trade
- Change position sizing rules
- Optimize entry/exit criteria

## Best Practices

1. **Start Small**: Begin with short backtest periods (7-30 days)
2. **Validate Data**: Ensure sufficient historical data quality
3. **Multiple Runs**: Run backtests on different market conditions
4. **Risk Management**: Always use proper position sizing
5. **Paper Trading**: Validate results with paper trading before live deployment

## Next Steps

After successful backtesting:

1. **Review Results**: Analyze performance metrics thoroughly
2. **Parameter Tuning**: Optimize bot parameters based on results
3. **Paper Trading**: Test optimized parameters in real-time paper trading
4. **Live Deployment**: Gradually increase position sizes in live trading
5. **Monitoring**: Continuously monitor live performance

## Support

For issues or questions:
1. Check the log files for detailed error messages
2. Verify MT5 terminal connectivity
3. Ensure all Python dependencies are installed
4. Review the JSON report for detailed trade information
