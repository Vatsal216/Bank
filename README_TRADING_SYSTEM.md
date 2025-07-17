# Enhanced ML Trading System Documentation

## Overview

This project implements a comprehensive Machine Learning trading system with institutional-grade features, integrated into the existing Django banking application. The system provides advanced risk management, real-time monitoring, and adaptive configuration capabilities.

## Core Features

### 1. Configuration Management (`config.py`)
- **Centralized Configuration**: All trading parameters managed from a single source
- **Adaptive Parameters**: Automatically adjusts parameters based on market conditions
- **Validation**: Built-in configuration validation to prevent invalid settings
- **Persistence**: Configuration saved to JSON file for persistence

### 2. Risk Management (`risk_management.py`)
- **Kelly Criterion**: Optimal position sizing based on historical performance
- **Dynamic Position Sizing**: Adjusts position size based on signal strength and volatility
- **Risk Controls**: Daily loss limits, maximum drawdown protection
- **Stop Loss/Take Profit**: Dynamic calculation based on market volatility

### 3. Performance Monitoring (`monitoring.py`)
- **Real-time Monitoring**: Continuous system health and performance tracking
- **Alerting System**: Email and Telegram alerts for critical events
- **Performance Metrics**: Win rate, Sharpe ratio, profit factor, drawdown tracking
- **System Health**: Memory, CPU, connection status monitoring

### 4. ML Trading System (`ML_EMA_10.py`)
- **Data Management**: Robust data validation and cleaning
- **Feature Engineering**: 23+ technical indicators with correlation filtering
- **ML Models**: Random Forest and Gradient Boosting with time-series cross-validation
- **Signal Generation**: Multi-factor signal scoring with confluence requirements

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Django Setup
```bash
python manage.py check
python manage.py migrate  # If needed
```

### 3. Run Demo
```bash
python demo_trading_system.py
```

## API Endpoints

The system provides RESTful API endpoints for integration:

### Trading System Status
```
GET /api/core/trading/status/
```
Returns current system status, positions, and metrics.

### Configuration Management
```
GET /api/core/trading/config/
POST /api/core/trading/config/
```
Get or update trading configuration parameters.

### Risk Metrics
```
GET /api/core/trading/risk-metrics/
```
Returns current risk metrics including account balance, daily PnL, drawdown.

### Performance Metrics
```
GET /api/core/trading/performance/
```
Returns performance metrics including win rate, Sharpe ratio, profit factor.

### System Health
```
GET /api/core/trading/health/
```
Returns system health status including memory, CPU, connection status.

### Trading Control
```
POST /api/core/trading/control/
```
Start or stop the trading system.

## Configuration Parameters

### Risk Management
- `max_position_size`: Maximum position size as fraction of account (default: 0.02)
- `max_daily_loss`: Maximum daily loss limit (default: 0.05)
- `max_drawdown`: Maximum drawdown limit (default: 0.15)
- `kelly_fraction`: Kelly Criterion safety multiplier (default: 0.25)

### ML Model
- `lookback_period`: Data points for training (default: 100)
- `prediction_horizon`: Bars ahead to predict (default: 5)
- `cv_folds`: Cross-validation folds (default: 5)
- `retrain_frequency`: Hours between retraining (default: 24)

### Signal Generation
- `min_signal_strength`: Minimum signal confidence (default: 0.6)
- `confluence_threshold`: Minimum indicators for signal (default: 3)
- `volatility_filter`: Enable volatility filtering (default: true)

### Trading Parameters
- `symbol`: Trading symbol (default: 'EURUSD')
- `timeframe`: Chart timeframe (default: 'M5')
- `stop_loss_pips`: Default stop loss in pips (default: 20)
- `take_profit_pips`: Default take profit in pips (default: 40)

## Usage Examples

### 1. Basic Configuration
```python
from core.config import config

# Get current configuration
symbol = config.get('trading.symbol')
max_pos = config.get('risk_management.max_position_size')

# Update configuration
config.set('trading.symbol', 'GBPUSD')
config.set('risk_management.max_position_size', 0.015)
config.save_config()
```

### 2. Risk Management
```python
from core.risk_management import risk_manager

# Calculate position size
signal_strength = 0.8
volatility = 0.015
position_size = risk_manager.calculate_position_size(signal_strength, volatility)

# Calculate stop loss and take profit
entry_price = 1.2500
stop_loss = risk_manager.calculate_stop_loss(entry_price, 'BUY', volatility)
take_profit = risk_manager.calculate_take_profit(entry_price, 'BUY', stop_loss)
```

### 3. Performance Monitoring
```python
from core.monitoring import performance_monitor

# Add trade result
trade_result = {
    'symbol': 'EURUSD',
    'position_type': 'BUY',
    'pnl_amount': 25.50,
    'pnl_percent': 0.015,
    'duration_minutes': 45,
    'signal_strength': 0.8
}
performance_monitor.add_trade_result(trade_result)

# Get performance metrics
metrics = performance_monitor.get_performance_metrics()
```

### 4. Signal Generation
```python
from core.ML_EMA_10 import SignalGenerator, DataManager

# Initialize components
signal_generator = SignalGenerator()
data_manager = DataManager()

# Get data and generate signal
data = data_manager.get_market_data('EURUSD', 'M5', 200)
features = signal_generator.feature_engineer.create_features(data)
signal_result = signal_generator.generate_signal(features)
```

## Key Algorithms

### 1. Kelly Criterion Position Sizing
```
f = (bp - q) / b
where:
- f = fraction of capital to risk
- b = average win / average loss
- p = win rate
- q = 1 - win rate
```

### 2. Multi-Factor Signal Scoring
The system combines multiple technical indicators with ML predictions:
- EMA crossovers
- RSI overbought/oversold
- MACD signals
- Bollinger Band positions
- Volume confirmation
- ML model predictions

### 3. Time-Series Cross-Validation
Uses `TimeSeriesSplit` to prevent data leakage and ensure realistic model validation.

### 4. Adaptive Parameter Updates
Automatically adjusts parameters based on:
- Market volatility
- Trend strength
- Volume patterns

## Testing

Run the comprehensive test suite:
```bash
python core/test_trading_system.py
```

Tests cover:
- Configuration management
- Risk management calculations
- Data validation and cleaning
- Feature engineering
- ML model training and prediction
- Signal generation
- Performance monitoring

## Production Deployment

### 1. Environment Variables
Set the following environment variables:
- `DJANGO_SETTINGS_MODULE=bank.settings`
- `DATABASE_URL=postgresql://...`

### 2. MT5 Integration
For live trading, configure MT5 connection parameters:
```python
config.set('mt5.server', 'YourBroker-Server')
config.set('mt5.login', your_login)
config.set('mt5.password', 'your_password')
config.set('mt5.enable_real_trading', True)  # Only when ready for live trading
```

### 3. Monitoring Setup
Configure email and Telegram alerts:
```python
config.set('monitoring.enable_email_alerts', True)
config.set('monitoring.email_smtp_server', 'smtp.gmail.com')
config.set('monitoring.email_user', 'your_email@gmail.com')
config.set('monitoring.email_password', 'your_app_password')
config.set('monitoring.email_to', 'alerts@yourcompany.com')
```

### 4. Background Service
For production, run the trading system as a background service:
```bash
# Using systemd
sudo systemctl enable trading-system
sudo systemctl start trading-system

# Using supervisor
supervisorctl start trading-system
```

## Security Considerations

1. **API Security**: Implement proper authentication for API endpoints
2. **Configuration Security**: Store sensitive configuration in environment variables
3. **Database Security**: Use encrypted connections to the database
4. **Monitoring**: Implement intrusion detection and monitoring
5. **Backup**: Regular backups of configuration and trade history

## Performance Optimization

1. **Database Indexing**: Add indexes for frequently queried trade data
2. **Caching**: Implement Redis caching for market data
3. **Async Processing**: Use Celery for background tasks
4. **Data Retention**: Implement data archival policies

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Configuration Errors**: Check configuration validation
3. **Data Quality Issues**: Monitor data quality scores
4. **Performance Issues**: Check system health metrics

### Logs
Check the following log files:
- `trading_system.log`: Main trading system logs
- Django logs: Check Django application logs

### Support
For issues or questions, check:
1. Test suite results
2. Demo script output
3. API endpoint responses
4. System health metrics

## License

This enhanced ML trading system is part of the Bank application and follows the same licensing terms.