"""
Configuration Management Module for ML Trading System
Provides centralized, adaptive parameter management
"""
import os
import json
import logging
from typing import Dict, Any, Optional
from decimal import Decimal
from datetime import datetime

logger = logging.getLogger(__name__)

class TradingConfig:
    """Centralized configuration management with adaptive parameters"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), 'trading_config.json')
        self.config = self._load_default_config()
        self._load_config_file()
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default trading configuration"""
        return {
            # Risk Management
            'risk_management': {
                'max_position_size': 0.02,  # 2% of account
                'max_daily_loss': 0.05,     # 5% daily loss limit
                'max_drawdown': 0.15,       # 15% max drawdown
                'kelly_fraction': 0.25,     # Conservative Kelly fraction
                'risk_free_rate': 0.02,     # 2% annual risk-free rate
                'position_sizing_method': 'kelly_criterion'
            },
            
            # ML Model Parameters
            'ml_model': {
                'lookback_period': 100,     # Data points for training
                'feature_window': 20,       # Feature calculation window
                'prediction_horizon': 5,    # Bars ahead to predict
                'train_test_split': 0.8,    # Training data ratio
                'cv_folds': 5,              # Cross-validation folds
                'min_samples_for_training': 200,
                'retrain_frequency': 24     # Hours between retraining
            },
            
            # Technical Indicators
            'indicators': {
                'ema_periods': [10, 20, 50],
                'rsi_period': 14,
                'bollinger_period': 20,
                'bollinger_std': 2,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'stoch_k': 14,
                'stoch_d': 3
            },
            
            # Signal Generation
            'signals': {
                'min_signal_strength': 0.6,     # Minimum signal confidence
                'signal_timeout': 10,            # Bars until signal expires
                'confluence_threshold': 3,       # Min indicators for signal
                'volatility_filter': True,      # Filter by volatility
                'trend_filter': True,           # Filter by trend
                'volume_filter': True           # Filter by volume
            },
            
            # Trading Parameters
            'trading': {
                'symbol': 'EURUSD',
                'timeframe': 'M5',              # 5-minute timeframe
                'spread_limit': 2.0,            # Max spread in pips
                'slippage_limit': 1.0,          # Max slippage in pips
                'min_trade_amount': 0.01,       # Minimum lot size
                'max_trade_amount': 1.0,        # Maximum lot size
                'stop_loss_pips': 20,           # Default stop loss
                'take_profit_pips': 40,         # Default take profit
                'trailing_stop': True,          # Enable trailing stop
                'trailing_step': 5              # Trailing step in pips
            },
            
            # Data Management
            'data': {
                'data_retention_days': 90,      # Days to keep historical data
                'data_validation_threshold': 0.05,  # Max missing data ratio
                'outlier_threshold': 3.0,       # Standard deviations for outliers
                'correlation_threshold': 0.95,  # Max feature correlation
                'feature_stability_window': 50, # Window for stability check
                'min_data_quality_score': 0.8   # Minimum data quality
            },
            
            # Monitoring
            'monitoring': {
                'performance_window': 50,       # Trades for performance calc
                'alert_threshold_pnl': -0.03,   # -3% for PnL alert
                'alert_threshold_drawdown': 0.1, # 10% for drawdown alert
                'alert_threshold_accuracy': 0.4, # 40% accuracy alert
                'health_check_interval': 300,   # 5 minutes
                'log_level': 'INFO',
                'enable_email_alerts': False,
                'enable_telegram_alerts': False
            },
            
            # MT5 Connection
            'mt5': {
                'server': '',
                'login': 0,
                'password': '',
                'timeout': 10000,              # Connection timeout
                'retry_attempts': 3,           # Number of retries
                'retry_delay': 1,              # Seconds between retries
                'enable_real_trading': False   # Safety flag
            }
        }
    
    def _load_config_file(self):
        """Load configuration from file if it exists"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                self._merge_config(self.config, file_config)
                logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.warning(f"Could not load config file: {e}. Using defaults.")
    
    def _merge_config(self, default: Dict, custom: Dict):
        """Recursively merge custom config into default config"""
        for key, value in custom.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Could not save config file: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'risk_management.max_position_size')"""
        keys = key_path.split('.')
        current = self.config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        current = self.config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        logger.info(f"Configuration updated: {key_path} = {value}")
    
    def update_adaptive_params(self, market_data: Dict[str, Any]):
        """Update parameters based on market conditions"""
        try:
            # Adapt position sizing based on volatility
            volatility = market_data.get('volatility', 0.01)
            if volatility > 0.03:  # High volatility
                self.set('risk_management.max_position_size', 0.01)  # Reduce position size
                self.set('trading.stop_loss_pips', 30)  # Wider stops
            elif volatility < 0.005:  # Low volatility
                self.set('risk_management.max_position_size', 0.025)  # Increase position size
                self.set('trading.stop_loss_pips', 15)  # Tighter stops
            
            # Adapt signal thresholds based on market regime
            trend_strength = market_data.get('trend_strength', 0.5)
            if trend_strength > 0.8:  # Strong trend
                self.set('signals.min_signal_strength', 0.5)  # Lower threshold
            else:  # Ranging market
                self.set('signals.min_signal_strength', 0.7)  # Higher threshold
                
            logger.info("Adaptive parameters updated based on market conditions")
            
        except Exception as e:
            logger.error(f"Error updating adaptive parameters: {e}")
    
    def validate_config(self) -> bool:
        """Validate configuration values"""
        try:
            # Check risk management parameters
            max_pos = self.get('risk_management.max_position_size')
            if not (0 < max_pos <= 0.1):
                raise ValueError(f"Invalid max_position_size: {max_pos}")
            
            # Check ML parameters
            lookback = self.get('ml_model.lookback_period')
            if not (10 <= lookback <= 1000):
                raise ValueError(f"Invalid lookback_period: {lookback}")
            
            # Check trading parameters
            symbol = self.get('trading.symbol')
            if not symbol:
                raise ValueError("Trading symbol not specified")
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

# Global configuration instance
config = TradingConfig()