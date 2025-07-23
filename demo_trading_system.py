#!/usr/bin/env python3
"""
ML Trading System Demo Script
Demonstrates the enhanced trading system capabilities
"""
import os
import sys
import time
import json
from datetime import datetime

# Add Django setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'bank.settings')

import django
django.setup()

from core.config import config
from core.risk_management import risk_manager
from core.monitoring import performance_monitor
from core.ML_EMA_10 import MLTradingSystem, DataManager, FeatureEngineer, SignalGenerator

def demo_configuration_management():
    """Demonstrate configuration management"""
    print("=" * 60)
    print("1. CONFIGURATION MANAGEMENT DEMO")
    print("=" * 60)
    
    # Show current configuration
    print(f"Current symbol: {config.get('trading.symbol')}")
    print(f"Current max position size: {config.get('risk_management.max_position_size')}")
    print(f"Current ML lookback period: {config.get('ml_model.lookback_period')}")
    
    # Update configuration
    print("\nUpdating configuration...")
    config.set('trading.symbol', 'GBPUSD')
    config.set('risk_management.max_position_size', 0.015)
    
    print(f"Updated symbol: {config.get('trading.symbol')}")
    print(f"Updated max position size: {config.get('risk_management.max_position_size')}")
    
    # Validate configuration
    is_valid = config.validate_config()
    print(f"Configuration validation: {'PASSED' if is_valid else 'FAILED'}")

def demo_risk_management():
    """Demonstrate risk management features"""
    print("\n" + "=" * 60)
    print("2. RISK MANAGEMENT DEMO")
    print("=" * 60)
    
    # Show current risk metrics
    metrics = risk_manager.get_risk_metrics()
    print(f"Account balance: ${metrics['account_balance']:,.2f}")
    print(f"Daily PnL: ${metrics['daily_pnl']:,.2f}")
    print(f"Max drawdown: {metrics['max_drawdown']:.2f}%")
    
    # Test Kelly Criterion position sizing
    print("\nTesting Kelly Criterion position sizing:")
    win_rate = 0.65
    avg_win = 0.025
    avg_loss = 0.015
    
    position_size = risk_manager.calculate_kelly_position_size(win_rate, avg_win, avg_loss)
    print(f"Kelly position size for {win_rate:.0%} win rate: {position_size:.4f}")
    
    # Test dynamic position sizing
    signal_strength = 0.8
    market_volatility = 0.015
    
    dynamic_size = risk_manager.calculate_position_size(signal_strength, market_volatility)
    print(f"Dynamic position size (strength: {signal_strength}, vol: {market_volatility}): {dynamic_size:.4f}")
    
    # Test stop loss calculation
    entry_price = 1.2500
    stop_loss = risk_manager.calculate_stop_loss(entry_price, 'BUY', market_volatility)
    take_profit = risk_manager.calculate_take_profit(entry_price, 'BUY', stop_loss)
    
    print(f"For BUY at {entry_price:.5f}:")
    print(f"  Stop Loss: {stop_loss:.5f}")
    print(f"  Take Profit: {take_profit:.5f}")
    print(f"  Risk/Reward: {(take_profit - entry_price) / (entry_price - stop_loss):.2f}")

def demo_data_management():
    """Demonstrate data management and feature engineering"""
    print("\n" + "=" * 60)
    print("3. DATA MANAGEMENT & FEATURE ENGINEERING DEMO")
    print("=" * 60)
    
    # Initialize data manager and feature engineer
    data_manager = DataManager()
    feature_engineer = FeatureEngineer()
    
    # Get market data
    print("Retrieving market data...")
    symbol = config.get('trading.symbol', 'EURUSD')
    data = data_manager.get_market_data(symbol, 'M5', 200)
    
    print(f"Retrieved {len(data)} bars of {symbol} data")
    print(f"Data range: {data.index[0]} to {data.index[-1]}")
    print(f"Price range: {data['close'].min():.5f} - {data['close'].max():.5f}")
    
    # Create features
    print("\nEngineering features...")
    features = feature_engineer.create_features(data)
    
    print(f"Created {len(features.columns)} total columns")
    print(f"Feature columns: {len(feature_engineer.feature_columns)} stable features")
    
    # Show some feature statistics
    if 'ema_10' in features.columns:
        print(f"EMA 10 current value: {features['ema_10'].iloc[-1]:.5f}")
    if 'rsi' in features.columns:
        print(f"RSI current value: {features['rsi'].iloc[-1]:.2f}")
    if 'volatility' in features.columns:
        print(f"Current volatility: {features['volatility'].iloc[-1]:.4f}")

def demo_signal_generation():
    """Demonstrate signal generation"""
    print("\n" + "=" * 60)
    print("4. SIGNAL GENERATION DEMO")
    print("=" * 60)
    
    # Initialize signal generator
    signal_generator = SignalGenerator()
    data_manager = DataManager()
    
    # Get data and create features
    symbol = config.get('trading.symbol', 'EURUSD')
    data = data_manager.get_market_data(symbol, 'M5', 200)
    features = signal_generator.feature_engineer.create_features(data)
    
    if len(features) > 100:
        # Generate signal
        print("Generating trading signal...")
        signal_result = signal_generator.generate_signal(features)
        
        print(f"Signal: {signal_result['signal']} ({'BUY' if signal_result['signal'] > 0 else 'SELL' if signal_result['signal'] < 0 else 'NEUTRAL'})")
        print(f"Signal strength: {signal_result['strength']:.3f}")
        print(f"Bullish factors: {signal_result.get('bullish_factors', 0)}")
        print(f"Bearish factors: {signal_result.get('bearish_factors', 0)}")
        
        # Show individual signal factors
        factors = signal_result.get('factors', {})
        print("\nSignal factors:")
        for factor, value in factors.items():
            if isinstance(value, (int, float)):
                print(f"  {factor}: {value:.3f}")
    else:
        print("Not enough data for signal generation")

def demo_performance_monitoring():
    """Demonstrate performance monitoring"""
    print("\n" + "=" * 60)
    print("5. PERFORMANCE MONITORING DEMO")
    print("=" * 60)
    
    # Add some mock trade results
    print("Adding mock trade results...")
    
    mock_trades = [
        {'symbol': 'EURUSD', 'position_type': 'BUY', 'pnl_amount': 25.50, 'pnl_percent': 0.015, 'duration_minutes': 45, 'signal_strength': 0.8},
        {'symbol': 'EURUSD', 'position_type': 'SELL', 'pnl_amount': -12.30, 'pnl_percent': -0.008, 'duration_minutes': 30, 'signal_strength': 0.7},
        {'symbol': 'EURUSD', 'position_type': 'BUY', 'pnl_amount': 18.75, 'pnl_percent': 0.012, 'duration_minutes': 60, 'signal_strength': 0.75},
        {'symbol': 'EURUSD', 'position_type': 'SELL', 'pnl_amount': 31.20, 'pnl_percent': 0.018, 'duration_minutes': 90, 'signal_strength': 0.9},
        {'symbol': 'EURUSD', 'position_type': 'BUY', 'pnl_amount': -8.40, 'pnl_percent': -0.005, 'duration_minutes': 25, 'signal_strength': 0.6}
    ]
    
    for trade in mock_trades:
        performance_monitor.add_trade_result(trade)
        risk_manager.add_trade_result(trade)
    
    # Show performance metrics
    metrics = performance_monitor.get_performance_metrics()
    
    print(f"Total trades: {metrics['total_trades']}")
    print(f"Winning trades: {metrics['winning_trades']}")
    print(f"Win rate: {metrics['win_rate']:.1%}")
    print(f"Total PnL: ${metrics['total_pnl']:.2f}")
    print(f"Profit factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    
    # Show system health
    health = performance_monitor.get_system_health()
    print(f"\nSystem status: {health['status']}")

def demo_adaptive_configuration():
    """Demonstrate adaptive configuration"""
    print("\n" + "=" * 60)
    print("6. ADAPTIVE CONFIGURATION DEMO")
    print("=" * 60)
    
    # Show current parameters
    print("Current parameters:")
    print(f"  Max position size: {config.get('risk_management.max_position_size')}")
    print(f"  Stop loss pips: {config.get('trading.stop_loss_pips')}")
    print(f"  Signal threshold: {config.get('signals.min_signal_strength')}")
    
    # Simulate high volatility market conditions
    print("\nSimulating high volatility market conditions...")
    market_data = {
        'volatility': 0.035,  # High volatility
        'trend_strength': 0.9,  # Strong trend
        'volume_ratio': 2.0
    }
    
    config.update_adaptive_params(market_data)
    
    print("Updated parameters:")
    print(f"  Max position size: {config.get('risk_management.max_position_size')}")
    print(f"  Stop loss pips: {config.get('trading.stop_loss_pips')}")
    print(f"  Signal threshold: {config.get('signals.min_signal_strength')}")

def demo_complete_system():
    """Demonstrate complete system integration"""
    print("\n" + "=" * 60)
    print("7. COMPLETE SYSTEM INTEGRATION DEMO")
    print("=" * 60)
    
    print("Creating ML Trading System instance...")
    # Note: We won't start the actual trading loop in demo mode
    
    # Show system status
    trading_system = MLTradingSystem()
    status = trading_system.get_status()
    
    print(f"System running: {status['is_running']}")
    print(f"Current position: {status['current_position']}")
    
    risk_metrics = status['risk_metrics']
    print(f"Risk metrics available: {len(risk_metrics)} metrics")
    
    perf_metrics = status['performance_metrics']
    print(f"Performance metrics available: {len(perf_metrics)} metrics")
    
    system_health = status['system_health']
    print(f"System health: {system_health['status']}")

def main():
    """Main demo function"""
    print("ML TRADING SYSTEM ENHANCED DEMO")
    print("=" * 60)
    print("Demonstrating institutional-grade trading system features")
    print(f"Demo started at: {datetime.now()}")
    
    try:
        demo_configuration_management()
        demo_risk_management()
        demo_data_management()
        demo_signal_generation()
        demo_performance_monitoring()
        demo_adaptive_configuration()
        demo_complete_system()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("All enhanced features demonstrated:")
        print("✓ Configuration Management")
        print("✓ Risk Management (Kelly Criterion)")
        print("✓ Data Quality & Validation")
        print("✓ Feature Engineering")
        print("✓ Signal Generation")
        print("✓ Performance Monitoring")
        print("✓ Adaptive Parameters")
        print("✓ System Integration")
        
    except Exception as e:
        print(f"\nDemo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()