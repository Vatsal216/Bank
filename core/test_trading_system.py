"""
Test suite for ML Trading System components
"""
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import TradingConfig
from core.risk_management import RiskManager
from core.monitoring import PerformanceMonitor
from core.ML_EMA_10 import DataManager, FeatureEngineer, MLModel, SignalGenerator

class TestTradingConfig(unittest.TestCase):
    """Test trading configuration management"""
    
    def setUp(self):
        self.config = TradingConfig()
    
    def test_default_config_load(self):
        """Test that default configuration loads correctly"""
        self.assertIsInstance(self.config.config, dict)
        self.assertIn('risk_management', self.config.config)
        self.assertIn('ml_model', self.config.config)
        self.assertIn('trading', self.config.config)
    
    def test_config_get_set(self):
        """Test configuration get/set operations"""
        # Test setting and getting a value
        test_value = 0.025
        self.config.set('risk_management.max_position_size', test_value)
        retrieved_value = self.config.get('risk_management.max_position_size')
        self.assertEqual(retrieved_value, test_value)
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Should pass with default config
        self.assertTrue(self.config.validate_config())
        
        # Should fail with invalid position size
        self.config.set('risk_management.max_position_size', 0.5)  # Too high
        self.assertFalse(self.config.validate_config())

class TestRiskManager(unittest.TestCase):
    """Test risk management functionality"""
    
    def setUp(self):
        self.risk_manager = RiskManager()
    
    def test_kelly_position_sizing(self):
        """Test Kelly Criterion position sizing"""
        # Test with reasonable parameters
        win_rate = 0.6
        avg_win = 0.02
        avg_loss = 0.015
        
        position_size = self.risk_manager.calculate_kelly_position_size(win_rate, avg_win, avg_loss)
        
        # Should return a reasonable position size
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size, 0.1)  # Max 10%
    
    def test_position_size_calculation(self):
        """Test dynamic position size calculation"""
        signal_strength = 0.8
        market_volatility = 0.01
        
        position_size = self.risk_manager.calculate_position_size(signal_strength, market_volatility)
        
        # Should return a valid position size
        self.assertGreaterEqual(position_size, 0)
        self.assertLessEqual(position_size, 0.1)
    
    def test_stop_loss_calculation(self):
        """Test stop loss calculation"""
        entry_price = 1.1000
        position_type = 'BUY'
        volatility = 0.01
        
        stop_loss = self.risk_manager.calculate_stop_loss(entry_price, position_type, volatility)
        
        # Stop loss should be below entry price for BUY position
        self.assertLess(stop_loss, entry_price)
    
    def test_take_profit_calculation(self):
        """Test take profit calculation"""
        entry_price = 1.1000
        position_type = 'BUY'
        stop_loss = 1.0980
        
        take_profit = self.risk_manager.calculate_take_profit(entry_price, position_type, stop_loss)
        
        # Take profit should be above entry price for BUY position
        self.assertGreater(take_profit, entry_price)
    
    def test_trading_allowed(self):
        """Test trading permission check"""
        # Should be allowed initially
        allowed, reason = self.risk_manager.is_trading_allowed()
        self.assertTrue(allowed)
        
        # Should be disallowed after setting high daily loss
        self.risk_manager.daily_pnl = -self.risk_manager.account_balance * 0.1  # 10% loss
        allowed, reason = self.risk_manager.is_trading_allowed()
        self.assertFalse(allowed)

class TestDataManager(unittest.TestCase):
    """Test data management functionality"""
    
    def setUp(self):
        self.data_manager = DataManager()
    
    def test_market_data_generation(self):
        """Test market data generation (mock)"""
        data = self.data_manager.get_market_data('EURUSD', 'M5', 100)
        
        # Should return a DataFrame with OHLCV columns
        self.assertIsInstance(data, pd.DataFrame)
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, data.columns)
        
        # Should have the requested number of bars (approximately)
        self.assertGreater(len(data), 80)  # Some may be removed during cleaning
    
    def test_data_validation(self):
        """Test data validation and cleaning"""
        # Create test data with some issues
        dates = pd.date_range(end=datetime.now(), periods=50, freq='5T')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.normal(1.1000, 0.001, 50),
            'high': np.random.normal(1.1010, 0.001, 50),
            'low': np.random.normal(1.0990, 0.001, 50),
            'close': np.random.normal(1.1000, 0.001, 50),
            'volume': np.random.randint(100, 1000, 50)
        })
        data.set_index('timestamp', inplace=True)
        
        # Add some invalid data
        data.loc[data.index[5], 'high'] = data.loc[data.index[5], 'low'] - 0.001  # Invalid OHLC
        
        cleaned_data = self.data_manager._validate_and_clean_data(data)
        
        # Should have fewer rows after cleaning
        self.assertLessEqual(len(cleaned_data), len(data))

class TestFeatureEngineer(unittest.TestCase):
    """Test feature engineering functionality"""
    
    def setUp(self):
        self.feature_engineer = FeatureEngineer()
        
        # Create test data
        dates = pd.date_range(end=datetime.now(), periods=200, freq='5T')
        self.test_data = pd.DataFrame({
            'open': np.random.normal(1.1000, 0.001, 200),
            'high': np.random.normal(1.1010, 0.001, 200),
            'low': np.random.normal(1.0990, 0.001, 200),
            'close': np.random.normal(1.1000, 0.001, 200),
            'volume': np.random.randint(100, 1000, 200)
        }, index=dates)
    
    def test_feature_creation(self):
        """Test technical feature creation"""
        features = self.feature_engineer.create_features(self.test_data)
        
        # Should have more columns than original data
        self.assertGreater(len(features.columns), len(self.test_data.columns))
        
        # Should contain expected features
        expected_features = ['ema_10', 'ema_20', 'returns', 'rsi', 'macd']
        for feature in expected_features:
            self.assertIn(feature, features.columns)
    
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        prices = pd.Series([100, 101, 102, 101, 100, 99, 100, 101, 102, 103])
        rsi = self.feature_engineer._calculate_rsi(prices, 5)
        
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        self.assertTrue(all(0 <= x <= 100 for x in valid_rsi))
    
    def test_feature_selection(self):
        """Test feature selection and correlation filtering"""
        # Create data with highly correlated features
        features = self.feature_engineer.create_features(self.test_data)
        
        # Should remove some correlated features
        self.assertIsInstance(features, pd.DataFrame)

class TestMLModel(unittest.TestCase):
    """Test ML model functionality"""
    
    def setUp(self):
        self.ml_model = MLModel()
        
        # Create test data with features
        np.random.seed(42)
        self.X = np.random.random((200, 10))
        self.y = np.random.randint(0, 2, 200)
    
    def test_model_training(self):
        """Test model training process"""
        cv_scores = self.ml_model.train_model(self.X, self.y)
        
        # Should return cross-validation scores
        self.assertIsInstance(cv_scores, dict)
        self.assertTrue(self.ml_model.is_trained)
        self.assertIsNotNone(self.ml_model.last_training)
    
    def test_model_prediction(self):
        """Test model prediction"""
        # Train model first
        self.ml_model.train_model(self.X, self.y)
        
        # Make prediction
        test_sample = self.X[0]
        prediction, confidence = self.ml_model.predict(test_sample)
        
        # Should return valid prediction and confidence
        self.assertIn(prediction, [0, 1])
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_retraining_check(self):
        """Test retraining necessity check"""
        # Should need retraining initially
        self.assertTrue(self.ml_model.needs_retraining())
        
        # Should not need retraining after training
        self.ml_model.train_model(self.X, self.y)
        self.assertFalse(self.ml_model.needs_retraining())

class TestSignalGenerator(unittest.TestCase):
    """Test signal generation functionality"""
    
    def setUp(self):
        self.signal_generator = SignalGenerator()
        
        # Create test data with features
        dates = pd.date_range(end=datetime.now(), periods=200, freq='5T')
        np.random.seed(42)
        
        base_price = 1.1000
        price_changes = np.random.normal(0, 0.0001, 200)
        prices = base_price + np.cumsum(price_changes)
        
        self.test_data = pd.DataFrame({
            'open': prices,
            'high': prices + abs(np.random.normal(0, 0.00005, 200)),
            'low': prices - abs(np.random.normal(0, 0.00005, 200)),
            'close': prices,
            'volume': np.random.randint(100, 1000, 200)
        }, index=dates)
        
        # Add some basic features
        self.test_data['ema_10'] = self.test_data['close'].ewm(span=10).mean()
        self.test_data['ema_20'] = self.test_data['close'].ewm(span=20).mean()
        self.test_data['rsi'] = 50  # Simplified
        self.test_data['macd'] = 0  # Simplified
        self.test_data['macd_signal'] = 0  # Simplified
        self.test_data['bb_position'] = 0.5  # Simplified
        self.test_data['volume_ratio'] = 1.0  # Simplified
        self.test_data['volatility'] = 0.01  # Simplified
        self.test_data['trend_strength'] = 0.0  # Simplified
    
    def test_signal_generation(self):
        """Test complete signal generation"""
        signal_result = self.signal_generator.generate_signal(self.test_data)
        
        # Should return a valid signal result
        self.assertIsInstance(signal_result, dict)
        self.assertIn('signal', signal_result)
        self.assertIn('strength', signal_result)
        self.assertIn('factors', signal_result)
        
        # Signal should be -1, 0, or 1
        self.assertIn(signal_result['signal'], [-1, 0, 1])
        
        # Strength should be between 0 and 1
        self.assertGreaterEqual(signal_result['strength'], 0.0)
        self.assertLessEqual(signal_result['strength'], 1.0)
    
    def test_technical_signals(self):
        """Test technical analysis signals"""
        signals = self.signal_generator._get_technical_signals(self.test_data)
        
        # Should return a dictionary of signals
        self.assertIsInstance(signals, dict)
        expected_signals = ['ema', 'rsi', 'macd', 'bollinger', 'volume']
        for signal_type in expected_signals:
            if signal_type in signals:
                self.assertIn(signals[signal_type], [-1, 0, 1])

class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring functionality"""
    
    def setUp(self):
        self.monitor = PerformanceMonitor()
    
    def test_trade_result_addition(self):
        """Test adding trade results"""
        trade_result = {
            'symbol': 'EURUSD',
            'position_type': 'BUY',
            'pnl_amount': 10.50,
            'pnl_percent': 0.02,
            'duration_minutes': 30,
            'signal_strength': 0.8
        }
        
        initial_trades = self.monitor.total_trades
        self.monitor.add_trade_result(trade_result)
        
        # Should increment trade count
        self.assertEqual(self.monitor.total_trades, initial_trades + 1)
        
        # Should update metrics
        self.assertGreater(len(self.monitor.trade_metrics), 0)
    
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        # Add some mock trades
        for i in range(10):
            pnl = 10 if i % 2 == 0 else -5  # 50% win rate
            trade_result = {
                'symbol': 'EURUSD',
                'position_type': 'BUY',
                'pnl_amount': pnl,
                'pnl_percent': pnl / 1000,
                'duration_minutes': 30,
                'signal_strength': 0.8
            }
            self.monitor.add_trade_result(trade_result)
        
        metrics = self.monitor.get_performance_metrics()
        
        # Should return valid metrics
        self.assertIsInstance(metrics, dict)
        self.assertIn('win_rate', metrics)
        self.assertIn('total_trades', metrics)
        self.assertEqual(metrics['total_trades'], 10)
    
    def test_system_health(self):
        """Test system health monitoring"""
        health = self.monitor.get_system_health()
        
        # Should return health status
        self.assertIsInstance(health, dict)
        self.assertIn('status', health)

def run_tests():
    """Run all tests"""
    # Create test suite
    test_classes = [
        TestTradingConfig,
        TestRiskManager,
        TestDataManager,
        TestFeatureEngineer,
        TestMLModel,
        TestSignalGenerator,
        TestPerformanceMonitor
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    print(f"\nTest suite {'PASSED' if success else 'FAILED'}")
    exit(0 if success else 1)