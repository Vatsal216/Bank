"""
Enhanced ML Trading System with EMA-based signals
Comprehensive ML trading system with institutional-grade features
"""
import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')

# ML and technical analysis imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Import our custom modules
from .config import config
from .risk_management import risk_manager
from .monitoring import performance_monitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DataManager:
    """Enhanced data management with validation and cleaning"""
    
    def __init__(self):
        self.data_cache = {}
        self.last_update = datetime.now()
        
    def get_market_data(self, symbol: str, timeframe: str, bars: int = 1000) -> pd.DataFrame:
        """
        Get market data with comprehensive validation and cleaning
        
        Note: This is a mock implementation. In production, this would connect to MT5
        """
        try:
            # Generate sample data for demonstration
            dates = pd.date_range(end=datetime.now(), periods=bars, freq='5T')
            
            # Generate realistic OHLC data
            np.random.seed(42)  # For reproducible demo data
            
            base_price = 1.1000
            price_changes = np.random.normal(0, 0.0001, bars)
            prices = base_price + np.cumsum(price_changes)
            
            # Create OHLC from price series
            data = []
            for i, price in enumerate(prices):
                high = price + abs(np.random.normal(0, 0.00005))
                low = price - abs(np.random.normal(0, 0.00005))
                open_price = prices[i-1] if i > 0 else price
                close = price
                volume = np.random.randint(100, 1000)
                
                data.append([dates[i], open_price, high, low, close, volume])
            
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df.set_index('timestamp', inplace=True)
            
            # Validate and clean data
            df = self._validate_and_clean_data(df)
            
            logger.info(f"Retrieved {len(df)} bars for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return pd.DataFrame()
    
    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data validation and cleaning"""
        try:
            original_len = len(df)
            
            # Check for missing data
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            max_missing = config.get('data.data_validation_threshold', 0.05)
            
            if missing_ratio > max_missing:
                logger.warning(f"High missing data ratio: {missing_ratio:.2%}")
            
            # Remove rows with any missing values
            df = df.dropna()
            
            # Remove duplicate timestamps
            df = df[~df.index.duplicated(keep='first')]
            
            # Validate OHLC relationships
            df = df[(df['high'] >= df['low']) & 
                   (df['high'] >= df['open']) & 
                   (df['high'] >= df['close']) &
                   (df['low'] <= df['open']) & 
                   (df['low'] <= df['close'])]
            
            # Remove outliers using IQR method
            df = self._remove_outliers(df)
            
            # Ensure minimum data quality
            data_quality_score = len(df) / original_len
            min_quality = config.get('data.min_data_quality_score', 0.8)
            
            if data_quality_score < min_quality:
                logger.warning(f"Low data quality score: {data_quality_score:.2f}")
            
            logger.info(f"Data cleaned: {original_len} -> {len(df)} bars "
                       f"(quality: {data_quality_score:.2f})")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in data validation: {e}")
            return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using statistical methods"""
        try:
            threshold = config.get('data.outlier_threshold', 3.0)
            
            for col in ['open', 'high', 'low', 'close']:
                # Calculate z-scores
                mean = df[col].mean()
                std = df[col].std()
                z_scores = np.abs((df[col] - mean) / std)
                
                # Remove outliers
                df = df[z_scores < threshold]
            
            return df
            
        except Exception as e:
            logger.error(f"Error removing outliers: {e}")
            return df

class FeatureEngineer:
    """Advanced feature engineering with correlation-aware selection"""
    
    def __init__(self):
        self.feature_columns = []
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical features"""
        try:
            data = df.copy()
            
            # EMA features
            ema_periods = config.get('indicators.ema_periods', [10, 20, 50])
            for period in ema_periods:
                data[f'ema_{period}'] = data['close'].ewm(span=period).mean()
                data[f'ema_{period}_slope'] = data[f'ema_{period}'].diff(5)
                data[f'price_to_ema_{period}'] = data['close'] / data[f'ema_{period}'] - 1
            
            # Price features
            data['returns'] = data['close'].pct_change()
            data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            data['volatility'] = data['returns'].rolling(20).std()
            data['price_range'] = (data['high'] - data['low']) / data['close']
            
            # RSI
            rsi_period = config.get('indicators.rsi_period', 14)
            data['rsi'] = self._calculate_rsi(data['close'], rsi_period)
            data['rsi_oversold'] = (data['rsi'] < 30).astype(int)
            data['rsi_overbought'] = (data['rsi'] > 70).astype(int)
            
            # Bollinger Bands
            bb_period = config.get('indicators.bollinger_period', 20)
            bb_std = config.get('indicators.bollinger_std', 2)
            bb_mean = data['close'].rolling(bb_period).mean()
            bb_std_val = data['close'].rolling(bb_period).std()
            data['bb_upper'] = bb_mean + (bb_std_val * bb_std)
            data['bb_lower'] = bb_mean - (bb_std_val * bb_std)
            data['bb_position'] = (data['close'] - bb_lower) / (data['bb_upper'] - data['bb_lower'])
            
            # MACD
            macd_fast = config.get('indicators.macd_fast', 12)
            macd_slow = config.get('indicators.macd_slow', 26)
            macd_signal = config.get('indicators.macd_signal', 9)
            
            ema_fast = data['close'].ewm(span=macd_fast).mean()
            ema_slow = data['close'].ewm(span=macd_slow).mean()
            data['macd'] = ema_fast - ema_slow
            data['macd_signal'] = data['macd'].ewm(span=macd_signal).mean()
            data['macd_histogram'] = data['macd'] - data['macd_signal']
            
            # Volume features
            data['volume_ma'] = data['volume'].rolling(20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_ma']
            
            # Trend features
            data['trend_strength'] = self._calculate_trend_strength(data)
            data['support_resistance'] = self._calculate_support_resistance(data)
            
            # Market microstructure
            data['bid_ask_spread'] = self._estimate_spread(data)
            data['market_impact'] = self._estimate_market_impact(data)
            
            # Remove rows with NaN values
            data = data.dropna()
            
            # Select stable features
            data = self._select_stable_features(data)
            
            logger.info(f"Created {len(data.columns)} features from {len(df)} bars")
            return data
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trend strength indicator"""
        try:
            # Simple trend strength based on EMA alignment
            ema_10 = data['ema_10']
            ema_20 = data['ema_20']
            ema_50 = data['ema_50']
            
            # Uptrend: shorter EMAs above longer EMAs
            uptrend = ((ema_10 > ema_20) & (ema_20 > ema_50)).astype(int)
            downtrend = ((ema_10 < ema_20) & (ema_20 < ema_50)).astype(int)
            
            return uptrend - downtrend
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return pd.Series(0, index=data.index)
    
    def _calculate_support_resistance(self, data: pd.DataFrame) -> pd.Series:
        """Calculate support/resistance levels"""
        try:
            # Simple S/R based on recent highs/lows
            window = 20
            local_high = data['high'].rolling(window, center=True).max()
            local_low = data['low'].rolling(window, center=True).min()
            
            # Distance from S/R levels
            resistance_dist = (data['close'] - local_high) / data['close']
            support_dist = (data['close'] - local_low) / data['close']
            
            return np.minimum(abs(resistance_dist), abs(support_dist))
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return pd.Series(0, index=data.index)
    
    def _estimate_spread(self, data: pd.DataFrame) -> pd.Series:
        """Estimate bid-ask spread"""
        # Simple spread estimation based on high-low range
        return (data['high'] - data['low']) / data['close']
    
    def _estimate_market_impact(self, data: pd.DataFrame) -> pd.Series:
        """Estimate market impact"""
        # Simple market impact based on volume and volatility
        return data['volatility'] / (data['volume_ratio'] + 0.1)
    
    def _select_stable_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Select stable features with correlation filtering"""
        try:
            # Get feature columns (exclude OHLCV)
            feature_cols = [col for col in data.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            if len(feature_cols) == 0:
                return data
            
            feature_data = data[feature_cols]
            
            # Remove highly correlated features
            corr_threshold = config.get('data.correlation_threshold', 0.95)
            correlation_matrix = feature_data.corr().abs()
            
            # Find pairs with high correlation
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] > corr_threshold:
                        high_corr_pairs.append((correlation_matrix.columns[i], 
                                              correlation_matrix.columns[j]))
            
            # Remove one feature from each highly correlated pair
            features_to_remove = set()
            for feat1, feat2 in high_corr_pairs:
                if feat1 not in features_to_remove:
                    features_to_remove.add(feat2)
            
            # Keep features
            stable_features = [col for col in feature_cols if col not in features_to_remove]
            
            # Check feature stability
            stable_window = config.get('data.feature_stability_window', 50)
            if len(data) > stable_window:
                for feat in stable_features.copy():
                    # Check if feature has reasonable variance
                    recent_std = feature_data[feat].tail(stable_window).std()
                    overall_std = feature_data[feat].std()
                    
                    if recent_std > 0 and overall_std > 0:
                        stability_ratio = recent_std / overall_std
                        if stability_ratio > 2.0 or stability_ratio < 0.5:
                            stable_features.remove(feat)
            
            self.feature_columns = stable_features
            
            logger.info(f"Selected {len(stable_features)} stable features "
                       f"(removed {len(features_to_remove)} correlated)")
            
            # Return data with selected features
            return data[['open', 'high', 'low', 'close', 'volume'] + stable_features]
            
        except Exception as e:
            logger.error(f"Error selecting stable features: {e}")
            return data

class MLModel:
    """Machine Learning model with time-series aware cross-validation"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_training = None
        self.feature_importance = {}
        
    def prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training with proper labels"""
        try:
            # Create forward-looking labels
            horizon = config.get('ml_model.prediction_horizon', 5)
            
            # Calculate future returns
            future_returns = data['close'].shift(-horizon) / data['close'] - 1
            
            # Create binary labels (1 for positive returns, 0 for negative)
            labels = (future_returns > 0).astype(int)
            
            # Feature columns
            feature_cols = [col for col in data.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            if len(feature_cols) == 0:
                raise ValueError("No features available for training")
            
            X = data[feature_cols].values
            y = labels.values
            
            # Remove samples with NaN labels
            valid_idx = ~np.isnan(y)
            X = X[valid_idx]
            y = y[valid_idx]
            
            logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return np.array([]), np.array([])
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train model with time-series cross-validation"""
        try:
            if len(X) == 0 or len(y) == 0:
                raise ValueError("No training data available")
            
            min_samples = config.get('ml_model.min_samples_for_training', 200)
            if len(X) < min_samples:
                logger.warning(f"Insufficient training data: {len(X)} < {min_samples}")
                return {}
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Time-series cross-validation
            cv_folds = config.get('ml_model.cv_folds', 5)
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            # Train multiple models and select best
            models = {
                'rf': RandomForestClassifier(n_estimators=100, random_state=42),
                'gb': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }
            
            best_score = 0
            best_model = None
            cv_scores = {}
            
            for model_name, model in models.items():
                scores = []
                
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    model.fit(X_train, y_train)
                    val_pred = model.predict(X_val)
                    score = accuracy_score(y_val, val_pred)
                    scores.append(score)
                
                avg_score = np.mean(scores)
                cv_scores[model_name] = avg_score
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
                    
                logger.info(f"Model {model_name} CV score: {avg_score:.3f}")
            
            # Train best model on full data
            self.model = best_model
            self.model.fit(X_scaled, y)
            self.is_trained = True
            self.last_training = datetime.now()
            
            # Store feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(
                    range(len(self.model.feature_importances_)),
                    self.model.feature_importances_
                ))
            
            logger.info(f"Model trained successfully. Best score: {best_score:.3f}")
            
            return cv_scores
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {}
    
    def predict(self, X: np.ndarray) -> Tuple[int, float]:
        """Make prediction with confidence score"""
        try:
            if not self.is_trained or self.model is None:
                logger.warning("Model not trained")
                return 0, 0.0
            
            X_scaled = self.scaler.transform(X.reshape(1, -1))
            
            # Get prediction and probability
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            confidence = max(probabilities)
            
            return int(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 0, 0.0
    
    def needs_retraining(self) -> bool:
        """Check if model needs retraining"""
        if not self.is_trained or self.last_training is None:
            return True
        
        retrain_hours = config.get('ml_model.retrain_frequency', 24)
        time_since_training = datetime.now() - self.last_training
        
        return time_since_training.total_seconds() > (retrain_hours * 3600)

class SignalGenerator:
    """Multi-factor signal generation with quality scoring"""
    
    def __init__(self):
        self.ml_model = MLModel()
        self.feature_engineer = FeatureEngineer()
        
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive trading signal"""
        try:
            if len(data) < 100:
                return {'signal': 0, 'strength': 0.0, 'factors': {}}
            
            # Get latest data point
            latest = data.iloc[-1]
            
            # Technical analysis signals
            technical_signals = self._get_technical_signals(data)
            
            # ML prediction
            ml_signal, ml_confidence = self._get_ml_signal(data)
            
            # Market regime analysis
            regime_factor = self._analyze_market_regime(data)
            
            # Combine signals
            signal_factors = {
                'ema_signal': technical_signals['ema'],
                'rsi_signal': technical_signals['rsi'],
                'macd_signal': technical_signals['macd'],
                'bollinger_signal': technical_signals['bollinger'],
                'volume_signal': technical_signals['volume'],
                'ml_signal': ml_signal,
                'ml_confidence': ml_confidence,
                'regime_factor': regime_factor
            }
            
            # Calculate overall signal strength
            signal_strength = self._calculate_signal_strength(signal_factors)
            
            # Apply filters
            if not self._passes_filters(data, signal_strength):
                return {'signal': 0, 'strength': 0.0, 'factors': signal_factors}
            
            # Determine final signal direction
            bullish_factors = sum(1 for f in ['ema_signal', 'rsi_signal', 'macd_signal', 
                                            'bollinger_signal', 'ml_signal'] 
                                if signal_factors[f] > 0)
            bearish_factors = sum(1 for f in ['ema_signal', 'rsi_signal', 'macd_signal', 
                                            'bollinger_signal', 'ml_signal'] 
                                if signal_factors[f] < 0)
            
            confluence_threshold = config.get('signals.confluence_threshold', 3)
            
            if bullish_factors >= confluence_threshold:
                signal = 1  # Buy
            elif bearish_factors >= confluence_threshold:
                signal = -1  # Sell
            else:
                signal = 0  # No signal
            
            # Check minimum signal strength
            min_strength = config.get('signals.min_signal_strength', 0.6)
            if signal_strength < min_strength:
                signal = 0
            
            logger.info(f"Signal generated: {signal} (strength: {signal_strength:.2f})")
            
            return {
                'signal': signal,
                'strength': signal_strength,
                'factors': signal_factors,
                'bullish_factors': bullish_factors,
                'bearish_factors': bearish_factors
            }
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return {'signal': 0, 'strength': 0.0, 'factors': {}}
    
    def _get_technical_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get technical analysis signals"""
        try:
            latest = data.iloc[-1]
            
            # EMA signal
            ema_signal = 0
            if 'ema_10' in data.columns and 'ema_20' in data.columns:
                if latest['ema_10'] > latest['ema_20']:
                    ema_signal = 1
                elif latest['ema_10'] < latest['ema_20']:
                    ema_signal = -1
            
            # RSI signal
            rsi_signal = 0
            if 'rsi' in data.columns:
                rsi = latest['rsi']
                if rsi < 30:
                    rsi_signal = 1  # Oversold
                elif rsi > 70:
                    rsi_signal = -1  # Overbought
            
            # MACD signal
            macd_signal = 0
            if 'macd' in data.columns and 'macd_signal' in data.columns:
                if latest['macd'] > latest['macd_signal']:
                    macd_signal = 1
                elif latest['macd'] < latest['macd_signal']:
                    macd_signal = -1
            
            # Bollinger Bands signal
            bb_signal = 0
            if 'bb_position' in data.columns:
                bb_pos = latest['bb_position']
                if bb_pos < 0.2:
                    bb_signal = 1  # Near lower band
                elif bb_pos > 0.8:
                    bb_signal = -1  # Near upper band
            
            # Volume signal
            volume_signal = 0
            if 'volume_ratio' in data.columns:
                if latest['volume_ratio'] > 1.5:
                    volume_signal = 1  # High volume confirmation
            
            return {
                'ema': ema_signal,
                'rsi': rsi_signal,
                'macd': macd_signal,
                'bollinger': bb_signal,
                'volume': volume_signal
            }
            
        except Exception as e:
            logger.error(f"Error getting technical signals: {e}")
            return {}
    
    def _get_ml_signal(self, data: pd.DataFrame) -> Tuple[int, float]:
        """Get ML model prediction"""
        try:
            # Check if model needs retraining
            if self.ml_model.needs_retraining():
                logger.info("Retraining ML model...")
                
                # Prepare training data
                X, y = self.ml_model.prepare_training_data(data)
                
                if len(X) > 0:
                    cv_scores = self.ml_model.train_model(X, y)
                    logger.info(f"Model retrained with CV scores: {cv_scores}")
            
            # Get features for prediction
            if len(self.feature_engineer.feature_columns) == 0:
                return 0, 0.0
            
            latest_features = data[self.feature_engineer.feature_columns].iloc[-1].values
            
            if np.isnan(latest_features).any():
                return 0, 0.0
            
            # Make prediction
            prediction, confidence = self.ml_model.predict(latest_features)
            
            # Convert to signal (-1, 0, 1)
            signal = 1 if prediction == 1 else -1
            
            return signal, confidence
            
        except Exception as e:
            logger.error(f"Error getting ML signal: {e}")
            return 0, 0.0
    
    def _analyze_market_regime(self, data: pd.DataFrame) -> float:
        """Analyze current market regime"""
        try:
            # Simple regime analysis based on volatility and trend
            if 'volatility' in data.columns and 'trend_strength' in data.columns:
                recent_vol = data['volatility'].tail(20).mean()
                recent_trend = data['trend_strength'].tail(20).mean()
                
                # High volatility reduces signal reliability
                vol_factor = max(0.3, 1 - (recent_vol * 100))
                
                # Strong trend increases signal reliability
                trend_factor = 0.5 + abs(recent_trend) * 0.5
                
                return vol_factor * trend_factor
            
            return 1.0
            
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return 1.0
    
    def _calculate_signal_strength(self, factors: Dict[str, Any]) -> float:
        """Calculate overall signal strength"""
        try:
            # Weight different factors
            weights = {
                'ema_signal': 0.15,
                'rsi_signal': 0.1,
                'macd_signal': 0.15,
                'bollinger_signal': 0.1,
                'volume_signal': 0.1,
                'ml_confidence': 0.3,
                'regime_factor': 0.1
            }
            
            strength = 0.0
            total_weight = 0.0
            
            for factor, weight in weights.items():
                if factor in factors:
                    value = factors[factor]
                    if factor == 'ml_confidence' or factor == 'regime_factor':
                        strength += abs(value) * weight
                    else:
                        strength += abs(value) * weight
                    total_weight += weight
            
            return strength / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0.0
    
    def _passes_filters(self, data: pd.DataFrame, signal_strength: float) -> bool:
        """Apply signal filters"""
        try:
            # Volatility filter
            if config.get('signals.volatility_filter', True):
                if 'volatility' in data.columns:
                    recent_vol = data['volatility'].tail(10).mean()
                    if recent_vol > 0.05:  # Too volatile
                        return False
            
            # Trend filter
            if config.get('signals.trend_filter', True):
                if 'trend_strength' in data.columns:
                    recent_trend = data['trend_strength'].tail(10).mean()
                    if abs(recent_trend) < 0.3:  # Too choppy
                        return False
            
            # Volume filter
            if config.get('signals.volume_filter', True):
                if 'volume_ratio' in data.columns:
                    recent_volume = data['volume_ratio'].tail(5).mean()
                    if recent_volume < 0.5:  # Too low volume
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in signal filters: {e}")
            return False

class MLTradingSystem:
    """Main ML Trading System class"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.signal_generator = SignalGenerator()
        self.is_running = False
        self.current_position = None
        self.last_signal_time = None
        
        # Initialize monitoring
        performance_monitor.start_monitoring()
        
        logger.info("ML Trading System initialized")
    
    def start(self):
        """Start the trading system"""
        try:
            self.is_running = True
            logger.info("Starting ML Trading System...")
            
            # Validate configuration
            if not config.validate_config():
                logger.error("Configuration validation failed")
                return
            
            # Check if trading is allowed
            allowed, reason = risk_manager.is_trading_allowed()
            if not allowed:
                logger.warning(f"Trading not allowed: {reason}")
                return
            
            # Main trading loop
            while self.is_running:
                try:
                    self._trading_iteration()
                    time.sleep(60)  # Wait 1 minute between iterations
                    
                except KeyboardInterrupt:
                    logger.info("Trading system interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Error in trading iteration: {e}")
                    time.sleep(30)  # Short sleep on error
            
        except Exception as e:
            logger.error(f"Error starting trading system: {e}")
        finally:
            self.stop()
    
    def _trading_iteration(self):
        """Single trading iteration"""
        try:
            # Get market data
            symbol = config.get('trading.symbol', 'EURUSD')
            timeframe = config.get('trading.timeframe', 'M5')
            
            data = self.data_manager.get_market_data(symbol, timeframe, 500)
            
            if data.empty:
                logger.warning("No market data available")
                return
            
            # Create features
            data_with_features = self.signal_generator.feature_engineer.create_features(data)
            
            # Generate signal
            signal_result = self.signal_generator.generate_signal(data_with_features)
            
            # Process signal
            if signal_result['signal'] != 0:
                self._process_signal(signal_result, data_with_features.iloc[-1])
            
            # Check existing positions
            self._manage_positions()
            
            # Update adaptive parameters
            self._update_adaptive_parameters(data_with_features)
            
        except Exception as e:
            logger.error(f"Error in trading iteration: {e}")
    
    def _process_signal(self, signal_result: Dict[str, Any], latest_data: pd.Series):
        """Process trading signal"""
        try:
            signal = signal_result['signal']
            strength = signal_result['strength']
            
            # Check signal timeout
            signal_timeout = config.get('signals.signal_timeout', 10)
            if (self.last_signal_time and 
                (datetime.now() - self.last_signal_time).seconds < signal_timeout * 60):
                return
            
            # Check if we already have a position in the same direction
            if self.current_position and self.current_position['direction'] == signal:
                return
            
            # Calculate position size
            volatility = latest_data.get('volatility', 0.01)
            position_size = risk_manager.calculate_position_size(strength, volatility)
            
            if position_size <= 0:
                logger.info("Position size is zero, skipping trade")
                return
            
            # Execute trade (mock implementation)
            trade_result = self._execute_trade(signal, position_size, latest_data)
            
            if trade_result['success']:
                self.current_position = trade_result['position']
                self.last_signal_time = datetime.now()
                
                logger.info(f"Trade executed: {signal} {position_size:.4f} lots "
                           f"at {trade_result['price']:.5f}")
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    def _execute_trade(self, signal: int, position_size: float, 
                      latest_data: pd.Series) -> Dict[str, Any]:
        """Execute trade (mock implementation for demo)"""
        try:
            # Mock trade execution
            entry_price = latest_data['close']
            position_type = 'BUY' if signal > 0 else 'SELL'
            
            # Calculate stop loss and take profit
            volatility = latest_data.get('volatility', 0.01)
            stop_loss = risk_manager.calculate_stop_loss(entry_price, position_type, volatility)
            take_profit = risk_manager.calculate_take_profit(entry_price, position_type, stop_loss)
            
            # Mock position
            position = {
                'symbol': config.get('trading.symbol', 'EURUSD'),
                'direction': signal,
                'position_type': position_type,
                'size': position_size,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': datetime.now(),
                'signal_strength': 0.0  # Will be updated with actual strength
            }
            
            return {
                'success': True,
                'position': position,
                'price': entry_price
            }
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {'success': False}
    
    def _manage_positions(self):
        """Manage existing positions"""
        try:
            if not self.current_position:
                return
            
            # Mock position management
            # In real implementation, this would check MT5 positions
            
            # Check if position should be closed (mock logic)
            position_duration = datetime.now() - self.current_position['entry_time']
            
            # Close position after 1 hour for demo
            if position_duration.seconds > 3600:
                self._close_position()
            
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
    
    def _close_position(self):
        """Close current position"""
        try:
            if not self.current_position:
                return
            
            # Mock position closing
            entry_price = self.current_position['entry_price']
            
            # Generate random exit price for demo
            exit_price = entry_price * (1 + np.random.normal(0, 0.001))
            
            # Calculate PnL
            if self.current_position['direction'] > 0:  # Long position
                pnl_pips = (exit_price - entry_price) * 10000
            else:  # Short position
                pnl_pips = (entry_price - exit_price) * 10000
            
            pnl_percent = pnl_pips / 100  # Simplified calculation
            pnl_amount = pnl_percent * float(risk_manager.account_balance) * 0.01
            
            # Create trade result
            duration = datetime.now() - self.current_position['entry_time']
            
            trade_result = {
                'symbol': self.current_position['symbol'],
                'position_type': self.current_position['position_type'],
                'pnl_amount': pnl_amount,
                'pnl_percent': pnl_percent / 100,
                'duration_minutes': duration.total_seconds() / 60,
                'signal_strength': self.current_position.get('signal_strength', 0.6)
            }
            
            # Update risk manager and performance monitor
            risk_manager.add_trade_result(trade_result)
            performance_monitor.add_trade_result(trade_result)
            
            logger.info(f"Position closed: PnL {pnl_amount:.2f} ({pnl_percent:.2f}%)")
            
            self.current_position = None
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def _update_adaptive_parameters(self, data: pd.DataFrame):
        """Update adaptive parameters based on market conditions"""
        try:
            # Calculate market metrics
            latest = data.iloc[-1]
            
            market_data = {
                'volatility': latest.get('volatility', 0.01),
                'trend_strength': latest.get('trend_strength', 0.0),
                'volume_ratio': latest.get('volume_ratio', 1.0)
            }
            
            # Update configuration
            config.update_adaptive_params(market_data)
            
        except Exception as e:
            logger.error(f"Error updating adaptive parameters: {e}")
    
    def stop(self):
        """Stop the trading system"""
        try:
            self.is_running = False
            
            # Close any open positions
            if self.current_position:
                self._close_position()
            
            # Stop monitoring
            performance_monitor.stop_monitoring()
            
            # Save configuration
            config.save_config()
            
            logger.info("Trading system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping trading system: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            return {
                'is_running': self.is_running,
                'current_position': self.current_position,
                'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
                'risk_metrics': risk_manager.get_risk_metrics(),
                'performance_metrics': performance_monitor.get_performance_metrics(),
                'system_health': performance_monitor.get_system_health()
            }
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {}

# Function to run the trading system
def run_trading_system():
    """Main function to run the trading system"""
    try:
        # Initialize and start the trading system
        trading_system = MLTradingSystem()
        
        # Print initial status
        logger.info("ML Trading System v1.0 - Enhanced Edition")
        logger.info("Starting with institutional-grade features...")
        
        # Start trading
        trading_system.start()
        
    except Exception as e:
        logger.error(f"Error running trading system: {e}")

if __name__ == "__main__":
    run_trading_system()