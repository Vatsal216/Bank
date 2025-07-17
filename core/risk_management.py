"""
Risk Management Module for ML Trading System
Implements Kelly Criterion-based position sizing and advanced risk controls
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
from .config import config

logger = logging.getLogger(__name__)

class RiskManager:
    """Advanced risk management with Kelly Criterion and dynamic controls"""
    
    def __init__(self):
        self.account_balance = Decimal('10000')  # Default demo balance
        self.daily_pnl = Decimal('0')
        self.max_drawdown = Decimal('0')
        self.peak_balance = self.account_balance
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        self.trade_history = []
        
    def update_account_balance(self, new_balance: Decimal):
        """Update account balance and calculate drawdown"""
        self.account_balance = new_balance
        
        # Update peak balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
            
        # Calculate current drawdown
        current_drawdown = (self.peak_balance - new_balance) / self.peak_balance
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
            
        # Reset daily metrics if new day
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_pnl = Decimal('0')
            self.daily_trades = 0
            self.last_reset_date = current_date
            
        logger.info(f"Account balance updated: {new_balance}, Drawdown: {current_drawdown:.2%}")
    
    def calculate_kelly_position_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive value)
            
        Returns:
            Optimal position fraction (0-1)
        """
        try:
            if win_rate <= 0 or win_rate >= 1 or avg_win <= 0 or avg_loss <= 0:
                logger.warning("Invalid parameters for Kelly Criterion, using default size")
                return config.get('risk_management.max_position_size', 0.02)
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Apply safety margin (use only fraction of Kelly)
            kelly_multiplier = config.get('risk_management.kelly_fraction', 0.25)
            safe_kelly = kelly_fraction * kelly_multiplier
            
            # Ensure within bounds
            max_position = config.get('risk_management.max_position_size', 0.02)
            position_size = max(0, min(safe_kelly, max_position))
            
            logger.info(f"Kelly position size: {position_size:.4f} (raw Kelly: {kelly_fraction:.4f})")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating Kelly position size: {e}")
            return config.get('risk_management.max_position_size', 0.02)
    
    def calculate_position_size(self, signal_strength: float, market_volatility: float) -> float:
        """
        Calculate dynamic position size based on multiple factors
        
        Args:
            signal_strength: Confidence in the signal (0-1)
            market_volatility: Current market volatility
            
        Returns:
            Position size as fraction of account
        """
        try:
            # Get historical performance for Kelly calculation
            win_rate, avg_win, avg_loss = self._calculate_historical_performance()
            
            # Base Kelly position size
            kelly_size = self.calculate_kelly_position_size(win_rate, avg_win, avg_loss)
            
            # Adjust for signal strength
            signal_adjustment = 0.5 + (signal_strength * 0.5)  # 0.5-1.0 multiplier
            
            # Adjust for volatility (higher vol = smaller position)
            vol_adjustment = max(0.3, 1 - (market_volatility * 10))  # 0.3-1.0 multiplier
            
            # Calculate final position size
            position_size = kelly_size * signal_adjustment * vol_adjustment
            
            # Apply risk controls
            position_size = self._apply_risk_controls(position_size)
            
            logger.info(f"Position size: {position_size:.4f} "
                       f"(Kelly: {kelly_size:.4f}, Signal: {signal_adjustment:.2f}, "
                       f"Vol: {vol_adjustment:.2f})")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01  # Minimum safe position size
    
    def _calculate_historical_performance(self) -> Tuple[float, float, float]:
        """Calculate win rate and average win/loss from trade history"""
        if len(self.trade_history) < 10:
            # Not enough history, use conservative defaults
            return 0.5, 0.015, 0.01
        
        recent_trades = self.trade_history[-50:]  # Last 50 trades
        
        wins = [t['pnl'] for t in recent_trades if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in recent_trades if t['pnl'] < 0]
        
        win_rate = len(wins) / len(recent_trades) if recent_trades else 0.5
        avg_win = np.mean(wins) if wins else 0.015
        avg_loss = np.mean(losses) if losses else 0.01
        
        return win_rate, avg_win, avg_loss
    
    def _apply_risk_controls(self, position_size: float) -> float:
        """Apply various risk control limits"""
        
        # Daily loss limit
        max_daily_loss = config.get('risk_management.max_daily_loss', 0.05)
        if self.daily_pnl < -max_daily_loss * self.account_balance:
            logger.warning("Daily loss limit reached, no new positions")
            return 0
        
        # Maximum drawdown limit
        max_drawdown_limit = config.get('risk_management.max_drawdown', 0.15)
        if self.max_drawdown > max_drawdown_limit:
            logger.warning("Maximum drawdown exceeded, reducing position size")
            position_size *= 0.5
        
        # Maximum position size limit
        max_pos_limit = config.get('risk_management.max_position_size', 0.02)
        position_size = min(position_size, max_pos_limit)
        
        # Minimum position size
        min_pos_size = 0.001
        position_size = max(position_size, min_pos_size) if position_size > 0 else 0
        
        return position_size
    
    def calculate_stop_loss(self, entry_price: float, position_type: str, 
                          volatility: float) -> float:
        """
        Calculate dynamic stop loss based on volatility and position type
        
        Args:
            entry_price: Entry price of the position
            position_type: 'BUY' or 'SELL'
            volatility: Current market volatility
            
        Returns:
            Stop loss price
        """
        try:
            # Base stop loss from config
            base_stop_pips = config.get('trading.stop_loss_pips', 20)
            
            # Adjust for volatility
            vol_multiplier = max(0.5, min(2.0, volatility * 100))  # 0.5-2.0x
            adjusted_stop_pips = base_stop_pips * vol_multiplier
            
            # Convert pips to price
            pip_value = 0.0001  # For most forex pairs
            stop_distance = adjusted_stop_pips * pip_value
            
            if position_type.upper() == 'BUY':
                stop_loss = entry_price - stop_distance
            else:  # SELL
                stop_loss = entry_price + stop_distance
            
            logger.info(f"Stop loss calculated: {stop_loss:.5f} "
                       f"({adjusted_stop_pips:.1f} pips from {entry_price:.5f})")
            
            return stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            # Fallback: 1% stop loss
            if position_type.upper() == 'BUY':
                return entry_price * 0.99
            else:
                return entry_price * 1.01
    
    def calculate_take_profit(self, entry_price: float, position_type: str,
                            stop_loss: float, risk_reward_ratio: float = 2.0) -> float:
        """
        Calculate take profit based on risk-reward ratio
        
        Args:
            entry_price: Entry price of the position
            position_type: 'BUY' or 'SELL'
            stop_loss: Stop loss price
            risk_reward_ratio: Desired risk-reward ratio
            
        Returns:
            Take profit price
        """
        try:
            # Calculate risk (distance to stop loss)
            if position_type.upper() == 'BUY':
                risk = entry_price - stop_loss
                take_profit = entry_price + (risk * risk_reward_ratio)
            else:  # SELL
                risk = stop_loss - entry_price
                take_profit = entry_price - (risk * risk_reward_ratio)
            
            logger.info(f"Take profit calculated: {take_profit:.5f} "
                       f"(RR ratio: {risk_reward_ratio})")
            
            return take_profit
            
        except Exception as e:
            logger.error(f"Error calculating take profit: {e}")
            # Fallback: 2% take profit
            if position_type.upper() == 'BUY':
                return entry_price * 1.02
            else:
                return entry_price * 0.98
    
    def add_trade_result(self, trade_result: Dict[str, Any]):
        """Add completed trade to history for performance tracking"""
        try:
            pnl_pct = float(trade_result.get('pnl_percent', 0))
            
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': trade_result.get('symbol'),
                'position_type': trade_result.get('position_type'),
                'pnl': pnl_pct,
                'pnl_amount': trade_result.get('pnl_amount', 0),
                'duration': trade_result.get('duration_minutes', 0)
            }
            
            self.trade_history.append(trade_record)
            
            # Update daily PnL
            self.daily_pnl += Decimal(str(trade_result.get('pnl_amount', 0)))
            self.daily_trades += 1
            
            # Keep only recent history (last 200 trades)
            if len(self.trade_history) > 200:
                self.trade_history = self.trade_history[-200:]
            
            logger.info(f"Trade result added: PnL {pnl_pct:.2%}, Daily PnL: {self.daily_pnl}")
            
        except Exception as e:
            logger.error(f"Error adding trade result: {e}")
    
    def is_trading_allowed(self) -> Tuple[bool, str]:
        """
        Check if trading is currently allowed based on risk controls
        
        Returns:
            Tuple of (allowed, reason)
        """
        try:
            # Daily loss limit
            max_daily_loss = config.get('risk_management.max_daily_loss', 0.05)
            if self.daily_pnl < -max_daily_loss * self.account_balance:
                return False, "Daily loss limit exceeded"
            
            # Maximum drawdown
            max_drawdown_limit = config.get('risk_management.max_drawdown', 0.15)
            if self.max_drawdown > max_drawdown_limit:
                return False, "Maximum drawdown exceeded"
            
            # Account balance minimum
            if self.account_balance < Decimal('100'):
                return False, "Account balance too low"
            
            return True, "Trading allowed"
            
        except Exception as e:
            logger.error(f"Error checking trading status: {e}")
            return False, "Risk check error"
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics for monitoring"""
        try:
            win_rate, avg_win, avg_loss = self._calculate_historical_performance()
            
            return {
                'account_balance': float(self.account_balance),
                'daily_pnl': float(self.daily_pnl),
                'daily_pnl_percent': float(self.daily_pnl / self.account_balance * 100),
                'max_drawdown': float(self.max_drawdown * 100),
                'daily_trades': self.daily_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'risk_reward_ratio': avg_win / avg_loss if avg_loss > 0 else 0,
                'total_trades': len(self.trade_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return {}

# Global risk manager instance
risk_manager = RiskManager()