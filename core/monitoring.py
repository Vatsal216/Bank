"""
Monitoring Module for ML Trading System
Real-time system health monitoring and alerts
"""
import time
import logging
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque
import json
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from .config import config
from .risk_management import risk_manager

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Real-time performance monitoring and alerting system"""
    
    def __init__(self):
        self.trade_metrics = deque(maxlen=100)  # Last 100 trades
        self.system_metrics = deque(maxlen=1000)  # Last 1000 health checks
        self.alerts_sent = {}  # Track sent alerts to avoid spam
        self.is_monitoring = False
        self.monitor_thread = None
        self.last_health_check = datetime.now()
        
        # Performance tracking
        self.session_start_time = datetime.now()
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_consecutive_losses = 0
        self.current_consecutive_losses = 0
        
    def start_monitoring(self):
        """Start the monitoring thread"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                self._perform_health_check()
                self._check_performance_alerts()
                self._cleanup_old_alerts()
                
                # Sleep for configured interval
                interval = config.get('monitoring.health_check_interval', 300)
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Short sleep on error
    
    def _perform_health_check(self):
        """Perform system health check"""
        try:
            health_data = {
                'timestamp': datetime.now(),
                'system_status': 'healthy',
                'memory_usage': self._get_memory_usage(),
                'cpu_usage': self._get_cpu_usage(),
                'connection_status': self._check_mt5_connection(),
                'data_freshness': self._check_data_freshness(),
                'model_status': self._check_model_status()
            }
            
            # Determine overall system status
            if not health_data['connection_status']:
                health_data['system_status'] = 'connection_error'
            elif health_data['data_freshness'] > 600:  # 10 minutes
                health_data['system_status'] = 'stale_data'
            elif health_data['memory_usage'] > 80:
                health_data['system_status'] = 'high_memory'
            
            self.system_metrics.append(health_data)
            self.last_health_check = datetime.now()
            
            # Log health status
            if health_data['system_status'] != 'healthy':
                logger.warning(f"System health issue: {health_data['system_status']}")
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
    
    def add_trade_result(self, trade_result: Dict[str, Any]):
        """Add trade result to performance tracking"""
        try:
            # Extract trade metrics
            pnl = float(trade_result.get('pnl_amount', 0))
            is_winning = pnl > 0
            
            # Update counters
            self.total_trades += 1
            self.total_pnl += pnl
            
            if is_winning:
                self.winning_trades += 1
                self.current_consecutive_losses = 0
            else:
                self.current_consecutive_losses += 1
                self.max_consecutive_losses = max(
                    self.max_consecutive_losses, 
                    self.current_consecutive_losses
                )
            
            # Store trade metrics
            trade_metrics = {
                'timestamp': datetime.now(),
                'symbol': trade_result.get('symbol'),
                'pnl': pnl,
                'pnl_percent': float(trade_result.get('pnl_percent', 0)),
                'duration_minutes': trade_result.get('duration_minutes', 0),
                'position_type': trade_result.get('position_type'),
                'signal_strength': trade_result.get('signal_strength', 0),
                'is_winning': is_winning
            }
            
            self.trade_metrics.append(trade_metrics)
            
            # Check for immediate alerts
            self._check_trade_alerts(trade_metrics)
            
            logger.info(f"Trade result recorded: PnL {pnl:.2f}, "
                       f"Win rate: {self.get_win_rate():.1%}")
            
        except Exception as e:
            logger.error(f"Error adding trade result: {e}")
    
    def _check_performance_alerts(self):
        """Check for performance-based alerts"""
        try:
            # Get current performance metrics
            metrics = self.get_performance_metrics()
            
            # PnL alert
            pnl_threshold = config.get('monitoring.alert_threshold_pnl', -0.03)
            if metrics['total_pnl_percent'] <= pnl_threshold:
                self._send_alert('pnl_alert', 
                    f"Total PnL dropped to {metrics['total_pnl_percent']:.1%}")
            
            # Drawdown alert
            drawdown_threshold = config.get('monitoring.alert_threshold_drawdown', 0.1)
            if metrics['max_drawdown'] >= drawdown_threshold:
                self._send_alert('drawdown_alert',
                    f"Drawdown reached {metrics['max_drawdown']:.1%}")
            
            # Accuracy alert
            accuracy_threshold = config.get('monitoring.alert_threshold_accuracy', 0.4)
            if self.total_trades >= 10 and metrics['win_rate'] <= accuracy_threshold:
                self._send_alert('accuracy_alert',
                    f"Win rate dropped to {metrics['win_rate']:.1%}")
            
            # Consecutive losses alert
            if self.current_consecutive_losses >= 5:
                self._send_alert('consecutive_losses',
                    f"{self.current_consecutive_losses} consecutive losses")
            
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
    
    def _check_trade_alerts(self, trade_metrics: Dict[str, Any]):
        """Check for trade-specific alerts"""
        try:
            # Large loss alert
            if trade_metrics['pnl_percent'] <= -0.05:  # 5% loss
                self._send_alert('large_loss',
                    f"Large loss: {trade_metrics['pnl_percent']:.1%} on {trade_metrics['symbol']}")
            
            # Model confidence alert
            if trade_metrics['signal_strength'] < 0.3:
                self._send_alert('low_confidence',
                    f"Low signal confidence: {trade_metrics['signal_strength']:.2f}")
            
        except Exception as e:
            logger.error(f"Error checking trade alerts: {e}")
    
    def _send_alert(self, alert_type: str, message: str):
        """Send alert via configured channels"""
        try:
            # Check if alert was recently sent (avoid spam)
            current_time = datetime.now()
            last_sent = self.alerts_sent.get(alert_type)
            
            if last_sent and (current_time - last_sent).seconds < 3600:  # 1 hour cooldown
                return
            
            # Log alert
            logger.warning(f"ALERT [{alert_type}]: {message}")
            
            # Send email if enabled
            if config.get('monitoring.enable_email_alerts', False):
                self._send_email_alert(alert_type, message)
            
            # Send Telegram if enabled
            if config.get('monitoring.enable_telegram_alerts', False):
                self._send_telegram_alert(alert_type, message)
            
            # Record alert sent
            self.alerts_sent[alert_type] = current_time
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    def _send_email_alert(self, alert_type: str, message: str):
        """Send email alert"""
        try:
            # Email configuration (should be in config)
            smtp_server = config.get('monitoring.email_smtp_server', 'smtp.gmail.com')
            smtp_port = config.get('monitoring.email_smtp_port', 587)
            email_user = config.get('monitoring.email_user', '')
            email_password = config.get('monitoring.email_password', '')
            email_to = config.get('monitoring.email_to', '')
            
            if not all([email_user, email_password, email_to]):
                logger.warning("Email configuration incomplete")
                return
            
            # Create message
            msg = MimeMultipart()
            msg['From'] = email_user
            msg['To'] = email_to
            msg['Subject'] = f"Trading System Alert: {alert_type}"
            
            body = f"""
            Trading System Alert
            
            Alert Type: {alert_type}
            Message: {message}
            Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Performance Summary:
            {self._get_performance_summary()}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(email_user, email_password)
            text = msg.as_string()
            server.sendmail(email_user, email_to, text)
            server.quit()
            
            logger.info(f"Email alert sent: {alert_type}")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
    
    def _send_telegram_alert(self, alert_type: str, message: str):
        """Send Telegram alert"""
        try:
            # Telegram configuration (should be in config)
            bot_token = config.get('monitoring.telegram_bot_token', '')
            chat_id = config.get('monitoring.telegram_chat_id', '')
            
            if not all([bot_token, chat_id]):
                logger.warning("Telegram configuration incomplete")
                return
            
            # Format message
            telegram_message = f"🚨 *Trading Alert*\n\n"
            telegram_message += f"*Type:* {alert_type}\n"
            telegram_message += f"*Message:* {message}\n"
            telegram_message += f"*Time:* {datetime.now().strftime('%H:%M:%S')}\n\n"
            telegram_message += f"*Performance:*\n{self._get_performance_summary()}"
            
            # Note: Actual Telegram API call would go here
            # For now, just log it
            logger.info(f"Telegram alert would be sent: {alert_type}")
            
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
    
    def _get_performance_summary(self) -> str:
        """Get formatted performance summary"""
        try:
            metrics = self.get_performance_metrics()
            
            summary = f"""
            Total Trades: {self.total_trades}
            Win Rate: {metrics['win_rate']:.1%}
            Total PnL: {metrics['total_pnl']:.2f}
            Max Drawdown: {metrics['max_drawdown']:.1%}
            Consecutive Losses: {self.current_consecutive_losses}
            Session Duration: {metrics['session_duration_hours']:.1f}h
            """
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return "Performance data unavailable"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        try:
            # Calculate metrics
            win_rate = self.winning_trades / max(self.total_trades, 1)
            session_duration = (datetime.now() - self.session_start_time).total_seconds() / 3600
            
            # Get risk metrics
            risk_metrics = risk_manager.get_risk_metrics()
            
            metrics = {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.total_trades - self.winning_trades,
                'win_rate': win_rate,
                'total_pnl': self.total_pnl,
                'total_pnl_percent': risk_metrics.get('daily_pnl_percent', 0),
                'max_drawdown': risk_metrics.get('max_drawdown', 0) / 100,
                'max_consecutive_losses': self.max_consecutive_losses,
                'current_consecutive_losses': self.current_consecutive_losses,
                'session_duration_hours': session_duration,
                'avg_trade_duration': self._get_avg_trade_duration(),
                'trades_per_hour': self.total_trades / max(session_duration, 0.1),
                'sharpe_ratio': self._calculate_sharpe_ratio(),
                'profit_factor': self._calculate_profit_factor(),
                'last_health_check': self.last_health_check.isoformat()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def _get_avg_trade_duration(self) -> float:
        """Calculate average trade duration in minutes"""
        if not self.trade_metrics:
            return 0
        
        durations = [t['duration_minutes'] for t in self.trade_metrics if t['duration_minutes'] > 0]
        return sum(durations) / len(durations) if durations else 0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.trade_metrics) < 10:
            return 0
        
        returns = [t['pnl_percent'] for t in self.trade_metrics]
        
        if not returns:
            return 0
        
        import numpy as np
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        risk_free_rate = config.get('risk_management.risk_free_rate', 0.02) / 252  # Daily
        sharpe = (mean_return - risk_free_rate) / std_return
        
        return sharpe * np.sqrt(252)  # Annualized
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if not self.trade_metrics:
            return 0
        
        profits = sum(t['pnl'] for t in self.trade_metrics if t['pnl'] > 0)
        losses = abs(sum(t['pnl'] for t in self.trade_metrics if t['pnl'] < 0))
        
        return profits / losses if losses > 0 else float('inf')
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0  # psutil not available
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 0  # psutil not available
    
    def _check_mt5_connection(self) -> bool:
        """Check MT5 connection status"""
        try:
            # This would check actual MT5 connection
            # For now, return True as placeholder
            return True
        except Exception:
            return False
    
    def _check_data_freshness(self) -> int:
        """Check how old the latest data is (in seconds)"""
        try:
            # This would check actual data timestamp
            # For now, return 0 as placeholder
            return 0
        except Exception:
            return 999
    
    def _check_model_status(self) -> str:
        """Check ML model status"""
        try:
            # This would check model health
            # For now, return 'healthy' as placeholder
            return 'healthy'
        except Exception:
            return 'unknown'
    
    def _cleanup_old_alerts(self):
        """Clean up old alert timestamps"""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(hours=24)
            
            self.alerts_sent = {
                alert_type: timestamp 
                for alert_type, timestamp in self.alerts_sent.items()
                if timestamp > cutoff_time
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up alerts: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        try:
            if not self.system_metrics:
                return {'status': 'unknown', 'message': 'No health data available'}
            
            latest_health = self.system_metrics[-1]
            
            return {
                'status': latest_health['system_status'],
                'last_check': latest_health['timestamp'].isoformat(),
                'memory_usage': latest_health['memory_usage'],
                'cpu_usage': latest_health['cpu_usage'],
                'connection_status': latest_health['connection_status'],
                'data_freshness_seconds': latest_health['data_freshness'],
                'model_status': latest_health['model_status']
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {'status': 'error', 'message': str(e)}

# Global performance monitor instance
performance_monitor = PerformanceMonitor()