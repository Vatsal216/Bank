from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import CustomerAccount, Transaction
from decimal import Decimal
import logging

# Import ML Trading System components
from .ML_EMA_10 import MLTradingSystem
from .config import config
from .risk_management import risk_manager
from .monitoring import performance_monitor

logger = logging.getLogger(__name__)

class CreateAccountView(APIView):
    def post(self, request):
        name = request.data.get('name')
        if not name:
            return Response({'error': 'Name is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        account = CustomerAccount.objects.create(name=name)
        return Response({
            'id': account.id,
            'name': account.name,
            'balance': account.balance
        }, status=status.HTTP_201_CREATED)

class DepositView(APIView):
    def post(self, request, account_id):
        try:
            account = CustomerAccount.objects.get(id=account_id)
            amount = Decimal(str(request.data.get('amount', 0)))
            
            if amount <= 0:
                return Response({'error': 'Amount must be positive'}, status=status.HTTP_400_BAD_REQUEST)
            
            account.balance += amount
            account.save()
            
            Transaction.objects.create(
                account=account,
                amount=amount,
                transaction_type='deposit'
            )
            
            return Response({
                'message': 'Deposit successful',
                'new_balance': account.balance
            })
        except CustomerAccount.DoesNotExist:
            return Response({'error': 'Account not found'}, status=status.HTTP_404_NOT_FOUND)

class WithdrawView(APIView):
    def post(self, request, account_id):
        try:
            account = CustomerAccount.objects.get(id=account_id)
            amount = Decimal(str(request.data.get('amount', 0)))
            
            if amount <= 0:
                return Response({'error': 'Amount must be positive'}, status=status.HTTP_400_BAD_REQUEST)
            
            if account.balance < amount:
                return Response({'error': 'Insufficient funds'}, status=status.HTTP_400_BAD_REQUEST)
            
            account.balance -= amount
            account.save()
            
            Transaction.objects.create(
                account=account,
                amount=-amount,
                transaction_type='withdraw'
            )
            
            return Response({
                'message': 'Withdrawal successful',
                'new_balance': account.balance
            })
        except CustomerAccount.DoesNotExist:
            return Response({'error': 'Account not found'}, status=status.HTTP_404_NOT_FOUND)

class TransferView(APIView):
    def post(self, request):
        from_account_id = request.data.get('from_account_id')
        to_account_id = request.data.get('to_account_id')
        amount = Decimal(str(request.data.get('amount', 0)))
        
        if amount <= 0:
            return Response({'error': 'Amount must be positive'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            from_account = CustomerAccount.objects.get(id=from_account_id)
            to_account = CustomerAccount.objects.get(id=to_account_id)
            
            if from_account.balance < amount:
                return Response({'error': 'Insufficient funds'}, status=status.HTTP_400_BAD_REQUEST)
            
            from_account.balance -= amount
            to_account.balance += amount
            from_account.save()
            to_account.save()
            
            Transaction.objects.create(
                account=from_account,
                amount=-amount,
                transaction_type='transfer'
            )
            Transaction.objects.create(
                account=to_account,
                amount=amount,
                transaction_type='transfer'
            )
            
            return Response({
                'message': 'Transfer successful',
                'from_balance': from_account.balance,
                'to_balance': to_account.balance
            })
        except CustomerAccount.DoesNotExist:
            return Response({'error': 'Account not found'}, status=status.HTTP_404_NOT_FOUND)

class BalanceView(APIView):
    def get(self, request, account_id):
        try:
            account = CustomerAccount.objects.get(id=account_id)
            return Response({
                'account_id': account.id,
                'name': account.name,
                'balance': account.balance
            })
        except CustomerAccount.DoesNotExist:
            return Response({'error': 'Account not found'}, status=status.HTTP_404_NOT_FOUND)

# ML Trading System API Views

class TradingSystemStatusView(APIView):
    """Get trading system status and metrics"""
    
    def get(self, request):
        try:
            # Create trading system instance to get status
            trading_system = MLTradingSystem()
            status_data = trading_system.get_status()
            
            return Response({
                'status': 'success',
                'data': status_data
            })
        except Exception as e:
            logger.error(f"Error getting trading system status: {e}")
            return Response({
                'status': 'error',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class TradingConfigView(APIView):
    """Get and update trading configuration"""
    
    def get(self, request):
        try:
            return Response({
                'status': 'success',
                'config': config.config
            })
        except Exception as e:
            logger.error(f"Error getting trading config: {e}")
            return Response({
                'status': 'error',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def post(self, request):
        try:
            # Update configuration
            updates = request.data.get('updates', {})
            
            for key_path, value in updates.items():
                config.set(key_path, value)
            
            # Save configuration
            config.save_config()
            
            return Response({
                'status': 'success',
                'message': 'Configuration updated successfully'
            })
        except Exception as e:
            logger.error(f"Error updating trading config: {e}")
            return Response({
                'status': 'error',
                'message': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)

class RiskMetricsView(APIView):
    """Get risk management metrics"""
    
    def get(self, request):
        try:
            metrics = risk_manager.get_risk_metrics()
            
            return Response({
                'status': 'success',
                'risk_metrics': metrics
            })
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return Response({
                'status': 'error',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class PerformanceMetricsView(APIView):
    """Get performance metrics"""
    
    def get(self, request):
        try:
            metrics = performance_monitor.get_performance_metrics()
            
            return Response({
                'status': 'success',
                'performance_metrics': metrics
            })
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return Response({
                'status': 'error',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class SystemHealthView(APIView):
    """Get system health status"""
    
    def get(self, request):
        try:
            health = performance_monitor.get_system_health()
            
            return Response({
                'status': 'success',
                'system_health': health
            })
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return Response({
                'status': 'error',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class TradingControlView(APIView):
    """Control trading system (start/stop)"""
    
    def post(self, request):
        try:
            action = request.data.get('action')
            
            if action == 'start':
                # Note: Starting the trading system in a web request isn't ideal
                # In production, this should be handled by a background task/service
                return Response({
                    'status': 'success',
                    'message': 'Trading system start command received. Use background service for actual trading.'
                })
            elif action == 'stop':
                return Response({
                    'status': 'success',
                    'message': 'Trading system stop command received.'
                })
            else:
                return Response({
                    'status': 'error',
                    'message': 'Invalid action. Use "start" or "stop".'
                }, status=status.HTTP_400_BAD_REQUEST)
                
        except Exception as e:
            logger.error(f"Error controlling trading system: {e}")
            return Response({
                'status': 'error',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
