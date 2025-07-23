from django.urls import path
from .views import (
    CreateAccountView, DepositView, WithdrawView, TransferView, BalanceView,
    TradingSystemStatusView, TradingConfigView, RiskMetricsView,
    PerformanceMetricsView, SystemHealthView, TradingControlView
)

urlpatterns = [
    # Banking API endpoints
    path('account/create/', CreateAccountView.as_view(), name='create_account'),
    path('account/<int:account_id>/deposit/', DepositView.as_view(), name='deposit'),
    path('account/<int:account_id>/withdraw/', WithdrawView.as_view(), name='withdraw'),
    path('account/transfer/', TransferView.as_view(), name='transfer'),
    path('account/<int:account_id>/balance/', BalanceView.as_view(), name='balance'),
    
    # ML Trading System API endpoints
    path('trading/status/', TradingSystemStatusView.as_view(), name='trading_status'),
    path('trading/config/', TradingConfigView.as_view(), name='trading_config'),
    path('trading/risk-metrics/', RiskMetricsView.as_view(), name='risk_metrics'),
    path('trading/performance/', PerformanceMetricsView.as_view(), name='performance_metrics'),
    path('trading/health/', SystemHealthView.as_view(), name='system_health'),
    path('trading/control/', TradingControlView.as_view(), name='trading_control'),
]
