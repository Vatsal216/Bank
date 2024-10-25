from django.urls import path
from .views import CreateAccountView, DepositView, WithdrawView, TransferView, BalanceView

urlpatterns = [
    path('account/create/', CreateAccountView.as_view(), name='create_account'),
    path('account/<int:account_id>/deposit/', DepositView.as_view(), name='deposit'),
    path('account/<int:account_id>/withdraw/', WithdrawView.as_view(), name='withdraw'),
    path('account/transfer/', TransferView.as_view(), name='transfer'),
    path('account/<int:account_id>/balance/', BalanceView.as_view(), name='balance'),
]
