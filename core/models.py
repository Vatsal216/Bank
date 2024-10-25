from django.db import models
from django.utils import timezone

class CustomerAccount(models.Model):
    name = models.CharField(max_length=100)
    balance = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    created_at = models.DateTimeField(auto_now_add=True)

class Transaction(models.Model):
    account = models.ForeignKey(CustomerAccount, on_delete=models.CASCADE, related_name='transactions')
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    transaction_type = models.CharField(max_length=10, choices=[('deposit', 'Deposit'), ('withdraw', 'Withdraw'), ('transfer', 'Transfer')])
    created_at = models.DateTimeField(default=timezone.now)
