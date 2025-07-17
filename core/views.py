from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import CustomerAccount, Transaction
from decimal import Decimal

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
