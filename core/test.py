from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from .models import CustomerAccount, Transaction

class AccountTests(APITestCase):

    def setUp(self):
        # Create two accounts for testing transfers
        self.account_1 = CustomerAccount.objects.create(name="Alice", balance=1000)
        self.account_2 = CustomerAccount.objects.create(name="Bob", balance=500)

    def test_create_account(self):
        url = reverse('create_account')
        data = {
            "name": "Charlie",
            "balance": 300
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(CustomerAccount.objects.count(), 3)
        self.assertEqual(CustomerAccount.objects.get(name="Charlie").balance, 300)

    def test_deposit(self):
        url = reverse('deposit', args=[self.account_1.id])
        data = {"amount": 200}
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.account_1.refresh_from_db()
        self.assertEqual(self.account_1.balance, 1200)

    def test_withdraw(self):
        url = reverse('withdraw', args=[self.account_1.id])
        data = {"amount": 400}
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.account_1.refresh_from_db()
        self.assertEqual(self.account_1.balance, 600)

    def test_overdraft_withdraw(self):
        url = reverse('withdraw', args=[self.account_2.id])
        data = {"amount": 1000}  # More than balance
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.account_2.refresh_from_db()
        self.assertEqual(self.account_2.balance, 500)

    def test_transfer_funds(self):
        url = reverse('transfer')
        data = {
            "from_account": self.account_1.id,
            "to_account": self.account_2.id,
            "amount": 300
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.account_1.refresh_from_db()
        self.account_2.refresh_from_db()
        self.assertEqual(self.account_1.balance, 700)
        self.assertEqual(self.account_2.balance, 800)

    def test_insufficient_funds_transfer(self):
        url = reverse('transfer')
        data = {
            "from_account": self.account_2.id,
            "to_account": self.account_1.id,
            "amount": 600  # More than account_2's balance
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.account_1.refresh_from_db()
        self.account_2.refresh_from_db()
        self.assertEqual(self.account_1.balance, 1000)
        self.assertEqual(self.account_2.balance, 500)

    def test_view_balance_and_transaction_history(self):
        # Perform a deposit and withdrawal for transaction history testing
        self.client.post(reverse('deposit', args=[self.account_1.id]), {"amount": 200}, format='json')
        self.client.post(reverse('withdraw', args=[self.account_1.id]), {"amount": 100}, format='json')

        url = reverse('balance', args=[self.account_1.id])
        response = self.client.get(url, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['balance'], 1100)  # Initial 1000 + 200 - 100
        self.assertEqual(len(response.data['transactions']), 2)
        self.assertEqual(response.data['transactions'][0]['transaction_type'], 'deposit')
        self.assertEqual(response.data['transactions'][1]['transaction_type'], 'withdraw')
