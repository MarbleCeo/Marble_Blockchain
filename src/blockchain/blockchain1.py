import sqlite3
import time
import hashlib
from typing import List, Tuple

class Block:
    def __init__(self, index, previous_hash, timestamp, data, hash, nonce, miner_address, miner_reward):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.hash = hash
        self.nonce = nonce
        self.miner_address = miner_address
        self.miner_reward = miner_reward

    def calculate_hash(self):
        value = str(self.index) + self.previous_hash + str(self.timestamp) + str(self.data) + str(self.nonce) + self.miner_address + str(self.miner_reward)
        return hashlib.sha256(value.encode()).hexdigest()

    def mine_block(self, difficulty):
        while not self.hash.startswith('0' * difficulty):
            self.nonce += 1
            self.hash = self.calculate_hash()

class Blockchain:
    def __init__(self, db_name='blockchain.db'):
        self.chain = []
        self.difficulty = 2
        self.pending_transactions = []
        self.db_name = db_name

        self.initialize_database()
        self.add_first_block()

    def initialize_database(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('''CREATE TABLE IF NOT EXISTS transactions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            sender TEXT,
                            receiver TEXT,
                            amount REAL,
                            timestamp INTEGER
                        )''')

        conn.commit()
        conn.close()

    def add_first_block(self):
        """ Adiciona o bloco genesis à blockchain """
        data = {'message': 'Genesis Block'}
        genesis_block = Block(0, '0', int(time()), data, hashlib.sha256(dumps(data).encode()).hexdigest(), 0, None, 10000)
        self.chain.append(genesis_block)

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, new_block):
        new_block.previous_hash = self.get_latest_block().hash
        new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)

    def mine_pending_transactions(self, miner_address):
        reward_transaction = Transaction(None, miner_address, self.mining_reward)
        self.pending_transactions.append(reward_transaction)

        latest_block = self.get_latest_block()
        new_block = Block(latest_block.index + 1, latest_block.hash, int(time()), self.pending_transactions, None, 0, miner_address, self.mining_reward)
        new_block.mine_block(self.difficulty)
        self.add_block(new_block)

        self.store_transactions(self.pending_transactions)
        self.pending_transactions = []

    def create_transaction(self, transaction):
        if self.validate_transaction(transaction):
            self.pending_transactions.append(transaction)
            return True
        else:
            return False

    def get_balance_of_address(self, address):
        balance = 0
        transactions = self.get_all_transactions()

        for transaction in transactions:
            if transaction.input is None:
                continue
            if transaction.input.address == address:
                balance -= transaction.input.amount
            if transaction.output.address == address:
                balance += transaction.output.amount

        return balance

    def get_all_transactions(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM transactions')
        rows = cursor.fetchall()

        transactions = []

        for row in rows:
            transaction = Transaction(row[1], row[2], row[3], row[4])
            transactions.append(transaction)

        conn.close()

        return transactions

    def store_transactions(self, transactions):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        for transaction in transactions:
            cursor.execute('''INSERT INTO transactions (sender, receiver, amount, timestamp)
                              VALUES (?, ?, ?, ?)''', (transaction.sender, transaction.receiver, transaction.amount, int(time())))

        conn.commit()
        conn.close()

    def validate_transaction(self, transaction):
        if transaction.sender is None:
            return True

        sender_balance = self.get_balance_of_address(transaction.sender)

        if sender_balance < transaction.amount:
            return False

        return True

class Transaction:
    def __init__(self, sender, receiver, amount, timestamp=None):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.timestamp = timestamp or int(time())

    def is_valid(self, blockchain):
        if self.sender is None:
            return True
        if self.sender not in blockchain.get_balance_of_address(self.sender):
            return False
        if self.amount > blockchain.get_balance_of_address(self.sender):
            return False
        return True
