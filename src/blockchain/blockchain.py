from time import time
from json import dumps
from hashlib import sha256

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
        value = str(self.index) + self.previous_hash + str(self.timestamp) + dumps(self.data) + str(self.nonce) + self.miner_address + str(self.miner_reward)
        return sha256(value.encode()).hexdigest()

    def mine_block(self, difficulty):
        while not self.hash.startswith('0' * difficulty):
            self.nonce += 1
            self.hash = self.calculate_hash()

class Blockchain:
    def __init__(self):
        self.chain = []
        self.difficulty = 2
        self.pending_transactions = []

        MINING_REWARD = 10000  # Defina a recompensa de mineração aqui
        self.add_first_block(MINING_REWARD)

    def add_first_block(self, mining_reward):
        """ Adiciona o bloco genesis à blockchain """
        data = {'message': 'Genesis Block'}
        genesis_block = Block(0, '0', int(time()), data, sha256(dumps(data).encode()).hexdigest(), 0, None, mining_reward)
        self.chain.append(genesis_block)

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, new_block):
        new_block.previous_hash = self.get_latest_block().hash
        new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)

    def mine_pending_transactions(self, mining_reward_address):
        reward_transaction = Transaction(None, mining_reward_address, self.mining_reward)
        self.pending_transactions.append(reward_transaction)
        latest_block = self.get_latest_block()
        new_block = Block(latest_block.index + 1, latest_block.hash, int(time()), self.pending_transactions, None, 0, None, self.mining_reward)
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)
        self.pending_transactions = []

    def create_transaction(self, transaction):
        self.pending_transactions.append(transaction)

    def get_balance_of_address(self, address):
        balance = 0
        for block in self.chain:
            for transaction in block.data:
                if transaction.input is None:
                    continue
                if transaction.input.address == address:
                    balance -= transaction.input.amount
                if transaction.output.address == address:
                    balance += transaction.output.amount
        return balance

    def get_all_transactions(self):
        all_transactions = []
        for block in self.chain:
            all_transactions.extend(block.data)
        return all_transactions

class Transaction:
    def __init__(self, input, output, amount):
        self.input = input
        self.output = output
        self.amount = amount

    def is_valid(self, blockchain):
        if self.input is None:
            return True
        if self.input.address not in blockchain.get_balance_of_address(self.input.address):
            return False
        if self.amount > blockchain.get_balance_of_address(self.input.address):
            return False
        return True