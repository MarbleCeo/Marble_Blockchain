import hashlib
import json
import sqlite3
import time
from typing import Dict, List, Optional, Any, Union, Tuple


class Transaction:
    """
    Represents a transaction in the blockchain.
    
    A transaction contains sender, recipient, and amount information,
    along with a timestamp and optional signature.
    """
    
    def __init__(
        self, 
        sender: str, 
        recipient: str, 
        amount: float,
        timestamp: Optional[float] = None,
        signature: Optional[str] = None
    ):
        """
        Initialize a new transaction.
        
        Args:
            sender: The address of the sender
            recipient: The address of the recipient
            amount: The amount being transferred
            timestamp: Transaction creation time (default: current time)
            signature: Digital signature for verification (optional)
        """
        self.sender = sender
        self.recipient = recipient
        self.amount = amount
        self.timestamp = timestamp or time.time()
        self.signature = signature
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the transaction to a dictionary.
        
        Returns:
            Dict containing transaction data
        """
        return {
            'sender': self.sender,
            'recipient': self.recipient,
            'amount': self.amount,
            'timestamp': self.timestamp,
            'signature': self.signature
        }
    
    @classmethod
    def from_dict(cls, transaction_dict: Dict[str, Any]) -> 'Transaction':
        """
        Create a Transaction object from a dictionary.
        
        Args:
            transaction_dict: Dictionary containing transaction data
            
        Returns:
            Transaction object
        """
        return cls(
            sender=transaction_dict['sender'],
            recipient=transaction_dict['recipient'],
            amount=transaction_dict['amount'],
            timestamp=transaction_dict.get('timestamp'),
            signature=transaction_dict.get('signature')
        )
    
    def calculate_hash(self) -> str:
        """
        Calculate a hash of the transaction data.
        
        Returns:
            SHA256 hash of the transaction data
        """
        transaction_string = json.dumps(
            {
                'sender': self.sender,
                'recipient': self.recipient,
                'amount': self.amount,
                'timestamp': self.timestamp
            }, 
            sort_keys=True
        )
        return hashlib.sha256(transaction_string.encode()).hexdigest()
    
    def is_valid(self) -> bool:
        """
        Validate the transaction.
        
        This is a simplified validation that checks:
        - Amount is positive
        - Sender and recipient are not empty
        - Sender and recipient are different addresses
        
        In a real implementation, this would include signature verification.
        
        Returns:
            True if the transaction is valid, False otherwise
        """
        # Simplified validation rules
        if self.amount <= 0:
            return False
        
        if not self.sender or not self.recipient:
            return False
        
        if self.sender == self.recipient:
            return False
            
        # In a complete implementation, we would verify the signature here
        return True


class Block:
    """
    Represents a block in the blockchain.
    
    Each block contains a list of transactions, a timestamp, the hash of the
    previous block, a nonce for proof-of-work, and its own hash.
    """
    
    def __init__(
        self, 
        index: int, 
        transactions: List[Transaction], 
        timestamp: Optional[float] = None, 
        previous_hash: str = '', 
        nonce: int = 0, 
        hash: str = ''
    ):
        """
        Initialize a new block.
        
        Args:
            index: The position of the block in the chain
            transactions: List of transactions included in the block
            timestamp: Block creation time (default: current time)
            previous_hash: Hash of the previous block
            nonce: Value used for proof-of-work
            hash: The block's own hash (calculated if not provided)
        """
        self.index = index
        self.transactions = transactions
        self.timestamp = timestamp or time.time()
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = hash or self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """
        Calculate the hash of the block.
        
        Returns:
            SHA256 hash of the block data
        """
        # Convert transactions to a serializable format
        transactions_dict = [tx.to_dict() for tx in self.transactions]
        
        block_string = json.dumps({
            'index': self.index,
            'transactions': transactions_dict,
            'timestamp': self.timestamp,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }, sort_keys=True)
        
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty: int) -> None:
        """
        Mine the block with proof-of-work.
        
        Adjusts the nonce until the hash begins with the required number of zeros.
        
        Args:
            difficulty: Number of leading zeros required in the hash
        """
        target = '0' * difficulty
        
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the block to a dictionary.
        
        Returns:
            Dict containing block data
        """
        return {
            'index': self.index,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'timestamp': self.timestamp,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'hash': self.hash
        }
    
    @classmethod
    def from_dict(cls, block_dict: Dict[str, Any]) -> 'Block':
        """
        Create a Block object from a dictionary.
        
        Args:
            block_dict: Dictionary containing block data
            
        Returns:
            Block object
        """
        transactions = [
            Transaction.from_dict(tx_dict) 
            for tx_dict in block_dict['transactions']
        ]
        
        return cls(
            index=block_dict['index'],
            transactions=transactions,
            timestamp=block_dict['timestamp'],
            previous_hash=block_dict['previous_hash'],
            nonce=block_dict['nonce'],
            hash=block_dict['hash']
        )
    
    def is_valid(self) -> bool:
        """
        Validate the block.
        
        Checks if:
        - The block's hash is correctly calculated
        - All transactions in the block are valid
        
        Returns:
            True if the block is valid, False otherwise
        """
        if self.hash != self.calculate_hash():
            return False
            
        # Validate all transactions in the block
        return all(transaction.is_valid() for transaction in self.transactions)


class Blockchain:
    """
    Represents a blockchain with both in-memory storage and SQLite persistence.
    
    The blockchain consists of a chain of blocks, each containing multiple transactions.
    It provides methods for adding blocks, mining, and validating the chain.
    """
    
    def __init__(self, db_path: str = 'blockchain.db', difficulty: int = 4):
        """
        Initialize the blockchain.
        
        Args:
            db_path: Path to the SQLite database file
            difficulty: Mining difficulty (number of leading zeros required)
        """
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.difficulty = difficulty
        self.db_path = db_path
        self.mining_reward = 10.0
        
        # Initialize the database
        self._initialize_db()
        
        # Load the chain from the database or create the genesis block
        if not self._load_chain_from_db():
            self._create_genesis_block()
    
    def _initialize_db(self) -> None:
        """
        Initialize the SQLite database with required tables.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create blocks table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS blocks (
                index INTEGER PRIMARY KEY,
                timestamp REAL,
                previous_hash TEXT,
                hash TEXT,
                nonce INTEGER,
                data TEXT
            )
            ''')
            
            # Create transactions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                block_index INTEGER,
                sender TEXT,
                recipient TEXT,
                amount REAL,
                timestamp REAL,
                signature TEXT,
                FOREIGN KEY (block_index) REFERENCES blocks(index)
            )
            ''')
            
            # Create pending transactions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS pending_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT,
                recipient TEXT,
                amount REAL,
                timestamp REAL,
                signature TEXT
            )
            ''')
            
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            if conn:
                conn.close()
    
    def _load_chain_from_db(self) -> bool:
        """
        Load the blockchain from the database.
        
        Returns:
            True if the chain was loaded successfully, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if there are blocks in the database
            cursor.execute("SELECT COUNT(*) FROM blocks")
            count = cursor.fetchone()[0]
            
            if count == 0:
                return False
                
            # Load blocks
            cursor.execute("SELECT * FROM blocks ORDER BY index")
            blocks_data = cursor.fetchall()
            
            self.chain = []
            
            for block_data in blocks_data:
                index, timestamp, previous_hash, hash, nonce, data_json = block_data
                
                # Get transactions for this block
                cursor.execute(
                    "SELECT sender, recipient, amount, timestamp, signature FROM transactions WHERE block_index = ?", 
                    (index,)
                )
                transactions_data = cursor.fetchall()
                
                transactions = []
                for tx_data in transactions_data:
                    sender, recipient, amount, tx_timestamp, signature = tx_data
                    transaction = Transaction(
                        sender=sender,
                        recipient=recipient,
                        amount=amount,
                        timestamp=tx_timestamp,
                        signature=signature
                    )
                    transactions.append(transaction)
                
                block = Block(
                    index=index,
                    transactions=transactions,
                    timestamp=timestamp,
                    previous_hash=previous_hash,
                    nonce=nonce,
                    hash=hash
                )
                
                self.chain.append(block)
            
            # Load pending transactions
            cursor.execute(
                "SELECT sender, recipient, amount, timestamp, signature FROM pending_transactions"
            )
            pending_tx_data = cursor.fetchall()
            
            self.pending_transactions = []
            
            for tx_data in pending_tx_data:
                sender, recipient, amount, timestamp, signature = tx_data
                transaction = Transaction(
                    sender=sender,
                    recipient=recipient,
                    amount=amount,
                    timestamp=timestamp,
                    signature=signature
                )
                self.pending_transactions.append(transaction)
                
            return True
                
        except sqlite3.Error as e:
            print(f"Error loading blockchain from database: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def _create_genesis_block(self) -> None:
        """
        Create the genesis block and save it to the database.
        """
        genesis_block = Block(
            index=0,
            transactions=[],
            timestamp=time.time(),
            previous_hash="0"
        )
        
        self.chain.append(genesis_block)
        self._save_block_to_db(genesis_block)
    
    def _save_block_to_db(self, block: Block) -> None:
        """
        Save a block to the database.
        
        Args:
            block: The Block object to save
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Save block data
            data_json = json.dumps(block.to_dict())
            cursor.execute(
                "INSERT OR REPLACE INTO blocks VALUES (?, ?, ?, ?, ?, ?)",
                (block.index, block.timestamp, block.previous_hash, block.hash, block.nonce, data_json)
            )
            
            # Save transactions
            for transaction in block.transactions:
                cursor.execute(
                    "INSERT INTO transactions (block_index, sender, recipient, amount, timestamp, signature) VALUES (?, ?, ?, ?, ?, ?)",
                    (block.index, transaction.sender, transaction.recipient, transaction.amount, transaction.timestamp, transaction.signature)
                )
            
            conn.commit()
        except sqlite3.Error as e:
            print(f"Error saving block to database: {e}")
        finally:
            if conn:
                conn.close()
    
    def _save_pending_transaction(self, transaction: Transaction) -> None:
        """
        Save a pending transaction to the database.
        
        Args:
            transaction: The Transaction object to save
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO pending_transactions (sender, recipient, amount, timestamp, signature) VALUES (?, ?, ?, ?, ?)",
                (transaction.sender, transaction.recipient, transaction.amount, transaction.timestamp, transaction.signature)
            )
            
            conn.commit()
        except sqlite3.Error as e:
            print(f"Error saving pending transaction to database: {e}")
        finally:
            if conn:
                conn.close()
    
    def _clear_pending_transactions(self) -> None:
        """
        Clear all pending transactions from the database.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM pending_transactions")
            
            conn.commit()
        except sqlite3.Error as e:
            print(f"Error clearing pending transactions: {e}")
        finally:
            if conn:
                conn.close()
    
    def get_latest_block(self) -> Block:
        """
        Get the most recent block in the chain.
        
        Returns:
            The latest Block object
        """
        return self.chain[-1]
    
    def add_transaction(self, transaction: Transaction) -> bool:
        """
        Add a transaction to the pending transactions list.
        
        Args:
            transaction: The Transaction object to add
            
        Returns:
            True if the transaction was added successfully, False otherwise
        """
        if not transaction.is_valid():
            print("Transaction validation failed")
            return False
            
        self.pending_transactions.append(transaction)
        self._save_pending_transaction(transaction)
        return True

    def mine_pending_transactions(self, miner_address: str) -> Block:
        """
        Mine a new block with all pending transactions.
        
        Creates a mining reward transaction for the miner, adds it to the pending
        transactions, creates a new block, mines it using proof-of-work, adds it
        to the chain, and clears the pending transactions.
        
        Args:
            miner_address: The address to receive the mining reward
            
        Returns:
            The newly mined Block
        """
        # Create mining reward transaction
        reward_transaction = Transaction(
            sender="SYSTEM",
            recipient=miner_address,
            amount=self.mining_reward
        )
        self.pending_transactions.append(reward_transaction)
        
        # Create new block
        new_block = Block(
            index=len(self.chain),
            transactions=self.pending_transactions,
            previous_hash=self.get_latest_block().hash
        )
        
        # Mine the block
        new_block.mine_block(self.difficulty)
        
        # Add block to the chain
        self.chain.append(new_block)
        self._save_block_to_db(new_block)
        
        # Clear pending transactions
        self.pending_transactions = []
        self._clear_pending_transactions()
        
        return new_block
    
    def get_balance(self, address: str) -> float:
        """
        Calculate the balance for a given address.
        
        Traverses the entire blockchain and sums all transactions
        where the address is involved, either as sender or recipient.
        
        Args:
            address: The address to calculate the balance for
            
        Returns:
            The current balance of the address
        """
        balance = 0.0
        
        # Iterate through all blocks in the chain
        for block in self.chain:
            # Iterate through all transactions in the block
            for transaction in block.transactions:
                # If the address is the recipient, add the amount to the balance
                if transaction.recipient == address:
                    balance += transaction.amount
                
                # If the address is the sender, subtract the amount from the balance
                if transaction.sender == address:
                    balance -= transaction.amount
        
        return balance
    
    def is_chain_valid(self) -> bool:
        """
        Validate the entire blockchain.
        
        Checks that:
        - Each block has a valid hash
        - Each block's previous_hash points to the hash of the previous block
        - Each block has been properly mined (has the right number of leading zeros)
        - Each transaction in each block is valid
        
        Returns:
            True if the blockchain is valid, False otherwise
        """
        # Start from the second block (index 1)
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Check if the block's hash is valid
            if current_block.hash != current_block.calculate_hash():
                print(f"Invalid hash for block {i}")
                return False
            
            # Check if the previous_hash is correct
            if current_block.previous_hash != previous_block.hash:
                print(f"Invalid previous hash for block {i}")
                return False
            
            # Check if the block has been mined properly
            if not current_block.hash.startswith('0' * self.difficulty):
                print(f"Block {i} has not been mined properly")
                return False
            
            # Validate all transactions in the block
            for transaction in current_block.transactions:
                if not transaction.is_valid():
                    # Skip validation for mining reward transactions
                    if transaction.sender != "SYSTEM":
                        print(f"Invalid transaction found in block {i}")
                        return False
        
        return True

