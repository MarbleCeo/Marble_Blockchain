"""
Block Module - Core component of the blockchain system

This module defines the Block class, which represents a fundamental structure in the blockchain.
Each block contains:
- An index identifying its position in the chain
- A timestamp of when the block was created
- A list of transactions included in the block
- A reference to the previous block's hash, maintaining the chain's integrity
- A Proof of History (PoH) hash for temporal verification
- A Merkle tree for efficient transaction verification

Blocks are cryptographically secured through hashing algorithms and include
validation mechanisms to ensure data integrity throughout the blockchain.
The Block class supports:
- Transaction verification using Merkle trees
- Block validation against previous blocks
- Hash integrity checks
- Conversion to serializable format for storage and transmission
"""

import hashlib
import json
import time
from typing import List, Dict, Any, Optional

from .merkle_tree import MerkleTree
from .transaction import Transaction


class Block:
    """
    Represents a block in the blockchain.
    Includes Merkle tree support and PoH integration.
    """
    
    def __init__(
        self, 
        index: int, 
        transactions: List[Transaction], 
        previous_hash: str,
        timestamp: float = None,
        poh_hash: str = None
    ):
        self.index = index
        self.timestamp = timestamp or time.time()
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.poh_hash = poh_hash
        self.tx_count = len(transactions)
        
        # Convert transactions to dicts for Merkle tree
        tx_dicts = [tx.to_dict() for tx in transactions]
        self.merkle_tree = MerkleTree(tx_dicts)
        self.merkle_root = self.merkle_tree.get_root()
        
        # Calculate block hash
        self.hash = self.calculate_hash()
        
    def calculate_hash(self) -> str:
        """Calculate the hash of the block."""
        block_data = {
            'index': self.index,
            'timestamp': self.timestamp,
            'merkle_root': self.merkle_root,
            'previous_hash': self.previous_hash,
            'poh_hash': self.poh_hash,
            'tx_count': self.tx_count
        }
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary format."""
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'merkle_root': self.merkle_root,
            'previous_hash': self.previous_hash,
            'poh_hash': self.poh_hash,
            'hash': self.hash,
            'tx_count': self.tx_count,
            'transactions': [tx.to_dict() for tx in self.transactions]
        }
    
    def is_valid(self, previous_block: Optional['Block'] = None) -> bool:
        """
        Validate the block.
        Checks hash integrity, transaction validity, and link to previous block.
        """
        # Verify block hash
        if self.hash != self.calculate_hash():
            return False
            
        # Verify link to previous block if provided
        if previous_block and self.previous_hash != previous_block.hash:
            return False
            
        # Verify all transactions are valid
        for tx in self.transactions:
            if not tx.is_valid():
                return False
                
        # Verify Merkle root
        tx_dicts = [tx.to_dict() for tx in self.transactions]
        merkle_tree = MerkleTree(tx_dicts)
        if self.merkle_root != merkle_tree.get_root():
            return False
            
        return True
    
    def verify_transaction(self, tx: Transaction) -> bool:
        """Verify if a transaction is included in this block."""
        return self.merkle_tree.verify_transaction(tx.to_dict())

