"""
Merkle Tree Implementation Module

This module provides a Merkle Tree data structure implementation for efficient transaction verification
in the blockchain system. A Merkle Tree allows for O(log n) verification of whether a transaction is
included in a block, making it an essential component for blockchain scalability and security.

The MerkleTree class creates a binary tree of hashes where:
- Leaf nodes are hashes of individual transactions
- Non-leaf nodes are hashes of their child nodes combined
- The root hash represents a cryptographic summary of all transactions

This implementation enables:
- Efficient verification of transaction inclusion
- Tamper-resistant transaction sets
- Reduced data verification overhead
"""

import hashlib
import json
from typing import List, Dict, Any


class MerkleTree:
    """
    Merkle Tree implementation for efficient transaction verification.
    Allows for O(log n) verification of transactions within a block.
    """
    
    def __init__(self, transactions: List[Dict[str, Any]]):
        self.transactions = transactions
        self.leaves = [self._hash_transaction(tx) for tx in transactions]
        self.root = self._build_tree(self.leaves)
        
    def _hash_transaction(self, transaction: Dict[str, Any]) -> str:
        """Hash a transaction using SHA-256."""
        tx_string = json.dumps(transaction, sort_keys=True)
        return hashlib.sha256(tx_string.encode()).hexdigest()
    
    def _build_tree(self, leaves: List[str]) -> str:
        """Build the Merkle tree and return the root hash."""
        if not leaves:
            return hashlib.sha256("".encode()).hexdigest()
        
        if len(leaves) == 1:
            return leaves[0]
        
        # If odd number of leaves, duplicate the last one
        if len(leaves) % 2 != 0:
            leaves.append(leaves[-1])
            
        # Create next level of the tree
        next_level = []
        for i in range(0, len(leaves), 2):
            combined = leaves[i] + leaves[i+1]
            next_level.append(hashlib.sha256(combined.encode()).hexdigest())
        
        # Recursively build the tree
        return self._build_tree(next_level)
    
    def get_root(self) -> str:
        """Return the Merkle root."""
        return self.root
    
    def verify_transaction(self, transaction: Dict[str, Any]) -> bool:
        """Verify if a transaction is part of the Merkle tree."""
        tx_hash = self._hash_transaction(transaction)
        return tx_hash in self.leaves

