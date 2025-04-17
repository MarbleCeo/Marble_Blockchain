"""
Core module for the high-throughput blockchain system.

This module contains the fundamental building blocks of the blockchain:
- Block: The basic unit of the blockchain that contains transactions
- Transaction: Represents transfers or actions within the blockchain
- MerkleTree: Data structure for efficient verification of transaction integrity
- ProofOfHistory: Consensus mechanism for high-throughput blockchain operations

These components work together to provide a secure, efficient, and scalable
blockchain implementation.
"""

from .block import Block
from .transaction import Transaction
from .merkle_tree import MerkleTree
from .proof_of_history import ProofOfHistory

__version__ = '0.1.0'
__author__ = 'Blockchain Development Team'
__all__ = ['Block', 'Transaction', 'MerkleTree', 'ProofOfHistory']

