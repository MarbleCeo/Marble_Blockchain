"""
Transaction Module for Blockchain System

This module defines the Transaction class, which is the fundamental unit of data exchange in the blockchain system.
Each transaction represents a transfer of value or data from one entity to another, secured through cryptographic
signatures and hashing.

Features:
- Unique transaction identification through UUIDs
- Cryptographic security via SHA-256 hashing
- Transaction validation including input verification
- Support for arbitrary data payloads
- Timestamp-based transaction ordering

Usage:
    from blockchain.core.transaction import Transaction
    
    # Create a new transaction
    tx = Transaction(sender="Alice", recipient="Bob", amount=10.0, data={"notes": "Payment for services"})
    
    # Compute and verify transaction hash
    tx_hash = tx.calculate_hash()
    is_valid = tx.validate()

The Transaction class serves as a building block for blocks in the blockchain and provides
methods for validation, hashing, and serialization.
"""

import uuid
import time
import hashlib
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Transaction:
    """
    Represents a transaction within the blockchain.
    
    Each transaction has a unique identifier, sender, recipient, amount, 
    and optional additional data. Transactions are validated and secured 
    through cryptographic hashing.
    
    Attributes:
        id (str): Unique identifier for the transaction (UUID)
        sender (str): The sender's identifier (e.g., public key or address)
        recipient (str): The recipient's identifier
        amount (float): The amount being transferred
        timestamp (float): Unix timestamp when the transaction was created
        data (Dict[str, Any]): Additional transaction data
        signature (Optional[str]): Digital signature for transaction verification
        hash (Optional[str]): SHA-256 hash of the transaction
    """
    
    sender: str
    recipient: str
    amount: float
    data: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    signature: Optional[str] = None
    hash: Optional[str] = None
    
    def __post_init__(self):
        """Initialize the transaction hash after creation."""
        self.hash = self.calculate_hash()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the transaction to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary containing transaction data
        """
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "amount": self.amount,
            "timestamp": self.timestamp,
            "data": self.data,
            "signature": self.signature
        }
    
    def calculate_hash(self) -> str:
        """
        Calculate the SHA-256 hash of the transaction.
        
        The hash is based on the transaction's id, sender, recipient,
        amount, timestamp, and data.
        
        Returns:
            str: Hexadecimal string representation of the transaction hash
        """
        # Create a dictionary of the transaction data to hash
        tx_dict = {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "amount": self.amount,
            "timestamp": self.timestamp,
            "data": self.data
        }
        
        # Convert the dictionary to a JSON string and encode it
        tx_string = json.dumps(tx_dict, sort_keys=True).encode()
        
        # Calculate and return the SHA-256 hash
        return hashlib.sha256(tx_string).hexdigest()
    
    def sign(self, signature: str) -> None:
        """
        Sign the transaction with a digital signature.
        
        Args:
            signature (str): Digital signature to apply to the transaction
        """
        self.signature = signature
    
    def validate(self) -> bool:
        """
        Validate the transaction.
        
        Checks if:
        - The transaction has a valid structure
        - The sender is not empty
        - The transaction amount is positive
        - The calculated hash matches the stored hash
        
        Returns:
            bool: True if the transaction is valid, False otherwise
        """
        # Check sender
        if not self.sender:
            return False
        
        # Check amount
        if self.amount <= 0:
            return False
        
        # Verify hash integrity
        if self.hash != self.calculate_hash():
            return False
        
        # Transaction is valid
        return True
    
    @classmethod
    def from_dict(cls, tx_dict: Dict[str, Any]) -> 'Transaction':
        """
        Create a Transaction object from a dictionary.
        
        Args:
            tx_dict (Dict[str, Any]): Dictionary containing transaction data
            
        Returns:
            Transaction: A new Transaction instance
        """
        tx = cls(
            sender=tx_dict.get("sender", ""),
            recipient=tx_dict.get("recipient", ""),
            amount=tx_dict.get("amount", 0.0),
            data=tx_dict.get("data", {})
        )
        
        # Set additional fields if they exist in the dictionary
        if "id" in tx_dict:
            tx.id = tx_dict["id"]
        if "timestamp" in tx_dict:
            tx.timestamp = tx_dict["timestamp"]
        if "signature" in tx_dict:
            tx.signature = tx_dict["signature"]
        if "hash" in tx_dict:
            tx.hash = tx_dict["hash"]
        else:
            tx.hash = tx.calculate_hash()
            
        return tx

