"""
Database models for the blockchain system.

This module defines SQLAlchemy ORM models for the blockchain entities including
blocks, transactions, and related data. It provides a unified schema for 
storing blockchain data with proper relationships and indexes for performance.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Set, Union
import datetime
import json
import uuid

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Index,
    Text, LargeBinary, UniqueConstraint, func, event
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates, Session
from sqlalchemy.ext.hybrid import hybrid_property

Base = declarative_base()


class Block(Base):
    """
    Represents a block in the blockchain.
    
    Each block contains a set of transactions, references to the previous block,
    and proof-of-history data for consensus validation.
    """
    __tablename__ = 'blocks'

    # Primary fields
    id = Column(Integer, primary_key=True)
    block_hash = Column(String(64), unique=True, nullable=False, index=True)
    previous_hash = Column(String(64), nullable=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False, index=True)
    nonce = Column(String(64), nullable=False)
    difficulty = Column(Integer, nullable=False)
    merkle_root = Column(String(64), nullable=False)
    
    # PoH fields
    poh_sequence = Column(Text, nullable=False)
    poh_iterations = Column(Integer, nullable=False)
    
    # Metadata
    version = Column(String(10), nullable=False, default="1.0")
    size = Column(Integer, nullable=False)  # Size in bytes
    height = Column(Integer, nullable=False, index=True)  # Block height in the chain
    
    # Relationships
    transactions = relationship("Transaction", back_populates="block", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_blocks_height_hash', 'height', 'block_hash'),
        Index('idx_blocks_timestamp', 'timestamp'),
    )
    
    def __repr__(self) -> str:
        return f"<Block(hash='{self.block_hash[:8]}...', height={self.height})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the block to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary containing block data
        """
        return {
            'id': self.id,
            'block_hash': self.block_hash,
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp.isoformat(),
            'nonce': self.nonce,
            'difficulty': self.difficulty,
            'merkle_root': self.merkle_root,
            'poh_sequence': self.poh_sequence,
            'poh_iterations': self.poh_iterations,
            'version': self.version,
            'size': self.size,
            'height': self.height,
            'transactions': [tx.to_dict() for tx in self.transactions]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], session: Optional[Session] = None) -> Block:
        """
        Create a Block instance from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing block data
            session (Optional[Session]): SQLAlchemy session for loading relationships
            
        Returns:
            Block: New Block instance
        """
        # Handle nested transactions if present
        transactions_data = data.pop('transactions', [])
        
        # Convert timestamp string to datetime if needed
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.datetime.fromisoformat(data['timestamp'])
            
        block = cls(**{k: v for k, v in data.items() if k != 'id'})
        
        # Add transactions if provided and session exists
        if session and transactions_data:
            for tx_data in transactions_data:
                tx = Transaction.from_dict(tx_data)
                block.transactions.append(tx)
                
        return block


class Transaction(Base):
    """
    Represents a transaction in the blockchain.
    
    Transactions record transfers or operations within the blockchain system,
    including sender, recipient, amount, and additional data.
    """
    __tablename__ = 'transactions'
    
    # Primary fields
    id = Column(Integer, primary_key=True)
    tx_hash = Column(String(64), unique=True, nullable=False, index=True)
    sender = Column(String(64), nullable=False, index=True)
    recipient = Column(String(64), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False, index=True)
    signature = Column(String(128), nullable=True)
    
    # Foreign keys
    block_id = Column(Integer, ForeignKey('blocks.id'), nullable=True)
    
    # Status fields
    is_confirmed = Column(Boolean, default=False, nullable=False)
    confirmations = Column(Integer, default=0, nullable=False)
    fee = Column(Float, default=0.0, nullable=False)
    
    # Additional data (JSON encoded)
    data = Column(Text, nullable=True)
    
    # Relationships
    block = relationship("Block", back_populates="transactions")
    
    # Indexes
    __table_args__ = (
        Index('idx_tx_block_hash', 'block_id', 'tx_hash'),
        Index('idx_tx_sender_recipient', 'sender', 'recipient'),
        Index('idx_tx_timestamp', 'timestamp'),
    )
    
    def __repr__(self) -> str:
        return f"<Transaction(hash='{self.tx_hash[:8]}...', amount={self.amount})>"
    
    @validates('data')
    def validate_data(self, key, value):
        """Ensure data is stored as JSON string"""
        if isinstance(value, dict):
            return json.dumps(value)
        return value
    
    @property
    def data_json(self) -> Dict[str, Any]:
        """Parse and return data as JSON object"""
        if self.data:
            return json.loads(self.data)
        return {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the transaction to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary containing transaction data
        """
        return {
            'id': self.id,
            'tx_hash': self.tx_hash,
            'sender': self.sender,
            'recipient': self.recipient,
            'amount': self.amount,
            'timestamp': self.timestamp.isoformat(),
            'signature': self.signature,
            'block_id': self.block_id,
            'is_confirmed': self.is_confirmed,
            'confirmations': self.confirmations,
            'fee': self.fee,
            'data': self.data_json
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Transaction:
        """
        Create a Transaction instance from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing transaction data
            
        Returns:
            Transaction: New Transaction instance
        """
        # Handle data field conversion
        if 'data' in data and isinstance(data['data'], dict):
            data['data'] = json.dumps(data['data'])
            
        # Convert timestamp string to datetime if needed
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.datetime.fromisoformat(data['timestamp'])
            
        return cls(**{k: v for k, v in data.items() if k != 'id'})


class Wallet(Base):
    """
    Represents a wallet in the blockchain system.
    
    Wallets store user's address, public key, and related information.
    They track balances and transaction history.
    """
    __tablename__ = 'wallets'
    
    id = Column(Integer, primary_key=True)
    address = Column(String(64), unique=True, nullable=False, index=True)
    public_key = Column(String(512), nullable=True)  # Optional for watch-only wallets
    label = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    last_updated = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Optional metadata
    metadata = Column(Text, nullable=True)  # JSON encoded data
    
    # Indexes
    __table_args__ = (
        Index('idx_wallet_address', 'address'),
    )
    
    def __repr__(self) -> str:
        return f"<Wallet(address='{self.address[:8]}...', label='{self.label}')>"
    
    @validates('metadata')
    def validate_metadata(self, key, value):
        """Ensure metadata is stored as JSON string"""
        if isinstance(value, dict):
            return json.dumps(value)
        return value
    
    @property
    def metadata_json(self) -> Dict[str, Any]:
        """Parse and return metadata as JSON object"""
        if self.metadata:
            return json.loads(self.metadata)
        return {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the wallet to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary containing wallet data
        """
        return {
            'id': self.id,
            'address': self.address,
            'public_key': self.public_key,
            'label': self.label,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'metadata': self.metadata_json
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Wallet:
        """
        Create a Wallet instance from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing wallet data
            
        Returns:
            Wallet: New Wallet instance
        """
        # Handle metadata field conversion
        if 'metadata' in data and isinstance(data['metadata'], dict):
            data['metadata'] = json.dumps(data['metadata'])
            
        # Convert timestamp strings to datetime if needed
        for field in ['created_at', 'last_updated']:
            if isinstance(data.get(field), str):
                data[field] = datetime.datetime.fromisoformat(data[field])
                
        return cls(**{k: v for k, v in data.items() if k != 'id'})


class PeerNode(Base):
    """
    Represents a peer node in the blockchain network.
    
    Tracks information about other nodes in the network including
    connection status, capabilities, and performance metrics.
    """
    __tablename__ = 'peer_nodes'
    
    id = Column(Integer, primary_key=True)
    node_id = Column(String(64), unique=True, nullable=False, index=True)
    ip_address = Column(String(45), nullable=False)  # IPv6 compatible
    port = Column(Integer, nullable=False)
    last_seen = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Node capabilities and version
    version = Column(String(20), nullable=True)
    services = Column(String(100), nullable=True)  # Comma-separated list of services
    
    # Performance metrics
    latency_ms = Column(Integer, nullable=True)
    trust_score = Column(Float, default=0.0, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_node_active', 'is_active'),
        UniqueConstraint('ip_address', 'port', name='uq_node_address'),
    )
    
    def __repr__(self) -> str:
        return f"<PeerNode(node_id='{self.node_id[:8]}...', address='{self.ip_address}:{self.port}')>"
    
    @property
    def services_list(self) -> List[str]:
        """Return services as a list of strings"""
        if self.services:
            return self.services.split(',')
        return []
    
    @services_list.setter
    def services_list(self, value: List[str]) -> None:
        """Set services from a list of strings"""
        if value:
            self.services = ','.join(value)
        else:
            self.services = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the peer node to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary containing peer node data
        """
        return {
            'id': self.id,
            'node_id': self.node_id,
            'ip_address': self.ip_address,
            'port': self.port,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'is_active': self.is_active,
            'version': self.version,
            'services': self.services_list,
            'latency_ms': self.latency_ms,
            'trust_score': self.trust_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PeerNode:
        """
        Create a PeerNode instance from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing peer node data
            
        Returns:
            PeerNode: New PeerNode instance
        """
        node_data = data.copy()
        
        # Handle services conversion
        if 'services' in node_data and isinstance(node_data['services'], list):
            node_data['services'] = ','.join(node_data['services'])
            
        # Convert timestamp string to datetime if needed
        if isinstance(node_data.get('last_seen'), str):
            node_data['last_seen'] = datetime.datetime.fromisoformat(node_data['last_seen'])
            
        return cls(**{k: v for k, v in node_data.items() if k != 'id'})


# Create event listeners and other model-related logic
@event.listens_for(Block, 'before_insert')
def set_block_timestamp(mapper, connection, block):
    """Ensure block timestamp is set if not provided"""
    if block.timestamp is None:
        block.timestamp = datetime.datetime.utcnow()

