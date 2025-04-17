"""
Database repositories for blockchain entities.

This module implements repository pattern for all blockchain related data access.
Each repository encapsulates CRUD operations and specialized queries for a specific model,
hiding the underlying implementation details from the service layer.
"""

from enum import Enum, auto
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import logging
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc

from .models import Block, Transaction, Wallet, PeerNode
from .connection import get_session

# Setup logging
logger = logging.getLogger(__name__)

class TransactionStatus(Enum):
    """Transaction status enum representing the lifecycle of a transaction."""
    PENDING = auto()
    CONFIRMED = auto()
    REJECTED = auto()
    EXPIRED = auto()
    INCLUDED = auto()  # Included in a block

class BlockStatus(Enum):
    """Block status enum representing the lifecycle of a block."""
    PENDING = auto()
    CONFIRMED = auto()
    REJECTED = auto()
    ORPHANED = auto()
    FINALIZED = auto()

class BaseRepository:
    """Base repository with common CRUD operations for all models."""

    def __init__(self, session: Optional[Session] = None):
        """
        Initialize repository with an optional session.

        Args:
            session: SQLAlchemy session, if None a new session will be created
        """
        self.session = session or get_session()
        self._model = None

    def _check_model(self):
        """Check if model is set for the repository."""
        if self._model is None:
            raise NotImplementedError("Model is not set for this repository")

    def create(self, **kwargs) -> Any:
        """
        Create a new entity.

        Args:
            **kwargs: Fields for the entity

        Returns:
            The created entity
        """
        self._check_model()
        entity = self._model(**kwargs)
        self.session.add(entity)
        self.session.commit()
        return entity

    def get_by_id(self, entity_id: Any) -> Optional[Any]:
        """
        Get entity by ID.

        Args:
            entity_id: The ID of the entity

        Returns:
            The entity if found, None otherwise
        """
        self._check_model()
        return self.session.query(self._model).filter(self._model.id == entity_id).first()

    def get_all(self) -> List[Any]:
        """
        Get all entities.

        Returns:
            List of all entities
        """
        self._check_model()
        return self.session.query(self._model).all()

    def update(self, entity_id: Any, **kwargs) -> Optional[Any]:
        """
        Update an entity.

        Args:
            entity_id: The ID of the entity to update
            **kwargs: Fields to update

        Returns:
            The updated entity if found, None otherwise
        """
        self._check_model()
        entity = self.get_by_id(entity_id)
        if entity:
            for key, value in kwargs.items():
                setattr(entity, key, value)
            self.session.commit()
        return entity

    def delete(self, entity_id: Any) -> bool:
        """
        Delete an entity.

        Args:
            entity_id: The ID of the entity to delete

        Returns:
            True if deleted, False if not found
        """
        self._check_model()
        entity = self.get_by_id(entity_id)
        if entity:
            self.session.delete(entity)
            self.session.commit()
            return True
        return False

    def close(self):
        """Close the session."""
        if self.session:
            self.session.close()


class BlockRepository(BaseRepository):
    """Repository for Block entities with specialized blockchain operations."""

    def __init__(self, session: Optional[Session] = None):
        """Initialize with Block model."""
        super().__init__(session)
        self._model = Block

    def get_by_hash(self, block_hash: str) -> Optional[Block]:
        """
        Get a block by its hash.

        Args:
            block_hash: The hash of the block

        Returns:
            The block if found, None otherwise
        """
        return self.session.query(Block).filter(Block.hash == block_hash).first()

    def get_by_height(self, height: int) -> Optional[Block]:
        """
        Get a block by its height.

        Args:
            height: The height of the block in the blockchain

        Returns:
            The block if found, None otherwise
        """
        return self.session.query(Block).filter(Block.height == height).first()

    def get_latest_block(self) -> Optional[Block]:
        """
        Get the latest block in the blockchain.

        Returns:
            The latest block if any, None otherwise
        """
        return self.session.query(Block).order_by(desc(Block.height)).first()

    def get_blocks_in_range(self, start_height: int, end_height: int) -> List[Block]:
        """
        Get blocks within a specified height range.

        Args:
            start_height: Start height (inclusive)
            end_height: End height (inclusive)

        Returns:
            List of blocks in the specified range
        """
        return (
            self.session.query(Block)
            .filter(and_(Block.height >= start_height, Block.height <= end_height))
            .order_by(Block.height)
            .all()
        )

    def get_blocks_by_status(self, status: BlockStatus) -> List[Block]:
        """
        Get blocks by their status.

        Args:
            status: The status of blocks to retrieve

        Returns:
            List of blocks with the specified status
        """
        return self.session.query(Block).filter(Block.status == status.value).all()

    def get_block_count(self) -> int:
        """
        Get the total number of blocks in the blockchain.

        Returns:
            Total number of blocks
        """
        return self.session.query(func.count(Block.id)).scalar() or 0

    def get_blocks_created_by(self, validator_id: str) -> List[Block]:
        """
        Get blocks created by a specific validator.

        Args:
            validator_id: The ID of the validator

        Returns:
            List of blocks created by the validator
        """
        return self.session.query(Block).filter(Block.validator_id == validator_id).all()

    def get_chain_difficulty(self) -> int:
        """
        Calculate the cumulative difficulty of the blockchain.

        Returns:
            The total difficulty (sum of all block difficulties)
        """
        return self.session.query(func.sum(Block.difficulty)).scalar() or 0


class TransactionRepository(BaseRepository):
    """Repository for Transaction entities with specialized query operations."""

    def __init__(self, session: Optional[Session] = None):
        """Initialize with Transaction model."""
        super().__init__(session)
        self._model = Transaction

    def get_by_hash(self, tx_hash: str) -> Optional[Transaction]:
        """
        Get a transaction by its hash.

        Args:
            tx_hash: The hash of the transaction

        Returns:
            The transaction if found, None otherwise
        """
        return self.session.query(Transaction).filter(Transaction.hash == tx_hash).first()

    def get_by_status(self, status: TransactionStatus) -> List[Transaction]:
        """
        Get transactions by their status.

        Args:
            status: The status of transactions to retrieve

        Returns:
            List of transactions with the specified status
        """
        return self.session.query(Transaction).filter(Transaction.status == status.value).all()

    def get_by_block_hash(self, block_hash: str) -> List[Transaction]:
        """
        Get all transactions in a specific block.

        Args:
            block_hash: The hash of the block

        Returns:
            List of transactions in the block
        """
        return self.session.query(Transaction).filter(Transaction.block_hash == block_hash).all()

    def get_by_address(self, address: str, is_sender: bool = None) -> List[Transaction]:
        """
        Get transactions by address (sender or receiver).

        Args:
            address: The wallet address
            is_sender: If True, get transactions where address is sender
                      If False, get transactions where address is receiver
                      If None, get all transactions involving the address

        Returns:
            List of transactions involving the address
        """
        if is_sender is None:
            return (
                self.session.query(Transaction)
                .filter(
                    or_(
                        Transaction.sender_address == address,
                        Transaction.receiver_address == address
                    )
                )
                .all()
            )
        elif is_sender:
            return (
                self.session.query(Transaction)
                .filter(Transaction.sender_address == address)
                .all()
            )
        else:
            return (
                self.session.query(Transaction)
                .filter(Transaction.receiver_address == address)
                .all()
            )

    def get_pending_transactions(self) -> List[Transaction]:
        """
        Get all pending transactions.

        Returns:
            List of pending transactions
        """
        return (
            self.session.query(Transaction)
            .filter(Transaction.status == TransactionStatus.PENDING.value)
            .all()
        )

    def get_by_time_range(self, start_time: datetime, end_time: datetime) -> List[Transaction]:
        """
        Get transactions within a specific time range.

        Args:
            start_time: Start datetime (inclusive)
            end_time: End datetime (inclusive)

        Returns:
            List of transactions in the specified time range
        """
        return (
            self.session.query(Transaction)
            .filter(and_(Transaction.timestamp >= start_time, Transaction.timestamp <= end_time))
            .all()
        )

    def get_transaction_count(self, status: Optional[TransactionStatus] = None) -> int:
        """
        Get the total number of transactions, optionally filtered by status.

        Args:
            status: If provided, count only transactions with this status

        Returns:
            Total number of transactions
        """
        query = self.session.query(func.count(Transaction.id))
        if status:
            query = query.filter(Transaction.status == status.value)
        return query.scalar() or 0

    def get_fee_statistics(self) -> Dict[str, float]:
        """
        Calculate statistics about transaction fees.

        Returns:
            Dictionary with min, max, and average fees
        """
        result = {
            "min_fee": self.session.query(func.min(Transaction.fee)).scalar() or 0,
            "max_fee": self.session.query(func.max(Transaction.fee)).scalar() or 0,
            "avg_fee": self.session.query(func.avg(Transaction.fee)).scalar() or 0,
        }
        return result


class WalletRepository(BaseRepository):
    """Repository for Wallet entities with specialized operations."""

    def __init__(self, session: Optional[Session] = None):
        """Initialize with Wallet model."""
        super().__init__(session)
        self._model = Wallet

    def get_by_address(self, address: str) -> Optional[Wallet]:
        """
        Get a wallet by its address.

        Args:
            address: The wallet address

        Returns:
            The wallet if found, None otherwise
        """
        return self.session.query(Wallet).filter(Wallet.address == address).first()

    def get_by_public_key(self, public_key: str) -> Optional[Wallet]:
        """
        Get a wallet by its public key.

        Args:
            public_key: The wallet's public key

        Returns:
            The wallet if found, None otherwise
        """
        return self.session.query(Wallet).filter(Wallet.public_key == public_key).first()

    def get_wallet_balance(self, address: str) -> float:
        """
        Get the balance of a wallet.

        Args:
            address: The wallet address

        Returns:
            The wallet balance
        """
        wallet = self.get_by_address(address)
        return wallet.balance if wallet else 0.0

    def update_balance(self, address: str, new_balance: float) -> Optional[Wallet]:
        """
        Update a wallet's balance.

        Args:
            address: The wallet address
            new_balance: The new balance

        Returns:
            The updated wallet if found, None otherwise
        """
        wallet = self.get_by_address(address)
        if wallet:
            wallet.balance = new_balance
            self.session.commit()
        return wallet

    def adjust_balance(self, address: str, amount: float) -> Optional[Wallet]:
        """
        Adjust a wallet's balance by adding/subtracting an amount.

        Args:
            address: The wallet address
            amount: The amount to adjust (positive to add, negative to subtract)

        Returns:
            The updated wallet if found, None otherwise
        """
        wallet = self.get_by_address(address)
        if wallet:
            wallet.balance += amount
            self.session.commit()
        return wallet

    def get_wallets_with_positive_balance(self) -> List[Wallet]:
        """
        Get all wallets with a positive balance.

        Returns:
            List of wallets with positive balance
        """
        return self.session.query(Wallet).filter(Wallet.balance > 0).all()

    def get_richest_wallets(self, limit: int = 10) -> List[Wallet]:
        """
        Get the wallets with the highest balances.

        Args:
            limit: Maximum number of wallets to return

        Returns:
            List of wallets ordered by balance (descending)
        """
        return (
            self.session.query(Wallet)
            .order_by(desc(Wallet.balance))
            .limit(limit)
            .all()
        )


class PeerNodeRepository(BaseRepository):
    """Repository for PeerNode entities with specialized operations for network management."""

    def __init__(self, session: Optional[Session] = None):
        """Initialize with PeerNode model."""
        super().__init__(session)
        self._model = PeerNode

    def get_by_address(self, address: str) -> Optional[PeerNode]:
        """
        Get a peer node by its network address.

        Args:
            address: The peer's network address (IP:port)

        Returns:
            The peer node if found, None otherwise
        """
        return self.session.query(PeerNode).filter(PeerNode.address == address).first()

    def get_active_peers(self) -> List[PeerNode]:
        """
        Get all active peer nodes.

        Returns:
            List of active peer nodes
        """
        return (
            self.session.query(PeerNode)
            .filter(PeerNode.is_active == True)
            .all()
        )

    def get_peers_by_region(self, region: str) -> List[PeerNode]:
        """
        Get peer nodes by geographical region.

        Args:
            region: The geographical region

        Returns:
            List of peer nodes in the specified region
        """
        return self.session.query(PeerNode).filter(PeerNode.region == region).all()

    def mark_peer_active(self, address: str, is_active: bool = True) -> Optional[PeerNode

