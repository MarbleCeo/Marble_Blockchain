"""
Repository Pattern Implementation for Blockchain Database Access.

This module implements the repository pattern to provide a clean interface
for accessing and manipulating blockchain data in the database. It includes
a base repository with common CRUD operations and specialized repositories
for each entity type with domain-specific query methods.

The repositories handle:
- Data access abstraction
- Transaction management
- Error handling
- Domain-specific query operations

Each repository is responsible for a specific domain entity, providing
a clean separation of concerns and encapsulating the data access logic.
"""

from typing import Generic, TypeVar, List, Optional, Dict, Any, Type, Tuple, Union, cast
from contextlib import contextmanager
from sqlalchemy.orm import Session, Query
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import desc, asc, func, or_, and_
import logging
from datetime import datetime

from .connection import Database
from .models import (
    Base, 
    Block, 
    Transaction, 
    Wallet, 
    PeerNode, 
    TransactionStatus,
    BlockStatus
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variable for generic repository
T = TypeVar('T', bound=Base)


class RepositoryError(Exception):
    """Base exception for repository errors."""
    pass


class EntityNotFoundError(RepositoryError):
    """Raised when an entity is not found in the database."""
    pass


class DuplicateEntityError(RepositoryError):
    """Raised when trying to create a duplicate entity."""
    pass


class BaseRepository(Generic[T]):
    """
    Base repository class implementing common CRUD operations.
    
    This generic class provides common data access methods that can be
    used by all repositories. It handles basic CRUD operations and
    transaction management.
    
    Attributes:
        db (Database): Database connection manager instance
        model_class (Type[T]): The SQLAlchemy model class this repository manages
    """
    
    def __init__(self, db: Database, model_class: Type[T]):
        """
        Initialize the repository with a database connection and model class.
        
        Args:
            db (Database): The database connection manager
            model_class (Type[T]): The SQLAlchemy model class this repository will handle
        """
        self.db = db
        self.model_class = model_class
    
    @contextmanager
    def _transaction(self):
        """
        Context manager for handling database transactions.
        
        Yields:
            Session: An active SQLAlchemy session
            
        Raises:
            RepositoryError: When a database error occurs
        """
        session = self.db.session()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error: {str(e)}")
            raise RepositoryError(f"Database operation failed: {str(e)}") from e
        except Exception as e:
            session.rollback()
            logger.error(f"Unexpected error: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_by_id(self, entity_id: Any) -> Optional[T]:
        """
        Retrieve an entity by its ID.
        
        Args:
            entity_id (Any): The primary key of the entity
            
        Returns:
            Optional[T]: The entity if found, None otherwise
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(self.model_class).get(entity_id)
        except Exception as e:
            logger.error(f"Error retrieving {self.model_class.__name__} with id {entity_id}: {str(e)}")
            raise
    
    def get_all(self) -> List[T]:
        """
        Retrieve all entities of the specified model class.
        
        Returns:
            List[T]: List of all entities
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(self.model_class).all()
        except Exception as e:
            logger.error(f"Error retrieving all {self.model_class.__name__} entities: {str(e)}")
            raise
    
    def create(self, entity: T) -> T:
        """
        Create a new entity in the database.
        
        Args:
            entity (T): The entity to create
            
        Returns:
            T: The created entity with updated ID
            
        Raises:
            RepositoryError: When a database error occurs
            DuplicateEntityError: When the entity already exists
        """
        try:
            with self._transaction() as session:
                session.add(entity)
                session.flush()  # Flush to get the ID without committing
                return entity
        except SQLAlchemyError as e:
            if "unique constraint" in str(e).lower() or "duplicate" in str(e).lower():
                raise DuplicateEntityError(f"Entity already exists: {str(e)}") from e
            raise
    
    def update(self, entity: T) -> T:
        """
        Update an existing entity in the database.
        
        Args:
            entity (T): The entity to update
            
        Returns:
            T: The updated entity
            
        Raises:
            RepositoryError: When a database error occurs
            EntityNotFoundError: When the entity doesn't exist
        """
        try:
            with self._transaction() as session:
                if hasattr(entity, 'id'):
                    existing = session.query(self.model_class).get(entity.id)
                    if not existing:
                        raise EntityNotFoundError(f"{self.model_class.__name__} with id {entity.id} not found")
                
                session.merge(entity)
                return entity
        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error updating {self.model_class.__name__}: {str(e)}")
            raise
    
    def delete(self, entity_id: Any) -> bool:
        """
        Delete an entity by its ID.
        
        Args:
            entity_id (Any): The primary key of the entity to delete
            
        Returns:
            bool: True if the entity was deleted, False otherwise
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                entity = session.query(self.model_class).get(entity_id)
                if not entity:
                    return False
                
                session.delete(entity)
                return True
        except Exception as e:
            logger.error(f"Error deleting {self.model_class.__name__} with id {entity_id}: {str(e)}")
            raise
    
    def count(self) -> int:
        """
        Count the number of entities in the database.
        
        Returns:
            int: The number of entities
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(func.count(self.model_class.id)).scalar()
        except Exception as e:
            logger.error(f"Error counting {self.model_class.__name__} entities: {str(e)}")
            raise


class BlockRepository(BaseRepository[Block]):
    """
    Repository for managing Block entities with specialized blockchain operations.
    
    This repository extends the base repository with blockchain-specific
    query methods for blocks, such as retrieving blocks by height, hash,
    or getting the latest blocks in the chain.
    """
    
    def __init__(self, db: Database):
        """
        Initialize the Block repository.
        
        Args:
            db (Database): The database connection manager
        """
        super().__init__(db, Block)
    
    def get_by_hash(self, block_hash: str) -> Optional[Block]:
        """
        Retrieve a block by its hash.
        
        Args:
            block_hash (str): The hash of the block to retrieve
            
        Returns:
            Optional[Block]: The block if found, None otherwise
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(Block).filter(Block.hash == block_hash).first()
        except Exception as e:
            logger.error(f"Error retrieving block with hash {block_hash}: {str(e)}")
            raise
    
    def get_by_height(self, height: int) -> Optional[Block]:
        """
        Retrieve a block by its height in the blockchain.
        
        Args:
            height (int): The height of the block to retrieve
            
        Returns:
            Optional[Block]: The block if found, None otherwise
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(Block).filter(Block.height == height).first()
        except Exception as e:
            logger.error(f"Error retrieving block at height {height}: {str(e)}")
            raise
    
    def get_latest_block(self) -> Optional[Block]:
        """
        Retrieve the latest block in the blockchain.
        
        Returns:
            Optional[Block]: The latest block if any blocks exist, None otherwise
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(Block).order_by(desc(Block.height)).first()
        except Exception as e:
            logger.error(f"Error retrieving latest block: {str(e)}")
            raise
    
    def get_block_range(self, start_height: int, end_height: int) -> List[Block]:
        """
        Retrieve a range of blocks by their heights.
        
        Args:
            start_height (int): The starting height (inclusive)
            end_height (int): The ending height (inclusive)
            
        Returns:
            List[Block]: List of blocks in the specified range
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(Block).filter(
                    Block.height >= start_height,
                    Block.height <= end_height
                ).order_by(asc(Block.height)).all()
        except Exception as e:
            logger.error(f"Error retrieving blocks from height {start_height} to {end_height}: {str(e)}")
            raise
    
    def get_blocks_by_timestamp_range(self, start_time: datetime, end_time: datetime) -> List[Block]:
        """
        Retrieve blocks created within a specified time range.
        
        Args:
            start_time (datetime): The start timestamp (inclusive)
            end_time (datetime): The end timestamp (inclusive)
            
        Returns:
            List[Block]: List of blocks in the specified time range
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(Block).filter(
                    Block.timestamp >= start_time,
                    Block.timestamp <= end_time
                ).order_by(asc(Block.timestamp)).all()
        except Exception as e:
            logger.error(f"Error retrieving blocks in time range: {str(e)}")
            raise
    
    def get_latest_blocks(self, limit: int = 10) -> List[Block]:
        """
        Retrieve the latest N blocks in the blockchain.
        
        Args:
            limit (int, optional): Maximum number of blocks to retrieve. Defaults to 10.
            
        Returns:
            List[Block]: List of the latest blocks
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(Block).order_by(desc(Block.height)).limit(limit).all()
        except Exception as e:
            logger.error(f"Error retrieving latest {limit} blocks: {str(e)}")
            raise
    
    def get_blockchain_height(self) -> int:
        """
        Get the current height of the blockchain.
        
        Returns:
            int: The height of the latest block, or 0 if no blocks exist
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                latest = session.query(func.max(Block.height)).scalar()
                return latest if latest is not None else 0
        except Exception as e:
            logger.error(f"Error retrieving blockchain height: {str(e)}")
            raise
    
    def get_blocks_by_validator(self, validator_id: str, limit: int = 100) -> List[Block]:
        """
        Retrieve blocks created by a specific validator.
        
        Args:
            validator_id (str): The ID of the validator
            limit (int, optional): Maximum number of blocks to retrieve. Defaults to 100.
            
        Returns:
            List[Block]: List of blocks created by the validator
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(Block).filter(
                    Block.validator_id == validator_id
                ).order_by(desc(Block.height)).limit(limit).all()
        except Exception as e:
            logger.error(f"Error retrieving blocks for validator {validator_id}: {str(e)}")
            raise


class TransactionRepository(BaseRepository[Transaction]):
    """
    Repository for managing Transaction entities with specialized query operations.
    
    This repository extends the base repository with transaction-specific
    query methods, such as retrieving transactions by status, sender, recipient,
    or finding transactions in a specific block.
    """
    
    def __init__(self, db: Database):
        """
        Initialize the Transaction repository.
        
        Args:
            db (Database): The database connection manager
        """
        super().__init__(db, Transaction)
    
    def get_by_hash(self, tx_hash: str) -> Optional[Transaction]:
        """
        Retrieve a transaction by its hash.
        
        Args:
            tx_hash (str): The hash of the transaction
            
        Returns:
            Optional[Transaction]: The transaction if found, None otherwise
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(Transaction).filter(Transaction.hash == tx_hash).first()
        except Exception as e:
            logger.error(f"Error retrieving transaction with hash {tx_hash}: {str(e)}")
            raise
    
    def get_by_block_id(self, block_id: str) -> List[Transaction]:
        """
        Retrieve all transactions in a specific block.
        
        Args:
            block_id (str): The ID of the block
            
        Returns:
            List[Transaction]: List of transactions in the block
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(Transaction).filter(Transaction.block_id == block_id).all()
        except Exception as e:
            logger.error(f"Error retrieving transactions for block {block_id}: {str(e)}")
            raise
    
    def get_by_sender(self, sender_address: str, limit: int = 100) -> List[Transaction]:
        """
        Retrieve transactions sent from a specific address.
        
        Args:
            sender_address (str): The address of the sender
            limit (int, optional): Maximum number of transactions to retrieve. Defaults to 100.
            
        Returns:
            List[Transaction]: List of transactions sent from the address
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(Transaction).filter(
                    Transaction.sender_address == sender_address
                ).order_by(desc(Transaction.timestamp)).limit(limit).all()
        except Exception as e:
            logger.error(f"Error retrieving transactions from sender {sender_address}: {str(e)}")
            raise
    
    def get_by_recipient(self, recipient_address: str, limit: int = 100) -> List[Transaction]:
        """
        Retrieve transactions sent to a specific address.
        
        Args:
            recipient_address (str): The address of the recipient
            limit (int, optional): Maximum number of transactions to retrieve. Defaults to 100.
            
        Returns:
            List[Transaction]: List of transactions sent to the address
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(Transaction).filter(
                    Transaction.recipient_address == recipient_address
                ).order_by(desc(Transaction.timestamp)).limit(limit).all()
        except Exception as e:
            logger.error(f"Error retrieving transactions to recipient {recipient_address}: {str(e)}")
            raise
    
    def get_by_address(self, address: str, limit: int = 100) -> List[Transaction]:
        """
        Retrieve transactions associated with a specific address (as sender or recipient).
        
        Args:
            address (str): The address to query
            limit (int, optional): Maximum number of transactions to retrieve. Defaults to 100.
            
        Returns:
            List[Transaction]: List of transactions involving the address
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(Transaction).filter(
                    or_(
                        Transaction.sender_address == address,
                        Transaction.recipient_address == address
                    )
                ).order_by(desc(Transaction.timestamp)).limit(limit).all()
        except Exception as e:
            logger.error(f"Error retrieving transactions for address {address}: {str(e)}")
            raise
    
    def get_by_status(self, status: TransactionStatus, limit: int = 100) -> List[Transaction]:
        """
        Retrieve transactions with a specific status.
        
        Args:
            status (TransactionStatus): The status to query
            limit (int, optional): Maximum number of transactions to retrieve. Defaults to 100.
            
        Returns:
            List[Transaction]: List of transactions with the specified status
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(Transaction).filter(
                    Transaction.status == status
                ).order_by(desc(Transaction.timestamp)).limit(limit).all()
        except Exception as e:
            logger.error(f"Error retrieving transactions with status {status}: {str(e)}")
            raise
    
    def get_by_time_range(self, start_time: datetime, end_time: datetime, limit: int = 100) -> List[Transaction]:
        """
        Retrieve transactions within a specific time range.
        
        Args:
            start_time (datetime): The start timestamp (inclusive)
            end_time (datetime): The end timestamp (inclusive)
            limit (int, optional): Maximum number of transactions to retrieve. Defaults to 100.
            
        Returns:
            List[Transaction]: List of transactions in the specified time range
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(Transaction).filter(
                    Transaction.timestamp >= start_time,
                    Transaction.timestamp <= end_time
                ).order_by(desc(Transaction.timestamp)).limit(limit).all()
        except Exception as e:
            logger.error(f"Error retrieving transactions in time range: {str(e)}")
            raise
    
    def get_pending_transactions(self, limit: int = 100) -> List[Transaction]:
        """
        Retrieve pending transactions that have not been included in a block yet.
        
        Args:
            limit (int, optional): Maximum number of transactions to retrieve. Defaults to 100.
            
        Returns:
            List[Transaction]: List of pending transactions
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(Transaction).filter(
                    Transaction.status == TransactionStatus.PENDING,
                    Transaction.block_id.is_(None)
                ).order_by(asc(Transaction.timestamp)).limit(limit).all()
        except Exception as e:
            logger.error(f"Error retrieving pending transactions: {str(e)}")
            raise
    
    def get_transaction_count_by_address(self, address: str) -> int:
        """
        Count the number of transactions associated with an address.
        
        Args:
            address (str): The address to query
            
        Returns:
            int: The total number of transactions
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(func.count(Transaction.id)).filter(
                    or_(
                        Transaction.sender_address == address,
                        Transaction.recipient_address == address
                    )
                ).scalar()
        except Exception as e:
            logger.error(f"Error counting transactions for address {address}: {str(e)}")
            raise


class WalletRepository(BaseRepository[Wallet]):
    """
    Repository for managing Wallet entities.
    
    This repository extends the base repository with wallet-specific
    operations, such as finding wallets by their address or managing
    wallet balances and transaction history.
    """
    
    def __init__(self, db: Database):
        """
        Initialize the Wallet repository.
        
        Args:
            db (Database): The database connection manager
        """
        super().__init__(db, Wallet)
    
    def get_by_address(self, address: str) -> Optional[Wallet]:
        """
        Retrieve a wallet by its public address.
        
        Args:
            address (str): The public address of the wallet
            
        Returns:
            Optional[Wallet]: The wallet if found, None otherwise
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(Wallet).filter(Wallet.address == address).first()
        except Exception as e:
            logger.error(f"Error retrieving wallet with address {address}: {str(e)}")
            raise
    
    def get_by_public_key(self, public_key: str) -> Optional[Wallet]:
        """
        Retrieve a wallet by its public key.
        
        Args:
            public_key (str): The public key of the wallet
            
        Returns:
            Optional[Wallet]: The wallet if found, None otherwise
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(Wallet).filter(Wallet.public_key == public_key).first()
        except Exception as e:
            logger.error(f"Error retrieving wallet with public key {public_key}: {str(e)}")
            raise
    
    def update_balance(self, address: str, new_balance: float) -> Optional[Wallet]:
        """
        Update the balance of a wallet.
        
        Args:
            address (str): The address of the wallet
            new_balance (float): The new balance
            
        Returns:
            Optional[Wallet]: The updated wallet if found, None otherwise
            
        Raises:
            RepositoryError: When a database error occurs
            EntityNotFoundError: When the wallet doesn't exist
        """
        try:
            with self._transaction() as session:
                wallet = session.query(Wallet).filter(Wallet.address == address).first()
                if not wallet:
                    raise EntityNotFoundError(f"Wallet with address {address} not found")
                
                wallet.balance = new_balance
                wallet.last_updated = datetime.now()
                return wallet
        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error updating balance for wallet {address}: {str(e)}")
            raise
    
    def get_active_wallets(self, days: int = 30, limit: int = 100) -> List[Wallet]:
        """
        Retrieve wallets that have been active in the specified number of days.
        
        Args:
            days (int, optional): Number of days to consider. Defaults to 30.
            limit (int, optional): Maximum number of wallets to retrieve. Defaults to 100.
            
        Returns:
            List[Wallet]: List of active wallets
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                cutoff_date = datetime.now() - datetime.timedelta(days=days)
                return session.query(Wallet).filter(
                    Wallet.last_updated >= cutoff_date
                ).order_by(desc(Wallet.last_updated)).limit(limit).all()
        except Exception as e:
            logger.error(f"Error retrieving active wallets in the last {days} days: {str(e)}")
            raise
    
    def get_top_wallets_by_balance(self, limit: int = 100) -> List[Wallet]:
        """
        Retrieve wallets with the highest balances.
        
        Args:
            limit (int, optional): Maximum number of wallets to retrieve. Defaults to 100.
            
        Returns:
            List[Wallet]: List of wallets with highest balances
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(Wallet).order_by(desc(Wallet.balance)).limit(limit).all()
        except Exception as e:
            logger.error(f"Error retrieving top wallets by balance: {str(e)}")
            raise


class PeerNodeRepository(BaseRepository[PeerNode]):
    """
    Repository for managing PeerNode entities.
    
    This repository extends the base repository with peer node-specific
    operations, such as finding active peers, managing peer reputation,
    and tracking peer status.
    """
    
    def __init__(self, db: Database):
        """
        Initialize the PeerNode repository.
        
        Args:
            db (Database): The database connection manager
        """
        super().__init__(db, PeerNode)
    
    def get_by_address(self, address: str) -> Optional[PeerNode]:
        """
        Retrieve a peer node by its network address.
        
        Args:
            address (str): The network address of the peer node
            
        Returns:
            Optional[PeerNode]: The peer node if found, None otherwise
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(PeerNode).filter(PeerNode.address == address).first()
        except Exception as e:
            logger.error(f"Error retrieving peer node with address {address}: {str(e)}")
            raise
    
    def get_active_peers(self, limit: int = 100) -> List[PeerNode]:
        """
        Retrieve active peer nodes.
        
        Args:
            limit (int, optional): Maximum number of peer nodes to retrieve. Defaults to 100.
            
        Returns:
            List[PeerNode]: List of active peer nodes
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(PeerNode).filter(
                    PeerNode.is_active == True
                ).order_by(desc(PeerNode.last_seen)).limit(limit).all()
        except Exception as e:
            logger.error(f"Error retrieving active peer nodes: {str(e)}")
            raise
    
    def mark_peer_active(self, address: str) -> Optional[PeerNode]:
        """
        Mark a peer node as active and update its last seen timestamp.
        
        Args:
            address (str): The network address of the peer node
            
        Returns:
            Optional[PeerNode]: The updated peer node if found, None otherwise
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                peer = session.query(PeerNode).filter(PeerNode.address == address).first()
                if not peer:
                    return None
                
                peer.is_active = True
                peer.last_seen = datetime.now()
                peer.connection_attempts += 1
                return peer
        except Exception as e:
            logger.error(f"Error marking peer {address} as active: {str(e)}")
            raise
    
    def mark_peer_inactive(self, address: str) -> Optional[PeerNode]:
        """
        Mark a peer node as inactive.
        
        Args:
            address (str): The network address of the peer node
            
        Returns:
            Optional[PeerNode]: The updated peer node if found, None otherwise
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                peer = session.query(PeerNode).filter(PeerNode.address == address).first()
                if not peer:
                    return None
                
                peer.is_active = False
                return peer
        except Exception as e:
            logger.error(f"Error marking peer {address} as inactive: {str(e)}")
            raise
    
    def update_peer_reputation(self, address: str, reputation_change: int) -> Optional[PeerNode]:
        """
        Update the reputation score of a peer node.
        
        Args:
            address (str): The network address of the peer node
            reputation_change (int): The amount to change the reputation by (positive or negative)
            
        Returns:
            Optional[PeerNode]: The updated peer node if found, None otherwise
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                peer = session.query(PeerNode).filter(PeerNode.address == address).first()
                if not peer:
                    return None
                
                peer.reputation += reputation_change
                peer.last_updated = datetime.now()
                return peer
        except Exception as e:
            logger.error(f"Error updating reputation for peer {address}: {str(e)}")
            raise
    
    def record_peer_interaction(self, address: str, success: bool, latency_ms: Optional[int] = None) -> Optional[PeerNode]:
        """
        Record a peer interaction, updating statistics and reputation.
        
        Args:
            address (str): The network address of the peer node
            success (bool): Whether the interaction was successful
            latency_ms (Optional[int], optional): The latency in milliseconds, if measured
            
        Returns:
            Optional[PeerNode]: The updated peer node if found, None otherwise
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                peer = session.query(PeerNode).filter(PeerNode.address == address).first()
                if not peer:
                    return None
                
                # Update statistics
                peer.total_interactions += 1
                if success:
                    peer.successful_interactions += 1
                    # Small reputation increase for successful interaction
                    peer.reputation += 1
                else:
                    peer.failed_interactions += 1
                    # Small reputation decrease for failed interaction
                    peer.reputation -= 1
                
                # Update latency statistics if provided
                if latency_ms is not None:
                    # If it's the first interaction or current avg_latency is None
                    if peer.avg_latency is None:
                        peer.avg_latency = latency_ms
                    else:
                        # Rolling average
                        peer.avg_latency = (peer.avg_latency * (peer.total_interactions - 1) + latency_ms) / peer.total_interactions
                
                peer.last_interaction = datetime.now()
                peer.last_seen = datetime.now()
                peer.is_active = True
                return peer
        except Exception as e:
            logger.error(f"Error recording interaction for peer {address}: {str(e)}")
            raise
    
    def record_data_transfer(self, address: str, bytes_sent: int, bytes_received: int) -> Optional[PeerNode]:
        """
        Record data transfer statistics for a peer node.
        
        Args:
            address (str): The network address of the peer node
            bytes_sent (int): Number of bytes sent to the peer
            bytes_received (int): Number of bytes received from the peer
            
        Returns:
            Optional[PeerNode]: The updated peer node if found, None otherwise
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                peer = session.query(PeerNode).filter(PeerNode.address == address).first()
                if not peer:
                    return None
                
                peer.bytes_sent += bytes_sent
                peer.bytes_received += bytes_received
                peer.last_data_transfer = datetime.now()
                return peer
        except Exception as e:
            logger.error(f"Error recording data transfer for peer {address}: {str(e)}")
            raise
    
    def get_peers_by_reputation(self, min_reputation: int = 0, limit: int = 100) -> List[PeerNode]:
        """
        Retrieve peer nodes with a minimum reputation score.
        
        Args:
            min_reputation (int, optional): Minimum reputation score required. Defaults to 0.
            limit (int, optional): Maximum number of peer nodes to retrieve. Defaults to 100.
            
        Returns:
            List[PeerNode]: List of peer nodes with reputation >= min_reputation
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                return session.query(PeerNode).filter(
                    PeerNode.reputation >= min_reputation
                ).order_by(desc(PeerNode.reputation)).limit(limit).all()
        except Exception as e:
            logger.error(f"Error retrieving peers by reputation: {str(e)}")
            raise
    
    def get_peers_by_last_seen(self, hours: int = 24, limit: int = 100) -> List[PeerNode]:
        """
        Retrieve peer nodes that have been seen within the specified number of hours.
        
        Args:
            hours (int, optional): Number of hours to consider. Defaults to 24.
            limit (int, optional): Maximum number of peer nodes to retrieve. Defaults to 100.
            
        Returns:
            List[PeerNode]: List of recently seen peer nodes
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                cutoff_time = datetime.now() - datetime.timedelta(hours=hours)
                return session.query(PeerNode).filter(
                    PeerNode.last_seen >= cutoff_time
                ).order_by(desc(PeerNode.last_seen)).limit(limit).all()
        except Exception as e:
            logger.error(f"Error retrieving peers by last seen timestamp: {str(e)}")
            raise
    
    def get_most_reliable_peers(self, min_success_rate: float = 0.8, limit: int = 20) -> List[PeerNode]:
        """
        Retrieve the most reliable peer nodes based on successful interaction rate.
        
        Args:
            min_success_rate (float, optional): Minimum success rate required (0.0-1.0). Defaults to 0.8.
            limit (int, optional): Maximum number of peer nodes to retrieve. Defaults to 20.
            
        Returns:
            List[PeerNode]: List of reliable peer nodes
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                # Find peers with at least some interactions and calculate success rate
                return session.query(PeerNode).filter(
                    PeerNode.total_interactions > 0,
                    PeerNode.successful_interactions / PeerNode.total_interactions >= min_success_rate
                ).order_by(
                    desc(PeerNode.successful_interactions / PeerNode.total_interactions),
                    desc(PeerNode.total_interactions)
                ).limit(limit).all()
        except Exception as e:
            logger.error(f"Error retrieving most reliable peers: {str(e)}")
            raise
    
    def prune_inactive_peers(self, days: int = 30) -> int:
        """
        Remove peer nodes that have been inactive for the specified number of days.
        
        Args:
            days (int, optional): Number of days of inactivity required for pruning. Defaults to 30.
            
        Returns:
            int: Number of peer nodes pruned
            
        Raises:
            RepositoryError: When a database error occurs
        """
        try:
            with self._transaction() as session:
                cutoff_time = datetime.now() - datetime.timedelta(days=days)
                inactive_peers = session.query(PeerNode).filter(
                    PeerNode.last_seen < cutoff_time
                ).all()
                
                count = len(inactive_peers)
                for peer in inactive_peers:
                    session.delete(peer)
                
                return count
        except Exception as e:
            logger.error(f"Error pruning inactive peers: {str(e)}")
            raise
