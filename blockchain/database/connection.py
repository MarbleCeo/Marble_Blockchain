"""
Database Connection Management Module

This module provides a robust database connection manager for SQLite or other SQL databases
using SQLAlchemy. It includes connection pooling, transaction management, and error handling
to ensure reliable database operations in the blockchain system.

Features:
- Connection pooling for efficient resource usage
- Transaction management with context managers
- Configurable connection settings
- Comprehensive error handling
- Thread-safe operations
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional, Union

from sqlalchemy import create_engine, event, exc
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConnectionError(Exception):
    """Exception raised for database connection errors."""
    pass


class TransactionError(Exception):
    """Exception raised for transaction-related errors."""
    pass


class DatabaseConnection:
    """
    Database connection manager that handles connection pooling and transaction management.

    This class provides methods to create database engines, manage connection pools,
    and handle database sessions and transactions in a thread-safe manner.

    Attributes:
        engine (Engine): SQLAlchemy engine instance
        session_factory (sessionmaker): Factory for creating new database sessions
        db_url (str): Database connection URL
        _pool_size (int): Maximum number of connections to keep in the pool
        _max_overflow (int): Maximum number of connections to create beyond pool_size
        _pool_timeout (int): Number of seconds to wait before timing out on getting a connection
        _pool_recycle (int): Number of seconds to wait before recycling connections
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        db_path: Optional[str] = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        echo: bool = False,
    ) -> None:
        """
        Initialize the database connection manager.

        Args:
            db_url: SQLAlchemy database URL (e.g., 'sqlite:///blockchain.db')
            db_path: Path to SQLite database file (alternative to db_url)
            pool_size: Maximum number of connections to keep in the pool
            max_overflow: Maximum number of connections to create beyond pool_size
            pool_timeout: Number of seconds to wait before timing out on getting a connection
            pool_recycle: Number of seconds to wait before recycling connections
            echo: If True, the engine will log all statements

        Raises:
            DatabaseConnectionError: If there's an error connecting to the database
        """
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool_timeout = pool_timeout
        self._pool_recycle = pool_recycle

        # Determine the database URL
        if db_url:
            self.db_url = db_url
        elif db_path:
            # Create directory for the database file if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
            self.db_url = f"sqlite:///{db_path}"
        else:
            self.db_url = "sqlite:///blockchain.db"  # Default database

        try:
            # Create engine with connection pooling
            self.engine = create_engine(
                self.db_url,
                poolclass=QueuePool,
                pool_size=self._pool_size,
                max_overflow=self._max_overflow,
                pool_timeout=self._pool_timeout,
                pool_recycle=self._pool_recycle,
                echo=echo,
            )

            # Configure SQLite to handle foreign key constraints
            if self.db_url.startswith('sqlite'):
                @event.listens_for(self.engine, "connect")
                def set_sqlite_pragma(dbapi_connection, connection_record):
                    cursor = dbapi_connection.cursor()
                    cursor.execute("PRAGMA foreign_keys=ON")
                    cursor.close()

            # Create a session factory
            self.session_factory = sessionmaker(bind=self.engine)
            
            # Test the connection
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
                
            logger.info(f"Successfully connected to database: {self.db_url}")
        except exc.SQLAlchemyError as e:
            logger.error(f"Failed to connect to database: {e}")
            raise DatabaseConnectionError(f"Database connection failed: {e}") from e

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Provide a transactional scope around a series of operations.
        
        This context manager creates a new session, handles commits and rollbacks,
        and properly closes the session when done.

        Yields:
            Session: SQLAlchemy session object

        Raises:
            TransactionError: If there's an error during the transaction
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Transaction failed, rolling back: {e}")
            raise TransactionError(f"Transaction failed: {e}") from e
        finally:
            session.close()

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute a raw SQL query with parameters.

        Args:
            query: SQL query string
            params: Parameters for the SQL query

        Returns:
            The result of the query execution

        Raises:
            DatabaseConnectionError: If there's an error executing the query
        """
        try:
            with self.engine.connect() as conn:
                if params:
                    result = conn.execute(query, params)
                else:
                    result = conn.execute(query)
                return result
        except exc.SQLAlchemyError as e:
            logger.error(f"Query execution failed: {e}")
            raise DatabaseConnectionError(f"Query execution failed: {e}") from e

    def get_connection_status(self) -> Dict[str, Union[str, int]]:
        """
        Get the current status of the database connection pool.

        Returns:
            Dictionary containing connection pool statistics
        """
        return {
            "db_url": self.db_url,
            "pool_size": self._pool_size,
            "max_overflow": self._max_overflow,
            "pool_timeout": self._pool_timeout,
            "pool_recycle": self._pool_recycle,
            "connections_in_use": self.engine.pool.checkedin(),
            "connections_available": self.engine.pool.checkedout(),
        }

    def close(self) -> None:
        """
        Close the database connection pool.

        This method should be called when the application is shutting down.
        """
        if hasattr(self, 'engine') and self.engine:
            self.engine.dispose()
            logger.info("Database connection pool closed")


# Example usage
if __name__ == "__main__":
    # Example of how to use the DatabaseConnection class
    db = DatabaseConnection(db_path="example.db")
    
    # Using the session context manager for a transaction
    with db.session_scope() as session:
        # Example query (replace with actual model queries)
        result = session.execute("SELECT 1").fetchone()
        print(f"Connection test result: {result}")
        
    # Close connections when done
    db.close()

