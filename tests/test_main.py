import pytest
from datetime import datetime
from hashlib import sha256
import json
from typing import Dict, List

# Assuming these are the main components in your blockchain
from blockchain import (
    Transaction,
    Block,
    Blockchain,
    create_genesis_block,
    validate_transaction,
    calculate_hash,
    proof_of_work,
    adjust_difficulty
)


@pytest.fixture
def sample_transaction() -> Transaction:
    """Fixture for a valid transaction."""
    return Transaction(
        sender="Alice",
        recipient="Bob",
        amount=10,
        timestamp=datetime.now().timestamp()
    )


@pytest.fixture
def genesis_block() -> Block:
    """Fixture for genesis block."""
    return create_genesis_block()


@pytest.fixture
def blockchain(genesis_block) -> Blockchain:
    """Fixture for blockchain starting with genesis block."""
    return Blockchain([genesis_block])


class TestTransactionValidation:
    """Test suite for transaction validation."""

    def test_valid_transaction(self, sample_transaction):
        """Test that valid transactions pass validation."""
        assert validate_transaction(sample_transaction) is True

    def test_transaction_negative_amount(self):
        """Test that transactions with negative amounts fail."""
        invalid_tx = Transaction(
            sender="Alice",
            recipient="Bob",
            amount=-10,
            timestamp=datetime.now().timestamp()
        )
        assert validate_transaction(invalid_tx) is False

    def test_transaction_missing_sender(self):
        """Test that transactions without sender fail."""
        invalid_tx = Transaction(
            sender="",
            recipient="Bob",
            amount=10,
            timestamp=datetime.now().timestamp()
        )
        assert validate_transaction(invalid_tx) is False

    def test_transaction_invalid_timestamp(self):
        """Test that transactions with future timestamps fail."""
        future_time = datetime.now().timestamp() + 3600  # 1 hour in future
        invalid_tx = Transaction(
            sender="Alice",
            recipient="Bob",
            amount=10,
            timestamp=future_time
        )
        assert validate_transaction(invalid_tx) is False


class TestBlockOperations:
    """Test suite for block creation and validation."""

    def test_block_creation(self, sample_transaction, genesis_block):
        """Test that blocks are created correctly with transactions."""
        transactions = [sample_transaction]
        new_block = Block(
            index=1,
            timestamp=datetime.now().timestamp(),
            transactions=transactions,
            previous_hash=genesis_block.hash,
            nonce=0,
            hash=""
        )
        new_block.hash = calculate_hash(new_block)
        assert new_block.is_valid(genesis_block) is True

    def test_block_tampering(self, sample_transaction, genesis_block):
        """Test that tampered blocks fail validation."""
        transactions = [sample_transaction]
        new_block = Block(
            index=1,
            timestamp=datetime.now().timestamp(),
            transactions=transactions,
            previous_hash=genesis_block.hash,
            nonce=0,
            hash=""
        )
        new_block.hash = calculate_hash(new_block)
        
        # Tamper with the block
        new_block.transactions[0].amount = 1000
        assert new_block.is_valid(genesis_block) is False

    def test_block_invalid_previous_hash(self, sample_transaction, genesis_block):
        """Test that blocks with invalid previous hashes fail."""
        transactions = [sample_transaction]
        new_block = Block(
            index=1,
            timestamp=datetime.now().timestamp(),
            transactions=transactions,
            previous_hash="invalid_previous_hash",
            nonce=0,
            hash=""
        )
        new_block.hash = calculate_hash(new_block)
        assert new_block.is_valid(genesis_block) is False


class TestProofOfWork:
    """Test suite for proof-of-work and difficulty adjustment."""

    def test_pow_calculation(self, sample_transaction, genesis_block):
        """Test that proof-of-work produces valid hashes."""
        transactions = [sample_transaction]
        block = Block(
            index=1,
            timestamp=datetime.now().timestamp(),
            transactions=transactions,
            previous_hash=genesis_block.hash,
            nonce=0,
            hash=""
        )
        proof = proof_of_work(block, difficulty=4)
        assert proof.hash.startswith('0000')

    def test_difficulty_adjustment_slow(self):
        """Test difficulty increases when blocks are mined too fast."""
        block_times = [5, 5, 5]  # Fast block times (5 seconds)
        current_difficulty = 4
        new_difficulty = adjust_difficulty(current_difficulty, block_times)
        assert new_difficulty > current_difficulty

    def test_difficulty_adjustment_fast(self):
        """Test difficulty decreases when blocks are mined too slowly."""
        block_times = [45, 45, 45]  # Slow block times (45 seconds)
        current_difficulty = 4
        new_difficulty = adjust_difficulty(current_difficulty, block_times)
        assert new_difficulty < current_difficulty


class TestBlockchainSynchronization:
    """Test suite for blockchain synchronization scenarios."""

    def test_blockchain_fork_resolution(self, genesis_block):
        """Test that the longer chain is chosen during forks."""
        # Create two competing chains
        blockchain1 = Blockchain([genesis_block])
        blockchain2 = Blockchain([genesis_block])
        
        # Add blocks to each chain
        for i in range(1, 3):
            block = create_mock_block(blockchain1.chain[-1])
            blockchain1.add_block(block)
        
        for i in range(1, 4):
            block = create_mock_block(blockchain2.chain[-1])
            blockchain2.add_block(block)
        
        # Simulate receiving both chains
        blockchain1.sync(blockchain2.chain)
        assert len(blockchain1.chain) == 4  # Chose the longer chain

    def test_blockchain_reorg(self, blockchain):
        """Test that blockchain handles reorganizations correctly."""
        main_chain = blockchain
        fork_block = main_chain.chain[-1]  # Genesis block
        
        # Create a fork that's longer than main chain
        fork_chain = [fork_block]
        for i in range(1, 4):
            block = create_mock_block(fork_chain[-1])
            fork_chain.append(block)
        
        # Sync should replace main chain with longer fork
        main_chain.sync(fork_chain)
        assert len(main_chain.chain) == 4

    def test_invalid_chain_rejection(self, blockchain):
        """Test that invalid chains are rejected during sync."""
        invalid_chain = blockchain.chain.copy()
        invalid_chain[0].transactions = []  # Tamper with genesis block
        
        with pytest.raises(ValueError):
            blockchain.sync(invalid_chain)


# Helper functions
def create_mock_block(previous_block: Block) -> Block:
    """Helper to create mock blocks for testing."""
    mock_tx = Transaction(
        sender="Test",
        recipient="Test",
        amount=1,
        timestamp=datetime.now().timestamp()
    )
    block = Block(
        index=previous_block.index + 1,
        timestamp=datetime.now().timestamp(),
        transactions=[mock_tx],
        previous_hash=previous_block.hash,
        nonce=0,
        hash=""
    )
    block.hash = calculate_hash(block)
    return block

