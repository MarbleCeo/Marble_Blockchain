import unittest
import sys
import os
import json
from datetime import datetime

# Add the src directory to the path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from blockchain.blockchain_merged import Block, Transaction, Blockchain

class TestBlockchain(unittest.TestCase):
    """Test cases for the blockchain implementation."""
    
    def setUp(self):
        """Set up a fresh blockchain instance for each test."""
        # Use in-memory SQLite for testing
        self.blockchain = Blockchain(db_file=":memory:")
        
    def test_genesis_block(self):
        """Test that a new blockchain has a genesis block."""
        self.assertEqual(len(self.blockchain.chain), 1)
        self.assertEqual(self.blockchain.chain[0].index, 0)
        self.assertEqual(self.blockchain.chain[0].previous_hash, "0")
        
    def test_add_transaction(self):
        """Test adding a transaction to the blockchain."""
        tx = Transaction(sender="alice", recipient="bob", amount=10.0)
        result = self.blockchain.add_transaction(tx)
        self.assertTrue(result)
        self.assertEqual(len(self.blockchain.pending_transactions), 1)
        
    def test_mine_block(self):
        """Test mining a new block with pending transactions."""
        # Add a transaction
        tx = Transaction(sender="alice", recipient="bob", amount=5.0)
        self.blockchain.add_transaction(tx)
        
        # Mine a block
        miner_address = "miner1"
        block = self.blockchain.mine_pending_transactions(miner_address)
        
        # Verify block was mined and added
        self.assertEqual(len(self.blockchain.chain), 2)
        self.assertEqual(block.index, 1)
        self.assertEqual(block.previous_hash, self.blockchain.chain[0].hash)
        
        # Verify mining reward transaction was added
        # The miner should have a reward in the pending transactions
        self.assertEqual(len(self.blockchain.pending_transactions), 1)
        self.assertEqual(self.blockchain.pending_transactions[0].recipient, miner_address)
        
    def test_blockchain_validity(self):
        """Test blockchain validation."""
        # Add and mine some blocks
        self.blockchain.add_transaction(Transaction("alice", "bob", 10.0))
        self.blockchain.mine_pending_transactions("miner1")
        
        # Blockchain should be valid
        self.assertTrue(self.blockchain.is_chain_valid())
        
        # Tamper with a block
        self.blockchain.chain[1].transactions[0].amount = 100.0
        
        # Blockchain should now be invalid
        self.assertFalse(self.blockchain.is_chain_valid())
        
    def test_get_balance(self):
        """Test getting account balance."""
        # Add transactions and mine
        self.blockchain.add_transaction(Transaction("genesis", "alice", 50.0))
        self.blockchain.mine_pending_transactions("miner1")
        
        # Mine another block to add the mining reward
        self.blockchain.mine_pending_transactions("miner1")
        
        # Check balances
        self.assertEqual(self.blockchain.get_balance("alice"), 50.0)
        self.assertGreater(self.blockchain.get_balance("miner1"), 0)
        
if __name__ == '__main__':
    unittest.main()

