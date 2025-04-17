import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time

# Import the modules to be tested
from micro_os.ai.transaction_optimizer import (
    TransactionOptimizer,
    TransactionPool,
    OptimizationStrategy
)
from micro_os.ai.mining_optimizer import (
    MiningOptimizer,
    NeuralHashingModel,
    MiningPerformanceMonitor
)
from micro_os.ai.network_optimizer import (
    NetworkOptimizer,
    PeerDiscovery,
    RoutingTable,
    NetworkCondition
)


# ===== Fixtures =====

@pytest.fixture
def mock_blockchain():
    """Mock blockchain environment for testing."""
    mock = Mock()
    mock.get_difficulty = Mock(return_value=0.0001)
    mock.get_mempool_size = Mock(return_value=100)
    mock.get_current_block_height = Mock(return_value=1000)
    return mock


@pytest.fixture
def transaction_pool():
    """Create a transaction pool with sample transactions."""
    pool = TransactionPool()
    # Create 10 mock transactions with varying properties
    for i in range(10):
        tx = Mock()
        tx.fee = 0.001 * (i + 1)
        tx.size = 1000 * (i % 3 + 1)
        tx.timestamp = time.time() - (i * 60)  # Spread over last several minutes
        tx.hash = f"0x{i:064x}"
        tx.sender = f"0x{i+100:040x}"
        tx.recipient = f"0x{i+200:040x}"
        tx.validate = Mock(return_value=True)
        # Every third transaction fails validation
        if i % 3 == 0:
            tx.validate = Mock(return_value=False)
        pool.add_transaction(tx)
    return pool


@pytest.fixture
def transaction_optimizer(transaction_pool, mock_blockchain):
    """Create a transaction optimizer instance."""
    return TransactionOptimizer(
        transaction_pool=transaction_pool,
        blockchain=mock_blockchain,
        strategy=OptimizationStrategy.BALANCED
    )


@pytest.fixture
def mining_optimizer(mock_blockchain):
    """Create a mining optimizer instance."""
    model = NeuralHashingModel()
    monitor = MiningPerformanceMonitor()
    return MiningOptimizer(
        blockchain=mock_blockchain,
        neural_model=model,
        performance_monitor=monitor
    )


@pytest.fixture
def network_data():
    """Create sample network data for testing."""
    peers = [f"peer{i}" for i in range(10)]
    latencies = {p: 50 + i * 10 for i, p in enumerate(peers)}
    bandwidth = {p: 1000 - i * 100 for i, p in enumerate(peers)}
    reliability = {p: 0.9 - (i * 0.05) for i, p in enumerate(peers)}
    return {
        'peers': peers,
        'latencies': latencies,
        'bandwidth': bandwidth,
        'reliability': reliability
    }


@pytest.fixture
def routing_table(network_data):
    """Create a routing table for testing."""
    table = RoutingTable()
    for peer in network_data['peers']:
        table.add_peer(
            peer,
            latency=network_data['latencies'][peer],
            bandwidth=network_data['bandwidth'][peer],
            reliability=network_data['reliability'][peer]
        )
    return table


@pytest.fixture
def network_optimizer(routing_table):
    """Create a network optimizer instance."""
    peer_discovery = PeerDiscovery()
    return NetworkOptimizer(
        routing_table=routing_table,
        peer_discovery=peer_discovery
    )


# ===== Transaction Optimizer Tests =====

class TestTransactionOptimizer:
    """Test the transaction optimizer functionality."""

    def test_initialization(self, transaction_optimizer):
        """Test if the transaction optimizer initializes correctly."""
        assert transaction_optimizer is not None
        assert transaction_optimizer.strategy == OptimizationStrategy.BALANCED

    def test_transaction_validation(self, transaction_optimizer):
        """Test transaction validation logic."""
        valid_txs, invalid_txs = transaction_optimizer.validate_all()
        # Based on our fixture, 1/3 of transactions should be invalid
        assert len(valid_txs) >= 6
        assert len(invalid_txs) <= 4

    def test_transaction_prioritization(self, transaction_optimizer):
        """Test prioritization of transactions by fee, size, and age."""
        prioritized = transaction_optimizer.prioritize_transactions()
        
        # Check that the first transaction has high priority
        assert prioritized[0].fee > 0.001
        
        # Test with different strategies
        transaction_optimizer.strategy = OptimizationStrategy.FEE_PRIORITY
        fee_prioritized = transaction_optimizer.prioritize_transactions()
        
        transaction_optimizer.strategy = OptimizationStrategy.SIZE_EFFICIENCY
        size_prioritized = transaction_optimizer.prioritize_transactions()
        
        # Different strategies should produce different orderings
        assert fee_prioritized[0] != size_prioritized[0]

    def test_optimize_batch(self, transaction_optimizer):
        """Test batch optimization for inclusion in a block."""
        batch = transaction_optimizer.optimize_for_next_block(max_size=3000)
        assert len(batch) > 0
        assert sum(tx.size for tx in batch) <= 3000

    def test_optimization_performance(self, transaction_optimizer):
        """Test optimization performance with larger transaction pools."""
        # Patch the transaction pool to have more transactions
        with patch.object(transaction_optimizer, 'transaction_pool') as mock_pool:
            # Create a large pool of 1000 transactions
            txs = [Mock() for _ in range(1000)]
            for i, tx in enumerate(txs):
                tx.fee = 0.001 * (i % 10 + 1)
                tx.size = 500 + (i % 5) * 100
                tx.validate = Mock(return_value=True)
            mock_pool.get_all_transactions.return_value = txs
            
            # Measure optimization time
            start = time.time()
            optimized = transaction_optimizer.prioritize_transactions()
            end = time.time()
            
            assert len(optimized) == 1000
            # Optimization should be reasonably fast (adjust threshold as needed)
            assert end - start < 1.0, "Optimization took too long"

    def test_heuristic_adjustment(self, transaction_optimizer, mock_blockchain):
        """Test automatic adjustment of heuristics based on network conditions."""
        # Simulate changing network conditions
        mock_blockchain.get_difficulty.return_value = 0.01  # increased difficulty
        mock_blockchain.get_mempool_size.return_value = 500  # larger mempool
        
        # The optimizer should adjust its heuristics
        transaction_optimizer.adjust_to_network_conditions()
        
        # Verify that the optimizer now prioritizes differently
        assert transaction_optimizer.fee_weight != 1.0
        assert transaction_optimizer.size_weight != 1.0
        assert transaction_optimizer.age_weight != 1.0


# ===== Mining Optimizer Tests =====

class TestMiningOptimizer:
    """Test the mining optimizer functionality."""
    
    def test_initialization(self, mining_optimizer):
        """Test if the mining optimizer initializes correctly."""
        assert mining_optimizer is not None
        assert mining_optimizer.neural_model is not None
        assert mining_optimizer.performance_monitor is not None

    def test_hash_prediction(self, mining_optimizer):
        """Test the neural network's ability to predict promising hash ranges."""
        block_header = b"test_block_header_data"
        nonce_range = mining_optimizer.predict_optimal_nonce_range(block_header)
        
        assert len(nonce_range) == 2
        assert nonce_range[0] >= 0
        assert nonce_range[1] > nonce_range[0]

    @patch('micro_os.ai.mining_optimizer.NeuralHashingModel.predict')
    def test_neural_network_usage(self, mock_predict, mining_optimizer):
        """Test that the mining optimizer correctly uses the neural network."""
        mock_predict.return_value = np.array([0.1, 0.2, 0.8, 0.3, 0.05])
        
        block_header = b"test_block_header_data"
        mining_optimizer.predict_optimal_nonce_range(block_header)
        
        # Check that the neural model was called
        mock_predict.assert_called_once()

    def test_adaptive_difficulty(self, mining_optimizer, mock_blockchain):
        """Test adaptation to changing mining difficulty."""
        # Start with a low difficulty
        mock_blockchain.get_difficulty.return_value = 0.0001
        strategy1 = mining_optimizer.get_mining_strategy()
        
        # Change to high difficulty
        mock_blockchain.get_difficulty.return_value = 0.1
        mining_optimizer.update_network_conditions()
        strategy2 = mining_optimizer.get_mining_strategy()
        
        # Strategies should be different
        assert strategy1 != strategy2

    def test_performance_monitoring(self, mining_optimizer):
        """Test mining performance monitoring and learning."""
        # Simulate some mining attempts
        for i in range(10):
            success = i % 2 == 0  # Alternate between success and failure
            mining_optimizer.record_mining_attempt(
                block_header=f"header{i}".encode(),
                nonce_start=i * 1000,
                nonce_end=i * 1000 + 999,
                hash_rate=10e6,  # 10 MH/s
                success=success,
                nonce_found=i * 1000 + 500 if success else None,
                time_spent=1.0
            )
        
        stats = mining_optimizer.get_performance_stats()
        assert stats['total_attempts'] == 10
        assert stats['success_rate'] == 0.5
        assert stats['avg_time_to_solution'] > 0

    def test_neural_network_training(self, mining_optimizer):
        """Test that the neural network trains with mining feedback."""
        # Before training
        with patch.object(mining_optimizer.neural_model, 'train') as mock_train:
            # Simulate successful mining
            mining_optimizer.record_mining_attempt(
                block_header=b"successful_header",
                nonce_start=0,
                nonce_end=999,
                hash_rate=10e6,
                success=True,
                nonce_found=500,
                time_spent=0.5
            )
            
            # The optimizer should have updated the model
            mock_train.assert_called_once()


# ===== Network Optimizer Tests =====

class TestNetworkOptimizer:
    """Test the network optimizer functionality."""
    
    def test_initialization(self, network_optimizer):
        """Test if the network optimizer initializes correctly."""
        assert network_optimizer is not None
        assert network_optimizer.routing_table is not None
        assert network_optimizer.peer_discovery is not None

    def test_optimal_peer_selection(self, network_optimizer):
        """Test selection of optimal peers for different operations."""
        # Test for fast transaction broadcast
        tx_peers = network_optimizer.get_optimal_peers_for_transaction_broadcast(count=3)
        assert len(tx_peers) == 3
        
        # Test for reliable block propagation
        block_peers = network_optimizer.get_optimal_peers_for_block_propagation(count=3)
        assert len(block_peers) == 3
        
        # The optimal peers for different operations might differ
        assert set(tx_peers) != set(block_peers)

    def test_routing_optimization(self, network_optimizer, network_data):
        """Test optimization of the routing table based on network metrics."""
        # Get initial routing path
        source = "local"
        destination = network_data['peers'][5]
        initial_path = network_optimizer.get_optimal_path(source, destination)
        
        # Update network conditions
        for peer in network_data['peers'][:3]:
            network_optimizer.update_peer_metrics(
                peer,
                latency=200,  # Increased latency
                bandwidth=100,  # Decreased bandwidth
                reliability=0.5  # Decreased reliability
            )
        
        # Get new routing path
        new_path = network_optimizer.get_optimal_path(source, destination)
        
        # Paths should be different after network condition changes
        assert initial_path != new_path

    def test_peer_discovery(self, network_optimizer):
        """Test peer discovery mechanisms."""
        # Mock an external source of peer information
        mock_discovered_peers = [f"new_peer{i}" for i in range(5)]
        
        with patch.object(network_optimizer.peer_discovery, 'discover_peers') as mock_discover:
            mock_discover.return_value = mock_discovered_peers
            
            # Discover new peers
            new_peers = network_optimizer.discover_and_evaluate_peers()
            
            assert len(new_peers) == 5
            # Check that the peers were added to the routing table
            for peer in new_peers:
                assert network_optimizer.routing_table.has_peer(peer)

    def test_network_condition_adaptation(self, network_optimizer):
        """Test adaptation to changing network conditions."""
        # Initial network condition
        assert network_optimizer.get_network_condition() == NetworkCondition.NORMAL
        
        # Simulate deteriorating network conditions
        for peer in network_optimizer.routing_table.get_all_peers()[:5]:
            network_optimizer.update_peer_metrics(
                peer,
                latency=500,  # High latency
                bandwidth=50,  # Low bandwidth
                reliability=0.3  # Low reliability
            )
        
        # Check that the network condition has changed
        network_optimizer.evaluate_network_condition()
        assert network_optimizer.get_network_condition() == NetworkCondition.CONGESTED
        
        # Test that the optimizer adapts its routing strategy
        normal_strategy = network_optimizer.get_routing_strategy(NetworkCondition.NORMAL)
        congested_strategy = network_optimizer.get_routing_strategy(NetworkCondition.CONGESTED)
        assert normal_strategy != congested_strategy

    def test_peer_ranking(self, network_optimizer, network_data):
        """Test ranking of peers based on multiple metrics."""
        ranked_peers = network_optimizer.rank_peers(
            latency_weight=0.5,
            bandwidth_weight=0.3,
            reliability_weight=0.2
        )
        
        # The best peer should be at the top of the ranking
        assert ranked_peers[0] == network_data['peers'][0]
        
        # Change the weights to prioritize bandwidth
        bw_ranked = network_optimizer.rank_peers(
            latency_weight=0.1,
            bandwidth_weight=0.8,
            reliability_weight=0.1
        )
        
        # Rankings should be different with different weights
        assert ranked_peers != bw_ranked


# ===== Integration Tests =====

@pytest.mark.integration
class TestAIModuleIntegration:
    """Test integration between AI optimization modules."""
    
    def test_transaction_to_mining_integration(self, transaction_optimizer, mining_optimizer):
        """Test integration between transaction optimizer and mining optimizer."""
        # Create a mock block template from optimized transactions
        optim

