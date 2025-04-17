import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import sys
import json
from datetime import datetime, timedelta

# Import the ContainerManager class
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from micro_os.containers.container_manager import (
    ContainerManager, 
    Container, 
    ResourceAllocation, 
    ContainerNetworkConfig,
    BlockchainNodeTemplate,
    ContainerHealthStatus,
    ResourceUsageMetrics
)


# Fixtures for testing
@pytest.fixture
def mock_vm_environment():
    """Create a mock VM environment for testing."""
    mock_vm = Mock()
    mock_vm.get_available_resources.return_value = {
        'cpu': 8,
        'memory': 16384,  # 16GB in MB
        'disk': 100000,   # 100GB in MB
        'network': 1000   # 1Gbps
    }
    mock_vm.id = "test-vm-001"
    mock_vm.hostname = "test-vm-001.local"
    return mock_vm


@pytest.fixture
def container_manager(mock_vm_environment):
    """Create a ContainerManager instance for testing."""
    manager = ContainerManager(vm_environment=mock_vm_environment)
    # Clean up any containers that might exist
    manager.containers = {}
    return manager


@pytest.fixture
def sample_container_config():
    """Sample container configuration for testing."""
    return {
        'name': 'test-container',
        'image': 'ubuntu:latest',
        'command': '/bin/bash',
        'resources': {
            'cpu': 2,
            'memory': 2048,  # 2GB in MB
            'disk': 10000,   # 10GB in MB
            'network': 100   # 100Mbps
        },
        'ports': {
            '80/tcp': 8080,
            '443/tcp': 8443
        },
        'environment': {
            'ENV_VAR1': 'value1',
            'ENV_VAR2': 'value2'
        }
    }


@pytest.fixture
def blockchain_node_template():
    """Blockchain node template for testing."""
    return {
        'type': 'ethereum',
        'version': '1.10.23',
        'network': 'mainnet',
        'sync_mode': 'fast',
        'resources': {
            'cpu': 4,
            'memory': 8192,  # 8GB in MB
            'disk': 500000,  # 500GB in MB
            'network': 500   # 500Mbps
        },
        'ports': {
            '30303/tcp': 30303,  # P2P
            '30303/udp': 30303,  # P2P discovery
            '8545/tcp': 8545,    # RPC
            '8546/tcp': 8546     # WebSocket
        },
    }


# Test Container Lifecycle Management
class TestContainerLifecycle:
    """Test container creation and lifecycle management."""

    def test_create_container(self, container_manager, sample_container_config):
        """Test container creation."""
        container_id = container_manager.create_container(**sample_container_config)
        assert container_id is not None
        assert container_id in container_manager.containers
        
        container = container_manager.containers[container_id]
        assert container.name == sample_container_config['name']
        assert container.image == sample_container_config['image']
        assert container.status == 'created'

    def test_start_container(self, container_manager, sample_container_config):
        """Test starting a container."""
        # Create a container first
        container_id = container_manager.create_container(**sample_container_config)
        
        # Start the container
        result = container_manager.start_container(container_id)
        assert result is True
        
        # Check if the container status has been updated
        container = container_manager.containers[container_id]
        assert container.status == 'running'
        assert container.started_at is not None

    def test_stop_container(self, container_manager, sample_container_config):
        """Test stopping a container."""
        # Create and start a container
        container_id = container_manager.create_container(**sample_container_config)
        container_manager.start_container(container_id)
        
        # Stop the container
        result = container_manager.stop_container(container_id)
        assert result is True
        
        # Check if the container status has been updated
        container = container_manager.containers[container_id]
        assert container.status == 'stopped'
        assert container.stopped_at is not None

    def test_pause_container(self, container_manager, sample_container_config):
        """Test pausing a container."""
        # Create and start a container
        container_id = container_manager.create_container(**sample_container_config)
        container_manager.start_container(container_id)
        
        # Pause the container
        result = container_manager.pause_container(container_id)
        assert result is True
        
        # Check if the container status has been updated
        container = container_manager.containers[container_id]
        assert container.status == 'paused'

    def test_resume_container(self, container_manager, sample_container_config):
        """Test resuming a paused container."""
        # Create, start, and pause a container
        container_id = container_manager.create_container(**sample_container_config)
        container_manager.start_container(container_id)
        container_manager.pause_container(container_id)
        
        # Resume the container
        result = container_manager.resume_container(container_id)
        assert result is True
        
        # Check if the container status has been updated
        container = container_manager.containers[container_id]
        assert container.status == 'running'

    def test_container_full_lifecycle(self, container_manager, sample_container_config):
        """Test the full lifecycle of a container: create -> start -> pause -> resume -> stop."""
        # Create
        container_id = container_manager.create_container(**sample_container_config)
        assert container_manager.containers[container_id].status == 'created'
        
        # Start
        container_manager.start_container(container_id)
        assert container_manager.containers[container_id].status == 'running'
        
        # Pause
        container_manager.pause_container(container_id)
        assert container_manager.containers[container_id].status == 'paused'
        
        # Resume
        container_manager.resume_container(container_id)
        assert container_manager.containers[container_id].status == 'running'
        
        # Stop
        container_manager.stop_container(container_id)
        assert container_manager.containers[container_id].status == 'stopped'


# Test Resource Allocation
class TestResourceAllocation:
    """Test container resource allocation functionality."""

    def test_resource_allocation_during_creation(self, container_manager, sample_container_config):
        """Test resource allocation when creating a container."""
        # Create a container
        container_id = container_manager.create_container(**sample_container_config)
        
        # Check if resources were allocated correctly
        container = container_manager.containers[container_id]
        assert container.resources.cpu == sample_container_config['resources']['cpu']
        assert container.resources.memory == sample_container_config['resources']['memory']
        assert container.resources.disk == sample_container_config['resources']['disk']
        assert container.resources.network == sample_container_config['resources']['network']

    def test_resource_allocation_limits(self, container_manager, mock_vm_environment):
        """Test that resource allocation respects VM resource limits."""
        # Set up a resource request that exceeds VM limits
        config = {
            'name': 'resource-heavy-container',
            'image': 'ubuntu:latest',
            'resources': {
                'cpu': 16,      # Exceeds the 8 CPUs available
                'memory': 32768,  # Exceeds the 16GB available
                'disk': 200000,   # Exceeds the 100GB available
                'network': 2000   # Exceeds the 1Gbps available
            }
        }
        
        # Try to create a container with excessive resources
        with pytest.raises(ValueError, match="Insufficient resources"):
            container_manager.create_container(**config)

    def test_resource_overcommitment_prevention(self, container_manager, sample_container_config):
        """Test that the system prevents resource overcommitment."""
        # Create 4 containers with 2 CPUs each (total: 8 CPUs, which is the limit)
        for i in range(4):
            sample_container_config['name'] = f'container-{i}'
            container_manager.create_container(**sample_container_config)
        
        # Try to create one more container - should fail due to CPU limit
        sample_container_config['name'] = 'container-overcommit'
        with pytest.raises(ValueError, match="Insufficient resources"):
            container_manager.create_container(**sample_container_config)

    def test_resource_release_after_container_stop(self, container_manager, sample_container_config):
        """Test that resources are properly released after stopping a container."""
        # Create and start a container
        container_id = container_manager.create_container(**sample_container_config)
        container_manager.start_container(container_id)
        
        # Check initial resource usage
        initial_usage = container_manager.get_total_resource_usage()
        
        # Stop the container
        container_manager.stop_container(container_id)
        
        # Check resource usage after stopping
        final_usage = container_manager.get_total_resource_usage()
        
        # The resource usage should be lower after stopping
        assert final_usage['cpu'] < initial_usage['cpu']
        assert final_usage['memory'] < initial_usage['memory']
        assert final_usage['network'] < initial_usage['network']


# Test Container Networking
class TestContainerNetworking:
    """Test container networking functionality."""

    @patch('micro_os.containers.container_manager.VirtualNetwork')
    def test_container_networking_setup(self, mock_virtual_network, container_manager, sample_container_config):
        """Test setting up container networking."""
        # Add network configuration to the container config
        sample_container_config['network_config'] = {
            'type': 'bridge',
            'subnet': '172.17.0.0/16',
            'ip_address': '172.17.0.2'
        }
        
        # Create a container with network config
        container_id = container_manager.create_container(**sample_container_config)
        
        # Check if network was configured correctly
        container = container_manager.containers[container_id]
        assert container.network_config.type == 'bridge'
        assert container.network_config.subnet == '172.17.0.0/16'
        assert container.network_config.ip_address == '172.17.0.2'
        
        # Verify network setup was called
        mock_virtual_network.assert_called_once()

    @patch('micro_os.containers.container_manager.ContainerManager.connect_containers')
    def test_container_to_container_networking(self, mock_connect, container_manager, sample_container_config):
        """Test networking between containers."""
        # Create two containers
        sample_container_config['name'] = 'container1'
        container1_id = container_manager.create_container(**sample_container_config)
        
        sample_container_config['name'] = 'container2'
        container2_id = container_manager.create_container(**sample_container_config)
        
        # Connect the containers
        container_manager.connect_containers(container1_id, container2_id)
        
        # Verify connect_containers was called with the right arguments
        mock_connect.assert_called_once_with(container1_id, container2_id)

    @patch('micro_os.containers.container_manager.VMNetworkProtocol')
    def test_container_cross_vm_networking(self, mock_vm_protocol, container_manager, sample_container_config):
        """Test container networking across different VMs."""
        # Create a container for cross-VM networking
        sample_container_config['network_config'] = {
            'type': 'overlay',
            'subnet': '10.0.0.0/16',
            'ip_address': '10.0.0.2'
        }
        
        container_id = container_manager.create_container(**sample_container_config)
        
        # Set up a remote VM and container
        remote_vm_id = "remote-vm-001"
        remote_container_id = "remote-container-001"
        
        # Test cross-VM container connection
        container_manager.connect_to_remote_container(
            container_id, 
            remote_vm_id, 
            remote_container_id
        )
        
        # Verify VM network protocol was used for remote connection
        mock_vm_protocol.connect_containers.assert_called_once()


# Test Blockchain Node Templates
class TestBlockchainNodeTemplates:
    """Test container templates for blockchain node deployment."""

    def test_ethereum_node_template(self, container_manager, blockchain_node_template):
        """Test creating an Ethereum node from a template."""
        # Create an Ethereum node container from the template
        container_id = container_manager.create_from_template(
            name="ethereum-node-1",
            template=blockchain_node_template
        )
        
        # Verify the container was created with the correct specifications
        container = container_manager.containers[container_id]
        assert container.image == f"ethereum:{blockchain_node_template['version']}"
        assert container.resources.cpu == blockchain_node_template['resources']['cpu']
        assert container.resources.memory == blockchain_node_template['resources']['memory']
        assert container.status == 'created'

    def test_custom_blockchain_template(self, container_manager):
        """Test creating a custom blockchain node from template."""
        # Define a custom blockchain template
        custom_template = {
            'type': 'custom_chain',
            'version': '0.1.0',
            'image': 'custom-blockchain:latest',
            'network': 'testnet',
            'resources': {
                'cpu': 2,
                'memory': 4096,
                'disk': 20000,
                'network': 100
            },
            'ports': {
                '26656/tcp': 26656,  # P2P
                '26657/tcp': 26657   # RPC
            },
            'environment': {
                'CHAIN_ID': 'test-chain-1',
                'VALIDATOR_MODE': 'true'
            }
        }
        
        # Create a container from the custom template
        container_id = container_manager.create_from_template(
            name="custom-node-1",
            template=custom_template
        )
        
        # Verify the container was created with the correct specifications
        container = container_manager.containers[container_id]
        assert container.image == custom_template['image']
        assert container.environment['CHAIN_ID'] == custom_template['environment']['CHAIN_ID']
        assert container.environment['VALIDATOR_MODE'] == custom_template['environment']['VALIDATOR_MODE']

    def test_blockchain_node_auto_configuration(self

