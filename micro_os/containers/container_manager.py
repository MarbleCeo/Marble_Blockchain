import os
import logging
import time
import json
import uuid
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import threading
import asyncio
import psutil

# Import VM environment integration
from micro_os.vm.vm_environment import VMEnvironment

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContainerStatus(Enum):
    """Container status states"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class NetworkMode(Enum):
    """Container networking modes"""
    BRIDGE = "bridge"
    HOST = "host"
    OVERLAY = "overlay"
    P2P = "p2p"


class ResourcePriority(Enum):
    """Resource allocation priorities"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class ContainerResources:
    """Resource limits and reservations for containers"""
    cpu_limit: float = 1.0  # CPU cores
    memory_limit: int = 512  # MB
    cpu_reservation: float = 0.25  # CPU cores
    memory_reservation: int = 128  # MB
    gpu_enabled: bool = False
    gpu_memory: int = 0  # MB


@dataclass
class ContainerMetrics:
    """Container runtime metrics for monitoring"""
    cpu_usage: float = 0.0  # Percentage
    memory_usage: int = 0  # MB
    network_rx_bytes: int = 0
    network_tx_bytes: int = 0
    disk_read_bytes: int = 0
    disk_write_bytes: int = 0
    last_updated: float = 0.0


@dataclass
class ContainerConfig:
    """Container configuration"""
    id: str
    name: str
    image: str
    vm_id: str
    status: ContainerStatus = ContainerStatus.CREATED
    resources: ContainerResources = None
    network_mode: NetworkMode = NetworkMode.BRIDGE
    ports: Dict[int, int] = None  # host_port: container_port
    volumes: Dict[str, str] = None  # host_path: container_path
    environment: Dict[str, str] = None
    command: str = ""
    priority: ResourcePriority = ResourcePriority.MEDIUM
    metrics: ContainerMetrics = None
    
    def __post_init__(self):
        """Initialize default values for optional fields"""
        if self.resources is None:
            self.resources = ContainerResources()
        if self.ports is None:
            self.ports = {}
        if self.volumes is None:
            self.volumes = {}
        if self.environment is None:
            self.environment = {}
        if self.metrics is None:
            self.metrics = ContainerMetrics()


class ContainerNetworkManager:
    """Manages container networking across VMs"""
    
    def __init__(self):
        self.networks = {}
        self.container_network_mappings = {}
        self.logger = logging.getLogger(__name__ + ".ContainerNetworkManager")
    
    def create_network(self, network_name: str, network_type: NetworkMode, subnet: str) -> bool:
        """Create a new network for containers"""
        try:
            if network_name in self.networks:
                self.logger.warning(f"Network {network_name} already exists")
                return False
                
            self.networks[network_name] = {
                "type": network_type,
                "subnet": subnet,
                "containers": [],
                "created_at": time.time()
            }
            self.logger.info(f"Created network {network_name} of type {network_type.value}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create network {network_name}: {str(e)}")
            return False
    
    def connect_container(self, container_id: str, network_name: str, ip_address: Optional[str] = None) -> bool:
        """Connect a container to a network"""
        try:
            if network_name not in self.networks:
                self.logger.error(f"Network {network_name} does not exist")
                return False
                
            network = self.networks[network_name]
            if container_id in network["containers"]:
                self.logger.warning(f"Container {container_id} already connected to network {network_name}")
                return True
                
            network["containers"].append(container_id)
            self.container_network_mappings[container_id] = {
                "network": network_name,
                "ip_address": ip_address,
                "connected_at": time.time()
            }
            self.logger.info(f"Connected container {container_id} to network {network_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect container {container_id} to network {network_name}: {str(e)}")
            return False
    
    def disconnect_container(self, container_id: str, network_name: str) -> bool:
        """Disconnect a container from a network"""
        try:
            if network_name not in self.networks:
                self.logger.error(f"Network {network_name} does not exist")
                return False
                
            network = self.networks[network_name]
            if container_id not in network["containers"]:
                self.logger.warning(f"Container {container_id} not connected to network {network_name}")
                return True
                
            network["containers"].remove(container_id)
            if container_id in self.container_network_mappings:
                del self.container_network_mappings[container_id]
            self.logger.info(f"Disconnected container {container_id} from network {network_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to disconnect container {container_id} from network {network_name}: {str(e)}")
            return False
    
    def setup_vm_to_vm_networking(self, source_vm_id: str, target_vm_id: str) -> bool:
        """Configure networking between two VMs for container communication"""
        try:
            # In a real implementation, this would configure overlay networking, VPNs,
            # or other mechanisms to allow containers on different VMs to communicate
            network_name = f"vm-to-vm-{source_vm_id}-{target_vm_id}"
            
            self.create_network(
                network_name=network_name,
                network_type=NetworkMode.OVERLAY,
                subnet="10.0.{}.0/24".format(hash(network_name) % 255)
            )
            
            self.logger.info(f"Set up VM-to-VM networking between {source_vm_id} and {target_vm_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to set up VM-to-VM networking: {str(e)}")
            return False


class ContainerTemplateManager:
    """Manages predefined container templates for rapid deployment"""
    
    def __init__(self):
        self.templates = {}
        self._load_built_in_templates()
        self.logger = logging.getLogger(__name__ + ".ContainerTemplateManager")
    
    def _load_built_in_templates(self):
        """Load built-in container templates"""
        # Blockchain node templates
        self.templates["ethereum-node"] = {
            "image": "ethereum/client-go:latest",
            "resources": ContainerResources(
                cpu_limit=2.0,
                memory_limit=4096,
                cpu_reservation=1.0,
                memory_reservation=2048
            ),
            "network_mode": NetworkMode.P2P,
            "ports": {30303: 30303, 8545: 8545},
            "volumes": {"/blockchain/ethereum": "/root/.ethereum"},
            "command": "--syncmode light --http --http.api eth,net,web3",
            "description": "Ethereum blockchain node (geth) in light sync mode"
        }
        
        self.templates["bitcoin-node"] = {
            "image": "bitcoin-core:latest",
            "resources": ContainerResources(
                cpu_limit=2.0,
                memory_limit=8192,
                cpu_reservation=1.0,
                memory_reservation=4096
            ),
            "network_mode": NetworkMode.P2P,
            "ports": {8333: 8333, 8332: 8332},
            "volumes": {"/blockchain/bitcoin": "/root/.bitcoin"},
            "command": "-prune=550 -listen=1",
            "description": "Bitcoin Core node with pruning enabled"
        }
        
        # AI workload templates
        self.templates["tensorflow-inference"] = {
            "image": "tensorflow/serving:latest",
            "resources": ContainerResources(
                cpu_limit=4.0,
                memory_limit=4096,
                cpu_reservation=2.0,
                memory_reservation=2048,
                gpu_enabled=True,
                gpu_memory=2048
            ),
            "network_mode": NetworkMode.BRIDGE,
            "ports": {8501: 8501},
            "volumes": {"/ai/models": "/models"},
            "environment": {"MODEL_NAME": "default"},
            "description": "TensorFlow Serving for model inference"
        }
    
    def create_template(self, name: str, template_config: Dict[str, Any]) -> bool:
        """Register a new container template"""
        try:
            if name in self.templates:
                self.logger.warning(f"Template {name} already exists, overwriting")
            
            # Validate template configuration
            required_fields = ["image", "resources", "description"]
            for field in required_fields:
                if field not in template_config:
                    self.logger.error(f"Template {name} is missing required field: {field}")
                    return False
            
            self.templates[name] = template_config
            self.logger.info(f"Created template {name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create template {name}: {str(e)}")
            return False
    
    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a container template by name"""
        if name not in self.templates:
            self.logger.error(f"Template {name} does not exist")
            return None
        return self.templates[name]
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available templates"""
        return [{"name": name, **template} for name, template in self.templates.items()]
    
    def instantiate_from_template(self, template_name: str, container_name: str, vm_id: str) -> Optional[ContainerConfig]:
        """Create a container configuration from a template"""
        try:
            template = self.get_template(template_name)
            if not template:
                return None
            
            container_id = str(uuid.uuid4())
            
            # Create container resources from template
            resources = template.get("resources", ContainerResources())
            if not isinstance(resources, ContainerResources):
                resources = ContainerResources(**resources)
            
            # Create container configuration
            container_config = ContainerConfig(
                id=container_id,
                name=container_name,
                image=template["image"],
                vm_id=vm_id,
                resources=resources,
                network_mode=template.get("network_mode", NetworkMode.BRIDGE),
                ports=template.get("ports", {}),
                volumes=template.get("volumes", {}),
                environment=template.get("environment", {}),
                command=template.get("command", ""),
                priority=template.get("priority", ResourcePriority.MEDIUM)
            )
            
            self.logger.info(f"Instantiated container {container_name} from template {template_name}")
            return container_config
        except Exception as e:
            self.logger.error(f"Failed to instantiate container from template {template_name}: {str(e)}")
            return None


class ResourceAllocator:
    """Allocates and balances resources across containers"""
    
    def __init__(self):
        self.allocated_resources = {}  # vm_id -> resources
        self.container_allocations = {}  # container_id -> resources
        self.logger = logging.getLogger(__name__ + ".ResourceAllocator")
    
    def get_vm_available_resources(self, vm_id: str) -> Dict[str, Any]:
        """Get available resources on a VM"""
        try:
            # This would typically use the VM API to get real metrics
            # For demonstration, we're using psutil for the local system
            vm_resources = {
                "cpu_total": psutil.cpu_count(logical=True),
                "cpu_available": psutil.cpu_count(logical=True) - sum(
                    self.container_allocations.get(c_id, {}).get("cpu_reservation", 0)
                    for c_id, config in self.allocated_resources.get(vm_id, {}).items()
                ),
                "memory_total": psutil.virtual_memory().total // (1024 * 1024),  # MB
                "memory_available": psutil.virtual_memory().available // (1024 * 1024),  # MB
                "gpu_available": False,  # Would be determined by GPU monitoring
            }
            return vm_resources
        except Exception as e:
            self.logger.error(f"Failed to get VM resources for {vm_id}: {str(e)}")
            return {
                "cpu_total": 0,
                "cpu_available": 0,
                "memory_total": 0,
                "memory_available": 0,
                "gpu_available": False
            }
    
    def allocate_resources(self, container_config: ContainerConfig) -> bool:
        """Allocate resources for a container on its assigned VM"""
        try:
            vm_id = container_config.vm_id
            container_id = container_config.id
            
            # Check if VM exists in our allocation map
            if vm_id not in self.allocated_resources:
                self.allocated_resources[vm_id] = {}
            
            # Get VM available resources
            vm_resources = self.get_vm_available_resources(vm_id)
            
            # Check if VM has enough resources
            if (vm_resources["cpu_available"] < container_config.resources.cpu_reservation or
                vm_resources["memory_available"] < container_config.resources.memory_reservation):
                self.logger.error(
                    f"VM {vm_id} doesn't have enough resources for container {container_id}. "
                    f"Available: {vm_resources['cpu_available']} CPU, {vm_resources['memory_available']} MB. "
                    f"Required: {container_config.resources.cpu_reservation} CPU, {container_config.resources.memory_reservation} MB."
                )
                return False
            
            # If GPU is required, check availability
            if container_config.resources.gpu_enabled and not vm

