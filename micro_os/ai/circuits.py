import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union, Set
import numpy as np
import logging
import uuid
import asyncio
from datetime import datetime

class LensRefractor:
    """
    LensRefractor class for managing neural circuits and resource allocation.
    Implements dynamic neural architecture for resource optimization.
    """
    def __init__(self, input_dim: int = 64, hidden_layers: List[int] = [128, 256, 128]):
        self.logger = logging.getLogger(__name__)
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        
        # Initialize the neural architecture
        self.circuit = self._build_circuit()
        self.optimizer = torch.optim.Adam(self.circuit.parameters())
        self.resource_patterns: Dict[str, torch.Tensor] = {}
        
        # Distributed circuit management
        self.distributed_circuits: Dict[str, Dict[str, Any]] = {}
        self.p2p_network_manager = None
        self.connected_peers: Set[str] = set()

    def _build_circuit(self) -> nn.Module:
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer for resource allocation
        layers.append(nn.Linear(prev_dim, self.input_dim))
        
        return nn.Sequential(*layers)

    async def generate_circuit(self, resource_requirements: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """
        Generate optimized neural circuits based on resource requirements.
        """
        try:
            # Convert resource requirements to tensor
            input_tensor = self._prepare_input(resource_requirements)
            
            # Generate circuit configuration
            with torch.no_grad():
                output = self.circuit(input_tensor)
            
            # Normalize and structure the output
            circuit_config = self._structure_output(output)
            
            # Store the generated pattern
            pattern_id = len(self.resource_patterns)
            self.resource_patterns[f"pattern_{pattern_id}"] = circuit_config
            
            return {
                "circuit_id": f"pattern_{pattern_id}",
                "configuration": circuit_config,
                "resource_mapping": self._map_resources(circuit_config, resource_requirements)
            }
        
        except Exception as e:
            self.logger.error(f"Circuit generation failed: {str(e)}")
            raise RuntimeError(f"Failed to generate circuit: {str(e)}")

    def _prepare_input(self, requirements: Dict[str, float]) -> torch.Tensor:
        """Convert resource requirements to normalized input tensor."""
        input_array = np.zeros(self.input_dim)
        
        # Map requirements to input array
        for i, (key, value) in enumerate(requirements.items()):
            if i < self.input_dim:
                input_array[i] = value
        
        # Normalize
        input_array = (input_array - np.mean(input_array)) / (np.std(input_array) + 1e-8)
        return torch.FloatTensor(input_array).unsqueeze(0)

    def _structure_output(self, output: torch.Tensor) -> torch.Tensor:
        """Structure and normalize the circuit output."""
        output = output.squeeze(0)
        # Apply softmax for resource distribution
        return torch.nn.functional.softmax(output, dim=0)

    def _map_resources(self, circuit_config: torch.Tensor, 
                    requirements: Dict[str, float]) -> Dict[str, float]:
        """Map circuit configuration to resource allocations."""
        resource_mapping = {}
        config_array = circuit_config.numpy()
        
        for i, (resource, requirement) in enumerate(requirements.items()):
            if i < len(config_array):
                # Scale the configuration based on requirement
                resource_mapping[resource] = float(config_array[i] * requirement)
        
        return resource_mapping

    async def optimize_circuit(self, circuit_id: str, 
                            performance_metrics: Dict[str, float],
                            distributed: bool = False) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """
        Optimize existing circuit based on performance metrics.
        
        Args:
            circuit_id: ID of the circuit to optimize
            performance_metrics: Metrics to use for optimization
            distributed: If True, performs distributed optimization across nodes
        
        Returns:
            Dictionary containing optimization results
        """
        try:
            # Check if this is a distributed circuit
            is_distributed_circuit = circuit_id.startswith("dist_") and circuit_id in self.distributed_circuits
            
            # For distributed circuits, validate the circuit ID differently
            if distributed or is_distributed_circuit:
                if circuit_id not in self.distributed_circuits:
                    raise ValueError(f"Distributed circuit {circuit_id} not found")
                    
                # If distributed optimization is requested, aggregate metrics from peers
                if distributed and self.p2p_network_manager is not None:
                    # This would normally involve an actual network call to collect metrics
                    self.logger.info(f"Aggregating metrics for distributed circuit {circuit_id}")
                    
                    # Use the circuit's configuration as the base pattern for optimization
                    base_config = self.distributed_circuits[circuit_id]["base_configuration"]
                    self.resource_patterns[circuit_id] = base_config
                    
            elif circuit_id not in self.resource_patterns:
                raise ValueError(f"Circuit {circuit_id} not found")
            
            # Convert metrics to tensor
            metric_tensor = self._prepare_input(performance_metrics)
            
            # Optimization step
            self.optimizer.zero_grad()
            output = self.circuit(metric_tensor)
            
            # Calculate loss based on performance metrics
            target = torch.FloatTensor(list(performance_metrics.values()))
            loss = nn.MSELoss()(output.squeeze(), target)
            
            loss.backward()
            self.optimizer.step()
            
            # Generate new optimized configuration
            with torch.no_grad():
                optimized_config = self._structure_output(self.circuit(metric_tensor))
                
            # Update stored pattern
            self.resource_patterns[circuit_id] = optimized_config
            
            return {
                "circuit_id": circuit_id,
                "optimized_mapping": self._map_resources(
                    optimized_config, performance_metrics
                ),
                "distributed": distributed
            }
            
            # For distributed circuits, propagate optimized parameters to peers
            if distributed and is_distributed_circuit and self.p2p_network_manager is not None:
                await self.sync_circuit_state(
                    circuit_id=circuit_id,
                    peer_list=self.distributed_circuits[circuit_id]["peer_ids"]
                )
            
        
        except Exception as e:
            self.logger.error(f"Circuit optimization failed: {str(e)}")
            raise RuntimeError(f"Failed to optimize circuit: {str(e)}")

    async def create_distributed_circuit(
        self, 
        network_config: Dict[str, Any], 
        peer_ids: List[str],
        resource_requirements: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Creates a distributed neural circuit across multiple nodes in the network.
        
        Args:
            network_config: Configuration parameters for the distributed network
            peer_ids: List of peer IDs that will participate in the distributed circuit
            resource_requirements: Resource requirements for the circuit
            
        Returns:
            Dictionary containing circuit ID and distribution metadata
        """
        try:
            self.logger.info(f"Creating distributed circuit with {len(peer_ids)} peers")
            
            # Generate a unique ID for this distributed circuit
            circuit_id = f"dist_{uuid.uuid4().hex[:8]}"
            
            # Create a base circuit configuration
            base_circuit = await self.generate_circuit(resource_requirements)
            
            # Store distributed circuit metadata
            self.distributed_circuits[circuit_id] = {
                "created_at": datetime.now().isoformat(),
                "peer_ids": peer_ids,
                "network_config": network_config,
                "base_configuration": base_circuit["configuration"],
                "resource_requirements": resource_requirements,
                "peer_states": {peer: {"connected": False, "last_sync": None} for peer in peer_ids},
                "is_active": True
            }
            
            # Prepare the response with distribution metadata
            result = {
                "circuit_id": circuit_id,
                "distribution_metadata": {
                    "peer_count": len(peer_ids),
                    "network_topology": network_config.get("topology", "mesh"),
                    "synchronization_frequency": network_config.get("sync_frequency", 10),
                    "base_resource_allocation": base_circuit["resource_mapping"]
                },
                "status": "initialized"
            }
            
            self.logger.info(f"Distributed circuit {circuit_id} created successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Distributed circuit creation failed: {str(e)}")
            raise RuntimeError(f"Failed to create distributed circuit: {str(e)}")
    
    async def sync_circuit_state(
        self, 
        circuit_id: str, 
        peer_list: List[str],
        force_sync: bool = False
    ) -> Dict[str, Any]:
        """
        Synchronizes model parameters across multiple nodes in a distributed circuit.
        
        Args:
            circuit_id: ID of the distributed circuit to synchronize
            peer_list: List of peers to synchronize with
            force_sync: If True, forces synchronization regardless of time since last sync
            
        Returns:
            Dictionary containing synchronization status and metrics
        """
        try:
            if circuit_id not in self.distributed_circuits:
                raise ValueError(f"Distributed circuit {circuit_id} not found")
                
            circuit_data = self.distributed_circuits[circuit_id]
            
            # Check if all peers are valid participants in this circuit
            invalid_peers = [p for p in peer_list if p not in circuit_data["peer_ids"]]
            if invalid_peers:
                self.logger.warning(f"Invalid peers for circuit {circuit_id}: {invalid_peers}")
                peer_list = [p for p in peer_list if p in circuit_data["peer_ids"]]
            
            if not peer_list:
                raise ValueError("No valid peers to synchronize with")
            
            # Simulate parameter synchronization
            sync_timestamp = datetime.now().isoformat()
            sync_results = {}
            
            # Update peer states with sync information
            for peer in peer_list:
                circuit_data["peer_states"][peer] = {
                    "connected": True,
                    "last_sync": sync_timestamp
                }
                sync_results[peer] = {
                    "status": "success",
                    "timestamp": sync_timestamp
                }
            
            # Update the distributed circuit state
            circuit_data["last_sync"] = sync_timestamp
            circuit_data["sync_count"] = circuit_data.get("sync_count", 0) + 1
            
            return {
                "circuit_id": circuit_id,
                "sync_status": "completed",
                "synced_peers": sync_results,
                "sync_timestamp": sync_timestamp,
                "sync_count": circuit_data["sync_count"]
            }
            
        except Exception as e:
            self.logger.error(f"Circuit state synchronization failed: {str(e)}")
            return {
                "circuit_id": circuit_id,
                "sync_status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def register_with_p2p_network(self, p2p_network_manager: Any) -> Dict[str, Any]:
        """
        Connects the neural circuit to the P2P network for distributed operations.
        
        Args:
            p2p_network_manager: The P2P network manager instance to connect with
            
        Returns:
            Dictionary containing connection status and metadata
        """
        try:
            self.logger.info("Registering LensRefractor with P2P network")
            
            # Store reference to the P2P network manager
            self.p2p_network_manager = p2p_network_manager
            
            # Register callbacks for neural circuit operations
            if hasattr(p2p_network_manager, "register_handler"):
                p2p_network_manager.register_handler(
                    "neural_circuit_sync", 
                    self._handle_circuit_sync_request
                )
                
            # Get connected peers from the network manager
            if hasattr(p2p_network_manager, "get_connected_peers"):
                peers = p2p_network_manager.get_connected_peers()
                self.connected_peers = set(peers)
            
            registration_status = {
                "status": "connected",
                "connection_timestamp": datetime.now().isoformat(),
                "peer_count": len(self.connected_peers),
                "capabilities": ["distributed_training", "parameter_sync", "federated_optimization"]
            }
            
            self.logger.info(f"Successfully registered with P2P network. Connected to {len(self.connected_peers)} peers")
            return registration_status
            
        except Exception as e:
            self.logger.error(f"P2P network registration failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _handle_circuit_sync_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Internal handler for circuit synchronization requests from peers."""
        circuit_id = request.get("circuit_id")
        if not circuit_id or circuit_id not in self.distributed_circuits:
            return {"status": "error", "message": "Invalid circuit ID"}
            
        return await self.sync_circuit_state(
            circuit_id=circuit_id,
            peer_list=[request.get("peer_id")],
            force_sync=request.get("force_sync", False)
        )
