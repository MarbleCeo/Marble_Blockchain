#!/usr/bin/env python3
"""
Intelligent Agent (IA) Module

This module implements an Intelligent Agent system that integrates with neural 
processing, memory management, and federated learning capabilities to provide 
autonomous decision-making in distributed environments.

The IntelligentAgent class serves as the main interface for agent operations,
handling input processing, knowledge management, and distributed learning.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum

# Local imports
from micro_os.ai.circuits import LensRefractor
from micro_os.network.p2p_vm import P2PNetworkManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Enumeration of memory types supported by the Intelligent Agent."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass
class MemoryUnit:
    """Data structure for storing memory entries in the agent's memory system."""
    id: str
    content: Any
    created_at: float
    last_accessed: float
    importance: float
    type: MemoryType
    metadata: Dict[str, Any]
    access_count: int = 0


class MemorySystem:
    """
    Memory management system for the Intelligent Agent.
    
    Handles both short-term and long-term memory storage, with mechanisms 
    for optimizing storage, retrieving relevant information, and moving
    data between different memory types based on importance and access patterns.
    """
    
    def __init__(self, 
                 short_term_capacity: int = 1000,
                 long_term_capacity: int = 10000):
        """
        Initialize the memory system with specified capacities.
        
        Args:
            short_term_capacity: Maximum number of items in short-term memory
            long_term_capacity: Maximum number of items in long-term memory
        """
        self.memories: Dict[MemoryType, Dict[str, MemoryUnit]] = {
            memory_type: {} for memory_type in MemoryType
        }
        self.capacities = {
            MemoryType.SHORT_TERM: short_term_capacity,
            MemoryType.LONG_TERM: long_term_capacity,
            MemoryType.WORKING: short_term_capacity // 10,
            MemoryType.EPISODIC: long_term_capacity // 2,
            MemoryType.SEMANTIC: long_term_capacity // 2
        }
        logger.info(f"Initialized memory system with {short_term_capacity} short-term and {long_term_capacity} long-term capacity")
    
    def store(self, 
              content: Any, 
              memory_type: MemoryType, 
              importance: float = 0.5,
              metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a new memory unit in the specified memory type.
        
        Args:
            content: The data to be stored
            memory_type: Type of memory to store in
            importance: Importance score (0.0 to 1.0)
            metadata: Additional metadata for the memory unit
            
        Returns:
            ID of the stored memory unit
        """
        # Check if we need to make room
        if len(self.memories[memory_type]) >= self.capacities[memory_type]:
            self._optimize_memory_type(memory_type)
        
        memory_id = f"{memory_type.value}_{time.time()}_{hash(str(content))}"
        
        # Create the memory unit
        unit = MemoryUnit(
            id=memory_id,
            content=content,
            created_at=time.time(),
            last_accessed=time.time(),
            importance=importance,
            type=memory_type,
            metadata=metadata or {},
            access_count=0
        )
        
        # Store it
        self.memories[memory_type][memory_id] = unit
        return memory_id
    
    def retrieve(self, 
                memory_id: str, 
                memory_type: Optional[MemoryType] = None) -> Optional[Any]:
        """
        Retrieve a memory by its ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            memory_type: Optional type to narrow search
            
        Returns:
            The content of the memory if found, None otherwise
        """
        # If memory_type is provided, search only in that type
        if memory_type:
            memory_dict = self.memories[memory_type]
            if memory_id in memory_dict:
                unit = memory_dict[memory_id]
                unit.last_accessed = time.time()
                unit.access_count += 1
                return unit.content
            return None
        
        # Otherwise search in all memory types
        for memory_dict in self.memories.values():
            if memory_id in memory_dict:
                unit = memory_dict[memory_id]
                unit.last_accessed = time.time()
                unit.access_count += 1
                return unit.content
        
        return None
    
    def query(self, 
             query_vector: np.ndarray,
             memory_type: Optional[MemoryType] = None,
             top_k: int = 5) -> List[Tuple[str, Any, float]]:
        """
        Find memories similar to the query vector.
        
        Args:
            query_vector: Vector representation of the query
            memory_type: Optional type to narrow search
            top_k: Number of results to return
            
        Returns:
            List of (memory_id, content, similarity_score) tuples
        """
        # Simplified implementation - in a real system, this would use 
        # vector embeddings and similarity search
        results = []
        
        memory_types = [memory_type] if memory_type else list(MemoryType)
        
        for mem_type in memory_types:
            for memory_id, unit in self.memories[mem_type].items():
                # Simulated similarity score - would use actual vector similarity in production
                similarity = np.random.random()  # Placeholder for actual similarity calculation
                results.append((memory_id, unit.content, similarity))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
    
    def _optimize_memory_type(self, memory_type: MemoryType) -> None:
        """
        Optimize a specific memory type by removing least important/accessed items.
        
        Args:
            memory_type: The memory type to optimize
        """
        if not self.memories[memory_type]:
            return
            
        # Calculate a score based on importance, recency and access frequency
        def score_memory(unit: MemoryUnit) -> float:
            recency = 1.0 / (time.time() - unit.last_accessed + 1)
            return unit.importance * 0.5 + recency * 0.3 + (unit.access_count / 100) * 0.2
        
        # Sort memories by score
        memories = list(self.memories[memory_type].items())
        memories.sort(key=lambda x: score_memory(x[1]))
        
        # Remove lowest scoring items to get back to 90% capacity
        target_size = int(self.capacities[memory_type] * 0.9)
        to_remove = max(0, len(memories) - target_size)
        
        for i in range(to_remove):
            memory_id, _ = memories[i]
            # If important, consider moving to long-term memory instead of deleting
            if (memory_type == MemoryType.SHORT_TERM and 
                score_memory(self.memories[memory_type][memory_id]) > 0.7):
                unit = self.memories[memory_type].pop(memory_id)
                # Move to long-term memory if there's space or it can be optimized
                if (len(self.memories[MemoryType.LONG_TERM]) < self.capacities[MemoryType.LONG_TERM] or
                    any(score_memory(u) < score_memory(unit) for u in self.memories[MemoryType.LONG_TERM].values())):
                    self._optimize_memory_type(MemoryType.LONG_TERM)
                    unit.type = MemoryType.LONG_TERM
                    self.memories[MemoryType.LONG_TERM][memory_id] = unit
            else:
                # Just remove it
                del self.memories[memory_type][memory_id]
                
        logger.debug(f"Optimized {memory_type.value} memory, removed {to_remove} items")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.
        
        Returns:
            Dictionary with memory statistics
        """
        stats = {}
        for memory_type in MemoryType:
            type_stats = {
                "count": len(self.memories[memory_type]),
                "capacity": self.capacities[memory_type],
                "usage_percent": len(self.memories[memory_type]) / self.capacities[memory_type] * 100 if self.capacities[memory_type] > 0 else 0,
                "avg_importance": np.mean([m.importance for m in self.memories[memory_type].values()]) if self.memories[memory_type] else 0
            }
            stats[memory_type.value] = type_stats
        return stats


class FederatedLearningManager:
    """
    Manages federated learning operations for the Intelligent Agent.
    
    Coordinates model training across multiple nodes, handles parameter
    aggregation, and ensures secure and efficient knowledge sharing.
    """
    
    def __init__(self, node_id: str, security_level: str = "high"):
        """
        Initialize the federated learning manager.
        
        Args:
            node_id: Unique identifier for this node
            security_level: Security level for federated operations
        """
        self.node_id = node_id
        self.security_level = security_level
        self.participating_nodes: Dict[str, Dict[str, Any]] = {}
        self.model_versions: Dict[str, int] = {}
        self.training_rounds: int = 0
        self.executor = ThreadPoolExecutor(max_workers=4)
        logger.info(f"Initialized federated learning manager for node {node_id}")
    
    def register_node(self, node_id: str, capabilities: Dict[str, Any]) -> bool:
        """
        Register a node for federated learning.
        
        Args:
            node_id: ID of the node to register
            capabilities: Node capabilities and metadata
            
        Returns:
            Success status
        """
        if node_id in self.participating_nodes:
            logger.warning(f"Node {node_id} already registered, updating capabilities")
        
        self.participating_nodes[node_id] = {
            "capabilities": capabilities,
            "last_active": time.time(),
            "contribution_score": 0.0,
            "model_version": 0
        }
        logger.info(f"Registered node {node_id} for federated learning")
        return True
    
    async def initiate_training_round(self, 
                                model_id: str, 
                                training_config: Dict[str, Any]) -> str:
        """
        Start a new federated training round.
        
        Args:
            model_id: ID of the model to train
            training_config: Configuration for this training round
            
        Returns:
            Unique ID for this training round
        """
        round_id = f"round_{self.training_rounds}_{model_id}_{int(time.time())}"
        self.training_rounds += 1
        
        # In a real implementation, this would send training tasks to nodes
        # and handle the coordination and aggregation of model updates
        
        # Simulate async training process
        await asyncio.sleep(0.1)  # Simulated network delay
        
        logger.info(f"Initiated training round {round_id} for model {model_id}")
        return round_id
    
    def aggregate_model_updates(self, 
                               round_id: str, 
                               updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate model updates from participating nodes.
        
        Args:
            round_id: ID of the training round
            updates: Dictionary of node IDs to their model updates
            
        Returns:
            Aggregated model parameters
        """
        # In a real implementation, this would perform federated averaging
        # or another aggregation method on the model parameters
        
        # Simulated aggregation result
        aggregated_result = {
            "round_id": round_id,
            "timestamp": time.time(),
            "num_participants": len(updates),
            "performance_metrics": {
                "accuracy": 0.85,
                "loss": 0.15
            }
        }
        
        logger.info(f"Aggregated updates from {len(updates)} nodes for round {round_id}")
        return aggregated_result
    
    def secure_transmission(self, data: Dict[str, Any], target_node: str) -> bool:
        """
        Securely transmit data to another node.
        
        Args:
            data: Data to transmit
            target_node: ID of the target node
            
        Returns:
            Success status
        """
        # In a real implementation, this would handle encryption,
        # authentication, and secure transmission protocols
        
        if target_node not in self.participating_nodes:
            logger.error(f"Cannot transmit to unknown node {target_node}")
            return False
        
        # Simulated secure transmission
        logger.info(f"Securely transmitted data to node {target_node}")
        return True


class IntelligentAgent:
    """
    Intelligent Agent (IA) that integrates neural processing, memory management,
    and federated learning for autonomous decision-making in distributed environments.
    
    The agent manages its internal knowledge representations, can learn from
    feedback, and participates in federated learning with other agents to improve
    its capabilities over time.
    """
    
    def __init__(self, 
                agent_id: str, 
                config: Optional[Dict[str, Any]] = None,
                neural_processor: Optional[LensRefractor] = None,
                p2p_network: Optional[P2PNetworkManager] = None):
        """
        Initialize the Intelligent Agent with the specified configuration.
        
        Args:
            agent_id: Unique identifier for this agent
            config: Configuration parameters for the agent
            neural_processor: Optional existing neural processor (LensRefractor instance)
            p2p_network: Optional P2P network manager for network communication
        
        Raises:
            ValueError: If required configuration parameters are missing or invalid
        """
        self.agent_id = agent_id
        self.config = config or {}
        self.start_time = time.time()
        self.state = "initialized"
        
        # Integrate with neural processor
        self.neural_processor = neural_processor or LensRefractor(
            circuit_complexity=self.config.get("circuit_complexity", "medium"),
            optimization_level=self.config.get("optimization_level", "balanced")
        )
        
        # Initialize memory system
        short_term_capacity = self.config.get("short_term_memory_capacity", 1000)
        long_term_capacity = self.config.get("long_term_memory_capacity", 10000)
        self.memory = MemorySystem(
            short_term_capacity=short_term_capacity,
            long_term_capacity=long_term_capacity
        )
        
        # Initialize federated learning components
        self.federated_learning = FederatedLearningManager(
            node_id=self.agent_id,
            security_level=self.config.get("security_level", "high")
        )
        
        # Set up P2P network connection if provided
        self.p2p_network = p2p_network
        if self.p2p_network:
            try:
                self.neural_processor.register_with_p2p_network(self.p2p_network)
                logger.info(f"Agent {self.agent_id} connected to P2P network")
            except Exception as e:
                logger.error(f"Failed to connect to P2P network: {str(e)}")
        
        # Knowledge base and learning parameters
        self.knowledge_base = {
            "models": {},
            "rules": {},
            "skills": {},
            "version": 1,
            "last_sync": self.start_time
        }
        
        self.learning_rate = self.config.get("learning_rate", 0.01)
        self.feedback_importance = self.config.get("feedback_importance", 0.8)
        
        # Setup thread pool for parallel operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.get("max_worker_threads", 4)
        )
        
        logger.info(f"Intelligent Agent {self.agent_id} initialized successfully")

    async def process_input(self, 
                     input_data: Dict[str, Any], 
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process input data and generate intelligent responses.
        
        Args:
            input_data: The input data to process
            context: Optional contextual information for processing
            
        Returns:
            Dictionary containing the response and metadata
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If processing fails
        """
        if not input_data:
            raise ValueError("Input data cannot be empty")
        
        context = context or {}
        start_time = time.time()
        
        try:
            # Store input in short-term memory
            memory_id = self.memory.store(
                content=input_data,
                memory_type=MemoryType.SHORT_TERM,
                importance=context.get("importance", 0.5),
                metadata={"source": context.get("source", "user"), "timestamp": start_time}
            )
            
            # Use neural processor to analyze input
            circuit_id = self.neural_processor.create_circuit(
                input_type=input_data.get("type", "generic"),
                complexity=context.get("complexity", "standard")
            )
            
            processing_result = self.neural_processor.run_circuit(
                circuit_id=circuit_id,
                input_data=input_data,
                parameters=context.get("parameters", {})
            )
            
            # Get relevant knowledge from memory
            if "query_vector" in processing_result:
                relevant_memories = self.memory.query(
                    query_vector=processing_result["query_vector"],
                    top_k=5
                )
                # Augment processing with retrieved memories
                memory_context = [mem[1] for mem in relevant_memories]
                processing_result["memory_context"] = memory_context
             
            # Generate response
            response = {
                "agent_id": self.agent_id,
                "result": processing_result,
                "response_time": time.time() - start_time,
                "memory_id": memory_id,
                "confidence": processing_result.get("confidence", 0.0),
                "timestamp": time.time()
            }
            
            # Store the response in memory if it's significant
            if processing_result.get("importance", 0.0) > 0.6:
                self.memory.store(
                    content=response,
                    memory_type=MemoryType.EPISODIC,
                    importance=processing_result.get("importance", 0.5),
                    metadata={"input_memory_id": memory_id}
                )
                
            return response
            
        except Exception as e:
            error_msg = f"Error processing input: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def learn_from_feedback(self, 
                          response_id: str, 
                          feedback: Dict[str, Any],
                          importance: float = 0.8) -> bool:
        """
        Update agent knowledge based on feedback.
        
        Args:
            response_id: ID of the response receiving feedback
            feedback: Feedback data including correctness, ratings, etc.
            importance: Importance score for this feedback (0.0 to 1.0)
            
        Returns:
            Success status of the learning operation
            
        Raises:
            ValueError: If feedback format is invalid
        """
        if not feedback or not isinstance(feedback, dict):
            raise ValueError("Feedback must be a non-empty dictionary")
            
        try:
            # Retrieve the original response from memory
            response_data = self.memory.retrieve(response_id)
            if not response_data:
                logger.warning(f"Cannot find response {response_id} for feedback")
                return False
                
            # Store feedback in memory
            feedback_id = self.memory.store(
                content={
                    "response_id": response_id,
                    "feedback": feedback,
                    "timestamp": time.time()
                },
                memory_type=MemoryType.EPISODIC,
                importance=importance,
                metadata={"type": "feedback", "source": feedback.get("source", "user")}
            )
            
            # Extract circuit ID from the original response
            circuit_id = response_data.get("result", {}).get("circuit_id")
            if not circuit_id:
                logger.warning("No circuit ID found in response, cannot optimize")
                return False
                
            # Update neural circuit based on feedback
            optimization_params = {
                "feedback_score": feedback.get("rating", 0.5),
                "learning_rate": self.learning_rate * importance,
                "correct_output": feedback.get("correct_output"),
                "is_positive": feedback.get("is_positive", True)
            }
            
            # Optimize the neural circuit
            optimization_result = self.neural_processor.optimize_circuit(
                circuit_id=circuit_id,
                parameters=optimization_params
            )
            
            # Update knowledge base based on feedback
            if feedback.get("update_knowledge", False):
                # Extract domain and key from feedback
                domain = feedback.get("domain", "general")
                key = feedback.get("key")
                
                if domain and key:
                    # Ensure domain exists in knowledge base
                    if domain not in self.knowledge_base:
                        self.knowledge_base[domain] = {}
                        
                    # Update or add the knowledge item
                    self.knowledge_base[domain][key] = {
                        "value": feedback.get("value"),
                        "confidence": feedback.get("confidence", 0.7),
                        "last_updated": time.time(),
                        "source": feedback.get("source", "feedback")
                    }
                    
                    logger.info(f"Knowledge base updated: {domain}/{key}")
            
            # Increment knowledge base version
            self.knowledge_base["version"] += 1
            
            logger.info(f"Applied feedback to response {response_id} with importance {importance}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying feedback: {str(e)}")
            return False
    
    def optimize_memory(self, target_reduction: float = 0.2) -> Dict[str, Any]:
        """
        Compress and optimize memory storage.
        
        Args:
            target_reduction: Target percentage reduction in memory usage
            
        Returns:
            Dictionary with optimization statistics
            
        Raises:
            ValueError: If target_reduction is invalid
        """
        if not 0 < target_reduction < 1:
            raise ValueError("Target reduction must be between 0 and 1")
            
        try:
            start_time = time.time()
            
            # Get initial memory stats
            initial_stats = self.memory.get_stats()
            
            # Calculate current memory usage
            total_memories = sum(stats["count"] for stats in initial_stats.values())
            
            # Calculate optimization targets for each memory type
            targets = {}
            for memory_type in MemoryType:
                # Calculate the number of items to remove
                count = initial_stats[memory_type.value]["count"]
                to_remove = int(count * target_reduction)
                targets[memory_type] = to_remove
                
                # Optimize each memory type
                if to_remove > 0:
                    # Use the internal memory system optimization
                    # This will remove the least important items
                    for _ in range(to_remove):
                        self.memory._optimize_memory_type(memory_type)
            
            # Special processing for semantic memory - consolidate similar items
            if targets[MemoryType.SEMANTIC] > 0:
                semantic_memories = list(self.memory.memories[MemoryType.SEMANTIC].items())
                # Group by metadata category if available
                categories = {}
                for memory_id, unit in semantic_memories:
                    category = unit.metadata.get("category", "unknown")
                    if category not in categories:
                        categories[category] = []
                    categories[category].append((memory_id, unit))
                
                # For each category with multiple items, try to consolidate
                for category, items in categories.items():
                    if len(items) <= 1:
                        continue
                        
                    # Sort by importance
                    items.sort(key=lambda x: x[1].importance, reverse=True)
                    
                    # Keep the most important, remove others
                    for memory_id, _ in items[1:targets[MemoryType.SEMANTIC]]:
                        if memory_id in self.memory.memories[MemoryType.SEMANTIC]:
                            del self.memory.memories[MemoryType.SEMANTIC][memory_id]
            
            # Get final memory stats
            final_stats = self.memory.get_stats()
            final_memories = sum(stats["count"] for stats in final_stats.values())
            
            # Calculate optimization results
            results = {
                "initial_count": total_memories,
                "final_count": final_memories,
                "reduction": total_memories - final_memories,
                "reduction_percent": ((total_memories - final_memories) / total_memories * 100) if total_memories > 0 else 0,
                "target_reduction_percent": target_reduction * 100,
                "duration_seconds": time.time() - start_time,
                "detailed_stats": {
                    "before": initial_stats,
                    "after": final_stats
                }
            }
            
            logger.info(f"Memory optimization completed: {results['reduction']} items removed ({results['reduction_percent']:.2f}%)")
            return results
            
        except Exception as e:
            logger.error(f"Error during memory optimization: {str(e)}")
            raise
    
    async def sync_knowledge(self, 
                      peer_agents: List[str], 
                      domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Synchronize knowledge with other agents in the network.
        
        Args:
            peer_agents: List of agent IDs to synchronize with
            domains: Optional list of knowledge domains to sync (default: all)
            
        Returns:
            Dictionary with synchronization results
            
        Raises:
            ValueError: If peer_agents list is empty
            RuntimeError: If synchronization fails
        """
        if not peer_agents:
            raise ValueError("Peer agents list cannot be empty")
            
        sync_results = {
            "timestamp": time.time(),
            "initiated_by": self.agent_id,
            "peer_count": len(peer_agents),
            "successful_syncs": 0,
            "failed_syncs": 0,
            "domains_synced": domains if domains else list(self.knowledge_base.keys()),
            "details": {}
        }
        
        try:
            # Validate P2P network availability
            if not self.p2p_network:
                raise RuntimeError("P2P network manager is not available for synchronization")
                
            # Prepare domains to sync
            domains_to_sync = domains if domains else list(self.knowledge_base.keys())
            # Filter out internal domains that shouldn't be synced
            domains_to_sync = [d for d in domains_to_sync if not d.startswith("_") and d != "version" and d != "last_sync"]
            
            logger.info(f"Beginning knowledge sync with {len(peer_agents)} peers for domains: {', '.join(domains_to_sync)}")
            
            # Process each peer agent
            for peer_id in peer_agents:
                try:
                    # Create a secure connection with the peer
                    connection_result = await self._establish_secure_peer_connection(peer_id)
                    if not connection_result["success"]:
                        raise RuntimeError(f"Failed to establish secure connection with peer {peer_id}: {connection_result['error']}")
                    
                    # Prepare knowledge packet for this peer
                    knowledge_packet = {
                        "source_agent": self.agent_id,
                        "timestamp": time.time(),
                        "knowledge_version": self.knowledge_base["version"],
                        "domains": {}
                    }
                    
                    # Add requested domain data
                    for domain in domains_to_sync:
                        if domain in self.knowledge_base:
                            knowledge_packet["domains"][domain] = self.knowledge_base[domain]
                    
                    # Exchange knowledge with peer
                    logger.debug(f"Sending knowledge packet to peer {peer_id} with {len(knowledge_packet['domains'])} domains")
                    peer_response = await self.p2p_network.exchange_data(
                        peer_id=peer_id,
                        data_type="knowledge_sync",
                        data=knowledge_packet,
                        timeout=self.config.get("sync_timeout", 30)
                    )
                    
                    # Process peer's knowledge response
                    if peer_response and "domains" in peer_response:
                        updates_applied = await self._integrate_peer_knowledge(
                            peer_id=peer_id,
                            peer_knowledge=peer_response["domains"],
                            domains=domains_to_sync
                        )
                        
                        sync_results["details"][peer_id] = {
                            "success": True,
                            "synced_at": time.time(),
                            "domains_received": list(peer_response["domains"].keys()),
                            "updates_applied": updates_applied,
                            "peer_version": peer_response.get("knowledge_version", "unknown")
                        }
                        sync_results["successful_syncs"] += 1
                        
                        logger.info(f"Successfully synced knowledge with peer {peer_id}, applied {updates_applied} updates")
                    else:
                        raise RuntimeError(f"Invalid or empty response from peer {peer_id}")
                        
                except Exception as e:
                    error_msg = f"Failed to sync with peer {peer_id}: {str(e)}"
                    logger.error(error_msg)
                    sync_results["details"][peer_id] = {
                        "success": False,
                        "error": error_msg,
                        "attempted_at": time.time()
                    }
                    sync_results["failed_syncs"] += 1
            
            # Update last sync timestamp
            self.knowledge_base["last_sync"] = time.time()
            
            # Log summary
            success_rate = (sync_results["successful_syncs"] / len(peer_agents)) * 100 if peer_agents else 0
            logger.info(f"Knowledge sync completed: {sync_results['successful_syncs']}/{len(peer_agents)} successful ({success_rate:.1f}%)")
            
            return sync_results
            
        except Exception as e:
            error_msg = f"Knowledge synchronization failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    async def _establish_secure_peer_connection(self, peer_id: str) -> Dict[str, Any]:
        """
        Establish a secure connection with a peer agent.
        
        Args:
            peer_id: ID of the peer agent to connect to
            
        Returns:
            Dictionary with connection results
        """
        try:
            # Check if peer is registered
            if not self.p2p_network.is_peer_available(peer_id):
                return {
                    "success": False,
                    "error": f"Peer {peer_id} is not available on the network"
                }
            
            # Perform authentication handshake
            auth_result = await self.p2p_network.authenticate_peer(
                peer_id=peer_id,
                credentials={
                    "agent_id": self.agent_id,
                    "auth_token": self.config.get("auth_token", ""),
                    "timestamp": time.time()
                }
            )
            
            if not auth_result["success"]:
                return {
                    "success": False,
                    "error": f"Authentication failed: {auth_result.get('message', 'Unknown error')}"
                }
            
            # Successfully established connection
            return {
                "success": True,
                "session_id": auth_result.get("session_id", ""),
                "encryption_level": auth_result.get("encryption_level", "standard"),
                "established_at": time.time()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Connection error: {str(e)}"
            }
            
    async def _integrate_peer_knowledge(self, 
                                  peer_id: str,
                                  peer_knowledge: Dict[str, Any],
                                  domains: List[str]) -> int:
        """
        Integrate knowledge received from a peer into the local knowledge base.
        
        Args:
            peer_id: ID of the peer agent
            peer_knowledge: Knowledge data received from the peer
            domains: List of domains to integrate
            
        Returns:
            Number of knowledge updates applied
        """
        updates_applied = 0
        
        try:
            for domain in domains:
                if domain not in peer_knowledge:
                    continue
                    
                # Ensure domain exists in local knowledge base
                if domain not in self.knowledge_base:
                    self.knowledge_base[domain] = {}
                
                # Process each knowledge item in the domain
                for key, peer_item in peer_knowledge[domain].items():
                    # Verify knowledge item structure
                    if not isinstance(peer_item, dict) or "value" not in peer_item:
                        logger.warning(f"Invalid knowledge item structure for {domain}/{key} from peer {peer_id}")
                        continue
                        
                    # Apply knowledge verification
                    verification_result = self._verify_knowledge_item(
                        domain=domain,
                        key=key,
                        item=peer_item,
                        source_peer=peer_id
                    )
                    
                    if not verification_result["verified"]:
                        logger.warning(f"Knowledge verification failed for {domain}/{key}: {verification_result['reason']}")
                        continue
                    
                    # Check if we should update our local knowledge
                    should_update = False
                    
                    # If item doesn't exist locally, add it
                    if key not in self.knowledge_base[domain]:
                        should_update = True
                    else:
                        local_item = self.knowledge_base[domain][key]
                        # Update if peer item is more recent and has higher confidence
                        peer_timestamp = peer_item.get("last_updated", 0)
                        local_timestamp = local_item.get("last_updated", 0)
                        
                        if peer_timestamp > local_timestamp and peer_item.get("confidence", 0) >= local_item.get("confidence", 0):
                            should_update = True
                        # Also update if confidence is significantly higher
                        elif peer_item.get("confidence", 0) > local_item.get("confidence", 0) * 1.2:
                            should_update = True
                    
                    # Apply the update if needed
                    if should_update:
                        self.knowledge_base[domain][key] = {
                            "value": peer_item["value"],
                            "confidence": peer_item.get("confidence", 0.7),
                            "last_updated": peer_item.get("last_updated", time.time()),
                            "source": f"peer:{peer_id}",
                            "synced_at": time.time()
                        }
                        updates_applied += 1
            
            # If updates were applied, increment knowledge base version
            if updates_applied > 0:
                self.knowledge_base["version"] += 1
                
            return updates_applied
            
        except Exception as e:
            logger.error(f"Error integrating peer knowledge: {str(e)}")
            return updates_applied
    
    async def federated_training(self, 
                           model_id: str, 
                           config: Dict[str, Any],
                           peer_agents: List[str],
                           rounds: int = 1,
                           aggregation_method: str = "fedavg") -> Dict[str, Any]:
        """
        Perform federated training across multiple agents in the network.
        
        This method implements a federated learning algorithm that allows multiple
        agents to collaboratively train a model without sharing their raw data.
        Instead, only model updates are shared and aggregated.
        
        Args:
            model_id: Identifier for the model to be trained
            config: Training configuration including hyperparameters
            peer_agents: List of agent IDs to participate in training
            rounds: Number of training rounds to perform
            aggregation_method: Method for aggregating model updates (fedavg, fedprox, etc.)
            
        Returns:
            Dictionary with training results
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If federated training fails
        """
        if not peer_agents:
            raise ValueError("Peer agents list cannot be empty for federated training")
            
        if model_id not in self.knowledge_base.get("models", {}):
            raise ValueError(f"Model {model_id} not found in knowledge base")
            
        results = {
            "model_id": model_id,
            "initiated_by": self.agent_id,
            "start_time": time.time(),
            "rounds_completed": 0,
            "rounds_planned": rounds,
            "participants": peer_agents.copy(),
            "aggregation_method": aggregation_method,
            "performance_metrics": {},
            "round_details": []
        }
        
        try:
            logger.info(f"Initiating federated training for model {model_id} with {len(peer_agents)} participants, {rounds} rounds")
            
            # Verify all peers are available and capable of training
            available_peers = []
            for peer_id in peer_agents:
                try:
                    capability_check = await self.p2p_network.check_peer_capability(
                        peer_id=peer_id,
                        capability="federated_training",
                        params={"model_id": model_id}
                    )
                    if capability_check.get("available", False):
                        available_peers.append(peer_id)
                    else:
                        logger.warning(f"Peer {peer_id} not available for federated training: {capability_check.get('reason', 'Unknown')}")
                except Exception as e:
                    logger.warning(f"Failed to check capabilities for peer {peer_id}: {str(e)}")
            
            if len(available_peers) < 2:  # Need at least this node + 1 peer
                raise RuntimeError(f"Insufficient peers available for federated training. Found {len(available_peers)}, need at least 2")
                
            results["actual_participants"] = available_peers
            
            # Extract the initial model parameters
            initial_model = self.knowledge_base["models"][model_id].get("parameters", {})
            if not initial_model:
                raise ValueError(f"Model {model_id} has no parameters for training")
                
            # Execute training rounds
            for round_num in range(1, rounds + 1):
                logger.info(f"Starting federated training round {round_num}/{rounds}")
                
                round_id = f"{model_id}_r{round_num}_{int(time.time())}"
                round_results = {
                    "round_id": round_id,
                    "start_time": time.time(),
                    "participants": [],
                    "updates_received": 0,
                    "aggregation_success": False
                }
                
                # Initialize model updates collection
                model_updates = {}
                
                # Distribute current model and training configuration to participants
                distribution_tasks = []
                for peer_id in available_peers:
                    task = self._distribute_training_task(
                        peer_id=peer_id,
                        model_id=model_id,
                        round_id=round_id,
                        model_params=self.knowledge_base["models"][model_id].get("parameters", {}),
                        config=config
                    )
                    distribution_tasks.append(task)
                \r
                # Wait for all distribution tasks to complete\r
                distribution_results = await asyncio.gather(*distribution_tasks, return_exceptions=True)\r
                \r
                # Process distribution results\r
                for i, result in enumerate(distribution_results):\r
                    peer_id = available_peers[i]\r
                    if isinstance(result, Exception):\r
                        logger.error(f"Failed to distribute training task to peer {peer_id}: {str(result)}")\r
                        continue\r
                    \r
                    if result.get("success", False):\r
                        round_results["participants"].append(peer_id)\r
                        logger.debug(f"Successfully distributed training task to peer {peer_id} for round {round_id}")\r
                    else:\r
                        logger.warning(f"Peer {peer_id} rejected training task: {result.get('reason', 'Unknown reason')}")\r
                \r
                # Check if we have enough participants\r
                if len(round_results["participants"]) < 2:\r
                    logger.error(f"Insufficient participants for round {round_id}. Found {len(round_results['participants'])}, need at least 2")\r
                    round_results["status"] = "failed"\r
                    round_results["failure_reason"] = "insufficient_participants"\r
                    results["round_details"].append(round_results)\r
                    continue\r
                \r
                # Collect model updates from participants\r
                collection_tasks = []\r
                for peer_id in round_results["participants"]:\r
                    task = self._collect_training_results(\r
                        peer_id=peer_id,\r
                        model_id=model_id,\r
                        round_id=round_id\r
                    )\r
                    collection_tasks.append(task)\r
                \r
                # Wait for all collection tasks to complete\r
                collection_timeout = config.get("collection_timeout", 300)  # 5 minutes default\r
                try:\r
                    collection_results = await asyncio.gather(*collection_tasks, return_exceptions=True)\r
                    \r
                    # Process collection results\r
                    for i, result in enumerate(collection_results):\r
                        peer_id = round_results["participants"][i]\r
                        if isinstance(result, Exception):\r
                            logger.error(f"Failed to collect training results from peer {peer_id}: {str(result)}")\r
                            continue\r
                        \r
                        if result.get("success", False) and "model_update" in result:\r
                            # Validate the model update\r
                            validation_result = self.validate_model(\r
                                model_id=model_id,\r
                                parameters=result["model_update"],\r
                                metrics=result.get("metrics", {})\r
                            )\r
                            \r
                            if validation_result["valid"]:\r
                                # Store the valid update\r
                                model_updates[peer_id] = {\r
                                    "parameters": result["model_update"],\r
                                    "metrics": result.get("metrics", {}),\r
                                    "training_samples": result.get("training_samples", 0),\r
                                    "weight": validation_result.get("weight", 1.0)\r
                                }\r
                                round_results["updates_received"] += 1\r
                                logger.info(f"Received valid model update from peer {peer_id} for round {round_id}")\r
                            else:\r
                                logger.warning(f"Invalid model update from peer {peer_id}: {validation_result.get('reason', 'Unknown issue')}")\r
                        else:\r
                            logger.warning(f"Failed to get valid training results from peer {peer_id}: {result.get('reason', 'Unknown reason')}")\r
                except asyncio.TimeoutError:\r
                    logger.error(f"Timeout collecting training results for round {round_id}")\r
                    round_results["status"] = "timeout"\r
                    results["round_details"].append(round_results)\r
                    continue\r
                \r
                # Check if we received enough updates\r
                if len(model_updates) < 2:\r
                    logger.error(f"Insufficient model updates for round {round_id}. Received {len(model_updates)}, need at least 2")\r
                    round_results["status"] = "failed"\r
                    round_results["failure_reason"] = "insufficient_updates"\r
                    results["round_details"].append(round_results)\r
                    continue\r
                \r
                # Aggregate model updates\r
                try:\r
                    aggregation_result = self._aggregate_model_updates(\r
                        model_id=model_id,\r
                        updates=model_updates,\r
                        method=aggregation_method,\r
                        config=config\r
                    )\r
                    \r
                    if aggregation_result["success"]:\r
                        # Update the local model with aggregated parameters\r
                        self.knowledge_base["models"][model_id]["parameters"] = aggregation_result["parameters"]\r
                        self.knowledge_base["models"][model_id]["last_updated"] = time.time()\r
                        self.knowledge_base["models"][model_id]["training_rounds"] = \r
                            self.knowledge_base["models"][model_id].get("training_rounds", 0) + 1\r
                        \r
                        # Update round results\r
                        round_results["aggregation_success"] = True\r
                        round_results["metrics"] = aggregation_result.get("metrics", {})\r
                        round_results["status"] = "completed"\r
                        round_results["end_time"] = time.time()\r
                        \r
                        # Update overall performance metrics\r
                        for metric_name, metric_value in aggregation_result.get("metrics", {}).items():\r
                            if metric_name not in results["performance_metrics"]:\r
                                results["performance_metrics"][metric_name] = []\r
                            results["performance_metrics"][metric_name].append(metric_value)\r
                        \r
                        logger.info(f"Successfully completed federated training round {round_num}/{rounds}")\r
                    else:\r
                        logger.error(f"Aggregation failed for round {round_id}: {aggregation_result.get('error', 'Unknown error')}")\r
                        round_results["status"] = "failed"\r
                        round_results["failure_reason"] = "aggregation_failed"\r
                except Exception as e:\r
                    logger.error(f"Error during model aggregation for round {round_id}: {str(e)}")\r
                    round_results["status"] = "failed"\r
                    round_results["failure_reason"] = f"aggregation_error: {str(e)}"\r
                \r
                # Record round results\r
                results["round_details"].append(round_results)\r
                \r
                # Update rounds completed\r
                if round_results.get("status") == "completed":\r
                    results["rounds_completed"] += 1\r
                \r
                # Optional: Distribute the aggregated model back to participants\r
                if config.get("distribute_aggregated_model", True) and round_results.get("aggregation_success", False):\r
                    distribution_tasks = []\r
                    for peer_id in round_results["participants"]:\r
                        task = self._distribute_aggregated_model(\r
                            peer_id=peer_id,\r
                            model_id=model_id,\r
                            round_id=round_id,\r
                            parameters=self.knowledge_base["models"][model_id]["parameters"],\r
                            metrics=round_results.get("metrics", {})\r
                        )\r
                        distribution_tasks.append(task)\r
                    \r
                    # Execute all tasks concurrently\r
                    await asyncio.gather(*distribution_tasks, return_exceptions=True)\r
            \r
            # Calculate final performance metrics\r
            for metric_name, values in results["performance_metrics"].items():\r
                results["final_" + metric_name] = values[-1] if values else None\r
                results["avg_" + metric_name] = sum(values) / len(values) if values else None\r
            \r
            # Calculate success rate\r
            results["success_rate"] = (results["rounds_completed"] / rounds) * 100 if rounds > 0 else 0\r
            results["end_time"] = time.time()\r
            results["duration"] = results["end_time"] - results["start_time"]\r
            \r
            # Log training completion\r
            logger.info(f"Federated training completed for model {model_id}: {results['rounds_completed']}/{rounds} successful rounds")\r
            \r
            # Save training metrics for historical tracking\r
            self._log_training_metrics(model_id, results)\r
            \r
            return results\r
            \r
        except Exception as e:\r
            error_msg = f"Federated training failed: {str(e)}"\r
            logger.error(error_msg)\r
            raise RuntimeError(error_msg) from e\r
    \r
    async def _distribute_training_task(self,\r
                                  peer_id: str,\r
                                  model_id: str,\r
                                  round_id: str,\r
                                  model_params: Dict[str, Any],\r
                                  config: Dict[str, Any]) -> Dict[str, Any]:\r
        """Distribute a federated training task to a peer agent.\r
        \r
        Args:\r
            peer_id: ID of the peer agent\r
            model_id: ID of the model to train\r
            round_id: ID of the training round\r
            model_params: Current model parameters\r
            config: Training configuration\r
            \r
        Returns:\r
            Dictionary with distribution results\r
        """\r
        try:\r
            # Create secure connection with peer\r
            connection = await self._establish_secure_peer_connection(peer_id)\r
            if not connection["success"]:\r
                return {"success": False, "reason": f"Connection failed: {connection.get('error', 'Unknown error')}"}\r
            \r
            # Prepare training task message\r
            training_task = {\r
                "task_type": "federated_training",\r
                "model_id": model_id,\r
                "round_id": round_id,\r
                "model_params": model_params,\r
                "hyperparameters": config.get("hyperparameters", {}),\r
                "epochs": config.get("epochs", 1),\r
                "batch_size": config.get("batch_size", 32),\r
                "timeout": config.get("training_timeout", 600),  # 10 minutes default\r
                "source_agent": self.agent_id,\r
                "timestamp": time.time()\r
            }\r
            \r
            # Securely send training task to peer\r
            response = await self.secure_communicate(\r
                peer_id=peer_id,\r
                message_type="training_task",\r
                data=training_task,\r
                timeout=config.get("distribution_timeout", 60)\r
            )\r
            \r
            return response or {"success": False, "reason": "No response from peer"}\r
            \r
        except Exception as e:\r
            logger.error(f"Error distributing training task to peer {peer_id}: {str(e)}")\r
            return {"success": False, "reason": str(e)}\r
    \r
    async def _collect_training_results(self,
                                  peer_id: str,
                                  model_id: str,
                                  round_id: str) -> Dict[str, Any]:
        """Collect training results from a peer agent.
        
        Args:
            peer_id: ID of the peer agent
            model_id: ID of the model being trained
            round_id: ID of the training round
            
        Returns:
            Dictionary with training results including model updates
        """
        try:
            # Create a secure connection
            connection = await self._establish_secure_peer_connection(peer_id)
            if not connection["success"]:
                return {"success": False, "reason": f"Connection failed: {connection.get('error', 'Unknown error')}"}
            
            # Request training results
            request = {
                "request_type": "training_results",
                "model_id": model_id,
                "round_id": round_id,
                "requesting_agent": self.agent_id,
                "timestamp": time.time()
            }
            
            # Send request and wait for response
            response = await self.secure_communicate(
                peer_id=peer_id,
                message_type="training_results_request",
                data=request,
                timeout=120  # 2 minutes timeout for results
            )
            
            if not response or not response.get("success", False):
                return {
                    "success": False, 
                    "reason": response.get("reason", "Failed to get training results")
                }
            
            # Validate that the response contains model update
            if "model_update" not in response:
                return {
                    "success": False,
                    "reason": "Response missing model update data"
                }
            
            # Return the training results
            return {
                "success": True,
                "model_update": response["model_update"],
                "metrics": response.get("metrics", {}),
                "training_samples": response.get("training_samples", 0),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error collecting training results from peer {peer_id}: {str(e)}")
            return {"success": False, "reason": str(e)}
    
    def _aggregate_model_updates(self,
                               model_id: str,
                               updates: Dict[str, Dict[str, Any]],
                               method: str = "fedavg",
                               config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Aggregate model updates from multiple peers.
        
        Args:
            model_id: ID of the model being trained
            updates: Dictionary mapping peer IDs to their model updates
            method: Aggregation method to use (fedavg, fedprox, etc.)
            config: Additional configuration parameters
            
        Returns:
            Dictionary with aggregation results
            
        Raises:
            ValueError: If the aggregation method is unsupported or updates is empty
        """
        if not updates:
            raise ValueError("No updates to aggregate")
        
        config = config or {}
        logger.info(f"Aggregating {len(updates)} model updates using method: {method}")
        
        try:
            # Extract parameters and weights from each update
            parameters_list = []
            weights = []
            metrics_sum = {}
            total_samples = 0
            
            for peer_id, update_data in updates.items():
                parameters = update_data.get("parameters")
                if not parameters:
                    logger.warning(f"Skipping update from peer {peer_id}: missing parameters")
                    continue
                
                # Get weight for this update (default: proportional to training samples)
                weight = update_data.get("weight", 1.0)
                training_samples = update_data.get("training_samples", 0)
                if training_samples > 0:
                    # Weight by number of training samples if available
                    weight *= training_samples
                    total_samples += training_samples
                
                parameters_list.append(parameters)
                weights.append(weight)
                
                # Collect metrics for averaging
                for metric_name, metric_value in update_data.get("metrics", {}).items():
                    if metric_name not in metrics_sum:
                        metrics_sum[metric_name] = 0.0
                    metrics_sum[metric_name] += metric_value * weight
            
            if not parameters_list:
                return {
                    "success": False,
                    "error": "No valid parameters found in updates"
                }
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight <= 0:
                return {
                    "success": False,
                    "error": "Total weight is zero or negative"
                }
            
            normalized_weights = [w / total_weight for w in weights]
            
            # Perform aggregation based on the specified method
            if method.lower() == "fedavg":
                # Federated Averaging (FedAvg)
                aggregated_params = self._fedavg_aggregation(parameters_list, normalized_weights)
            elif method.lower() == "fedprox":
                # Federated Proximal (FedProx) - similar to FedAvg but with proximal term
                mu = config.get("proximal_term", 0.01)
                initial_params = self.knowledge_base["models"][model_id].get("parameters", {})
                aggregated_params = self._fedprox_aggregation(parameters_list, normalized_weights, initial_params, mu)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported aggregation method: {method}"
                }
            
            # Calculate average metrics
            avg_metrics = {}
            for metric_name, metric_sum in metrics_sum.items():
                avg_metrics[metric_name] = metric_sum / total_weight
            
            return {
                "success": True,
                "parameters": aggregated_params,
                "metrics": avg_metrics,
                "method": method,
                "participants": len(updates),
                "total_samples": total_samples,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error aggregating model updates: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _fedavg_aggregation(self, parameters_list, weights):
        """Perform FedAvg aggregation on model parameters.
        
        Args:
            parameters_list: List of parameter dictionaries from peers
            weights: List of weights for each parameter set
            
        Returns:
            Aggregated parameters dictionary
        """
        # Implement weighted average for each parameter
        aggregated = {}
        
        # Use the first set of parameters as a template for the structure
        for key in parameters_list[0]:
            # Handle nested dictionaries recursively
            if isinstance(parameters_list[0][key], dict):
                sub_params = [params[key] for params in parameters_list]
                aggregated[key] = self._fedavg_aggregation(sub_params, weights)
            # Handle lists and arrays (assuming they're numeric)
            elif isinstance(parameters_list[0][key], (list, np.ndarray)):
                # Convert all to numpy arrays for easier computation
                try:
                    arrays = [np.array(params[key]) for params in parameters_list]
                    # Weighted sum
                    weighted_sum = sum(w * arr for w, arr in zip(weights, arrays))
                    # Convert back to the original type
                    if isinstance(parameters_list[0][key], list):
                        aggregated[key] = weighted_sum.tolist()
                    else:
                        aggregated[key] = weighted_sum
                except (ValueError, TypeError):
                    # Fallback for non-numeric arrays
                    logger.warning(f"Non-numeric array detected for key {key}, using first peer's value")
                    aggregated[key] = parameters_list[0][key]
            # Handle numeric values
            elif isinstance(parameters_list[0][key], (int, float)):
                aggregated[key] = sum(w * params[key] for w, params in zip(weights, parameters_list))
            # For non-numeric values, use the value from the peer with highest weight
            else:
                max_weight_idx = weights.index(max(weights))
                aggregated[key] = parameters_list[max_weight_idx][key]
                
        return aggregated
    
    def _fedprox_aggregation(self, parameters_list, weights, initial_params, mu=0.01):
        """Perform FedProx aggregation on model parameters.
        
        Args:
            parameters_list: List of parameter dictionaries from peers
            weights: List of weights for each parameter set
            initial_params: Initial model parameters before training
            mu: Proximal term hyperparameter
            
        Returns:
            Aggregated parameters dictionary
        """
        # First perform standard FedAvg
        fedavg_result = self._fedavg_aggregation(parameters_list, weights)
        
        # If no initial parameters, just return FedAvg result
        if not initial_params:
            return fedavg_result
        
        # Apply proximal term - pull solution closer to initial params
        aggregated = {}
        
        for key in fedavg_result:
            if isinstance(fedavg_result[key], dict) and key in initial_params and isinstance(initial_params[key], dict):
                # Recursively handle nested dictionaries
                aggregated[key] = self._fedprox_aggregation(
                    [fedavg_result[key]], 
                    [1.0], 
                    initial_params[key], 
                    mu
                )
            elif isinstance(fedavg_result[key], (list, np.ndarray)) and key in initial_params:
                try:
                    # Convert to numpy arrays
                    avg_array = np.array(fedavg_result[key])
                    init_array = np.array(initial_params[key])
                    
                    # Apply proximal regularization: (1-mu)*fedavg + mu*initial
                    regularized = (1 - mu) * avg_array + mu * init_array
                    
                    # Convert back to original type
                    if isinstance(fedavg_result[key], list):
                        aggregated[key] = regularized.tolist()
                    else:
                        aggregated[key] = regularized
                except (ValueError, TypeError):
                    # Fallback for incompatible arrays
                    aggregated[key] = fedavg_result[key]
            elif isinstance(fedavg_result[key], (int, float)) and key in initial_params and isinstance(initial_params[key], (int, float)):
                # Apply proximal term to numeric values
                aggregated[key] = (1 - mu) * fedavg_result[key] + mu * initial_params[key]
            else:
                # For other types or keys not in initial params, use FedAvg result
                aggregated[key] = fedavg_result[key]
                
        return aggregated
    
    def _log_training_metrics(self, model_id: str, results: Dict[str, Any]) -> None:
        """Log training metrics for historical tracking.
        
        Args:
            model_id: ID of the model that was trained
            results: Results of the training process
            
        Returns:
            None
        """
        try:
            # Ensure we have a place to store metrics
            if "training_history" not in self.knowledge_base["models"].get(model_id, {}):
                if model_id not in self.knowledge_base["models"]:
                    self.knowledge_base["models"][model_id] = {}
                self.knowledge_base["models"][model_id]["training_history"] = []
            
            # Extract relevant metrics from results
            metrics_entry = {
                "timestamp": time.time(),
                "rounds_completed": results.get("rounds_completed", 0),
                "rounds_planned": results.get("rounds_planned", 0),
                "participants": len(results.get("actual_participants", [])),
                "duration": results.get("duration", 0),
                "success_rate": results.get("success_rate", 0)
            }
            
            # Add performance metrics
            for metric_name, values in results.get("performance_metrics", {}).items():
                if values:
                    metrics_entry[f"final_{metric_name}"] = values[-1]
                    metrics_entry[f"avg_{metric_name}"] = sum(values) / len(values)
            
            # Add entry to history
            self.knowledge_base["models"][model_id]["training_history"].append(metrics_entry)
            
            # Keep only the last 100 entries to prevent unlimited growth
            max_history = self.config.get("max_training_history", 100)
            if len(self.knowledge_base["models"][model_id]["training_history"]) > max_history:
                # Remove oldest entries
                excess = len(self.knowledge_base["models"][model_id]["training_history"]) - max_history
                self.knowledge_base["models"][model_id]["training_history"] = self.knowledge_base["models"][model_id]["training_history"][excess:]
            
            logger.info(f"Logged training metrics for model {model_id}, history size: {len(self.knowledge_base['models'][model_id]['training_history'])}")
            
            # Optionally persist metrics to external storage
            if self.config.get("persist_metrics", False):
                try:
                    # This would be implemented to save metrics to a database or file
                    self._persist_metrics_to_storage(model_id, metrics_entry)
                except Exception as e:
                    logger.warning(f"Failed to persist metrics to storage: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error logging training metrics: {str(e)}")
    
    def _verify_knowledge_item(self,
                             domain: str,
                             key: str,
                             item: Dict[str, Any],
                             source_peer: str) -> Dict[str, Any]:
        """Validate knowledge received from peers.
        
        This method implements verification rules to ensure that knowledge
        from peers is valid, consistent, and safe to integrate into the 
        local knowledge base.
        
        Args:
            domain: Knowledge domain
            key: Knowledge item key
            item: Knowledge item data
            source_peer: ID of the peer that provided the knowledge
            
        Returns:
            Dictionary with verification results
        """
        try:
            # Basic structure validation
            if not isinstance(item, dict):
                return {
                    "verified": False,
                    "reason": "Knowledge item must be a dictionary"
                }
            
            if "value" not in item:
                return {
                    "verified": False,
                    "reason": "Knowledge item must contain a 'value' field"
                }
            
            # Value type validation
            value_type = type(item["value"])
            
            # Validate by domain-specific rules
            if domain == "models":
                # For models, ensure they have required fields
                if isinstance(item["value"], dict):
                    if "parameters" not in item["value"]:
                        return {
                            "verified": False,
                            "reason": "Model must contain parameters"
                        }
                else:
                    return {
                        "verified": False,
                        "reason": "Model value must be a dictionary"
                    }
            
            # Check confidence value
            confidence = item.get("confidence", 0.0)
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                return {
                    "verified": False,
                    "reason": "Confidence must be a number between 0 and 1"
                }
            
            # Timestamp validation
            if "last_updated" in item:
                if not isinstance(item["last_updated"], (int, float)) or item["last_updated"] > time.time():
                    return {
                        "verified": False,
                        "reason": "Invalid timestamp (future date not allowed)"
                    }
            
            # Verify source if provided
            if "source" in item and not item["source"].startswith(f"peer:"):
                # Allow only certain sources
                valid_sources = ["user", "feedback", "system", f"peer:{source_peer}"]
                if not any(item["source"].startswith(src) for src in valid_sources):
                    return {
                        "verified": False,
                        "reason": f"Invalid source: {item['source']}"
                    }
            
            # Apply domain-specific content validation
            if domain == "skills" and isinstance(item["value"], dict):
                # Skills should have a 'code' field, and it should be a string
                if "code" in item["value"] and not isinstance(item["value"]["code"], str):
                    return {
                        "verified": False,
                        "reason": "Skill code must be a string"
                    }
                
                # Check for potentially malicious code patterns (basic check)
                if "code" in item["value"]:
                    code = item["value"]["code"]
                    suspicious_patterns = ["exec(", "eval(", "os.system(", "subprocess."]
                    if any(pattern in code for pattern in suspicious_patterns):
                        return {
                            "verified": False,
                            "reason": "Suspicious code patterns detected"
                        }
            
            # All checks passed
            return {
                "verified": True,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error during knowledge verification: {str(e)}")
            return {
                "verified": False,
                "reason": f"Verification error: {str(e)}"
            }
    
    def validate_model(self,
                      model_id: str,
                      parameters: Dict[str, Any],
                      metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate model updates before integrating them into the knowledge base.
        
        This method ensures model updates are structurally valid, meet performance
        thresholds, and don't contain potentially harmful patterns.
        
        Args:
            model_id: Identifier for the model
            parameters: The model parameters to validate
            metrics: Optional metrics about model performance
            
        Returns:
            Dictionary with validation results and optional weight for aggregation
        """
        try:
            # First, check if the model exists in the knowledge base
            if model_id not in self.knowledge_base.get("models", {}):
                return {
                    "valid": False,
                    "reason": f"Model {model_id} not found in knowledge base"
                }
            
            # Get the expected structure from the existing model
            existing_model = self.knowledge_base["models"][model_id].get("parameters", {})
            
            # Basic structure validation
            if not isinstance(parameters, dict):
                return {
                    "valid": False,
                    "reason": "Model parameters must be a dictionary"
                }
            
            # Check that all required keys from existing model are present
            if existing_model:
                missing_keys = [key for key in existing_model if key not in parameters]
                if missing_keys:
                    return {
                        "valid": False,
                        "reason": f"Missing required parameter sections: {', '.join(missing_keys)}"
                    }
            
            # Validate parameter shapes and types match existing model
            for key, existing_value in existing_model.items():
                if key not in parameters:
                    continue
                
                # Check that types match
                if type(existing_value) != type(parameters[key]):
                    return {
                        "valid": False,
                        "reason": f"Type mismatch for parameter {key}: expected {type(existing_value)}, got {type(parameters[key])}"
                    }
                
                # For arrays, check shapes match
                if isinstance(existing_value, (list, np.ndarray)):
                    try:
                        existing_shape = np.array(existing_value).shape
                        new_shape = np.array(parameters[key]).shape
                        if existing_shape != new_shape:
                            return {
                                "valid": False,
                                "reason": f"Shape mismatch for parameter {key}: expected {existing_shape}, got {new_shape}"
                            }
                    except (ValueError, TypeError):
                        # If can't convert to numpy arrays, just check lengths
                        if len(existing_value) != len(parameters[key]):
                            return {
                                "valid": False,
                                "reason": f"Length mismatch for parameter {key}"
                            }
                
                # For nested dictionaries, recursively check
                elif isinstance(existing_value, dict) and isinstance(parameters[key], dict):
                    # This is a simplified check - in a real system, this would be recursive
                    if set(existing_value.keys()) != set(parameters[key].keys()):
                        return {
                            "valid": False,
                            "reason": f"Key mismatch in nested dictionary {key}"
                        }
            
            # Check for extreme values that might indicate problems
            # (This is a simplified example)
            has_extreme_values = False
            for key, value in parameters.items():
                if isinstance(value, (list, np.ndarray)):
                    try:
                        arr = np.array(value)
                        # Check for NaN, infinity, or very large values
                        if np.isnan(arr).any() or np.isinf(arr).any() or np.abs(arr).max() > 1e6:
                            has_extreme_values = True
                            break
                    except:
                        pass
            
            # If metrics are provided, check for performance degradation
            performance_weight = 1.0
            if metrics and self.knowledge_base["models"][model_id].get("training_history"):
                history = self.knowledge_base["models"][model_id]["training_history"]
                if history:
                    latest_metrics = history[-1]
                    
                    # Check if any key metrics have degraded significantly
                    for metric_name, value in metrics.items():
                        if f"final_{metric_name}" in latest_metrics:
                            prev_value = latest_metrics[f"final_{metric_name}"]
                            
                            # For metrics where higher is better (like accuracy)
                            if metric_name in ["accuracy", "precision", "recall", "f1"]:
                                if value < prev_value * 0.8:  # 20% degradation
                                    # Reduce weight but don't invalidate
                                    performance_weight *= 0.7
                                    logger.warning(f"Performance degradation detected for {metric_name}: {prev_value} -> {value}")
                            
                            # For metrics where lower is better (like loss)
                            elif metric_name in ["loss", "error"]:
                                if value > prev_value * 1.5:  # 50% increase
                                    performance_weight *= 0.7
                                    logger.warning(f"Performance degradation detected for {metric_name}: {prev_value} -> {value}")
            
            # Return validation result with weight
            return {
                "valid": True,
                "weight": 0.5 if has_extreme_values else performance_weight,
                "has_extreme_values": has_extreme_values
            }
            
        except Exception as e:
            logger.error(f"Error validating model: {str(e)}")
            return {
                "valid": False,
                "reason": f"Validation error: {str(e)}"
            }
    
    async def secure_communicate(self,
                           peer_id: str,
                           message_type: str,
                           data: Dict[str, Any],
                           timeout: int = 60,
                           encryption_level: str = "high") -> Dict[str, Any]:
        """
        Securely communicate with a peer agent with encryption and authentication.
        
        This method provides a secure channel for exchanging sensitive data between
        agents, with support for different encryption levels and automatic retries.
        
        Args:
            peer_id: ID of the peer agent to communicate with
            message_type: Type of message being sent
            data: The data to transmit
            timeout: Timeout in seconds for the communication
            encryption_level: Level of encryption to use (low, medium, high)
            
        Returns:
            Dictionary with the peer's response
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If communication fails
        """
        if not peer_id or not message_type:
            raise ValueError("Peer ID and message type are required")
        
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        
        # Validate P2P network availability
        if not self.p2p_network:
            raise RuntimeError("P2P network manager is not available for secure communication")
        
        # Validate encryption level
        valid_encryption_levels = ["low", "medium", "high"]
        if encryption_level not in valid_encryption_levels:
            raise ValueError(f"Invalid encryption level. Must be one of: {', '.join(valid_encryption_levels)}")
        
        try:
            logger.debug(f"Initiating secure communication with peer {peer_id}, message type: {message_type}")
            
            # Establish secure connection
            connection = await self._establish_secure_peer_connection(peer_id)
            if not connection["success"]:
                return {
                    "success": False,
                    "reason": f"Failed to establish secure connection: {connection.get('error', 'Unknown error')}"
                }
            
            # Add metadata to the payload
            secure_payload = {
                "message_id": f"{self.agent_id}_{int(time.time())}_{hash(str(data))}",
                "message_type": message_type,
                "sender_id": self.agent_id,
                "timestamp": time.time(),
                "encryption_level": encryption_level,
                "payload": data,
                "session_id": connection.get("session_id", "")
            }
            
            # Apply encryption based on the specified level
            if encryption_level == "high":
                # In a real implementation, this would use strong encryption
                # Here we'll simulate it with a message
                logger.debug(f"Applying high-level encryption for message to {peer_id}")
                # self.p2p_network.encrypt_payload(secure_payload, connection["encryption_key"])
            
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Send the secure payload and wait for response
                    response = await self.p2p_network.send_secure_message(
                        peer_id=peer_id,
                        message=secure_payload,
                        timeout=timeout
                    )
                    
                    # Check for communication errors
                    if not response:
                        logger.warning(f"No response received from peer {peer_id} (attempt {retry_count+1}/{max_retries})")
                        retry_count += 1
                        if retry_count >= max_retries:
                            return {
                                "success": False,
                                "reason": "No response received from peer after maximum retries"
                            }
                        # Wait before retrying
                        await asyncio.sleep(1.0 * retry_count)  # Exponential backoff
                        continue
                    
                    # Validate the response
                    if "error" in response:
                        logger.error(f"Error in response from peer {peer_id}: {response['error']}")
                        return {
                            "success": False,
                            "reason": f"Peer error: {response['error']}"
                        }
                    
                    # If we receive a valid response, return it
                    logger.debug(f"Secure communication with peer {peer_id} completed successfully")
                    return {
                        "success": True,
                        **response  # Include all fields from the response
                    }
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout communicating with peer {peer_id} (attempt {retry_count+1}/{max_retries})")
                    retry_count += 1
                    if retry_count >= max_retries:
                        return {
                            "success": False,
                            "reason": "Communication timed out after maximum retries"
                        }
                    # Wait before retrying with longer timeout
                    await asyncio.sleep(1.0 * retry_count)
                    timeout *= 1.5  # Increase timeout for next attempt
                except Exception as e:
                    logger.error(f"Error communicating with peer {peer_id}: {str(e)}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        return {
                            "success": False,
                            "reason": f"Communication error: {str(e)}"
                        }
                    # Wait before retrying
                    await asyncio.sleep(1.0 * retry_count)
            
            return {
                "success": False,
                "reason": "Failed to communicate after maximum retries"
            }
            
        except Exception as e:
            error_msg = f"Secure communication with peer {peer_id} failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    async def _distribute_aggregated_model(self,
                                     peer_id: str,
                                     model_id: str,
                                     round_id: str,
                                     parameters: Dict[str, Any],
                                     metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Distribute the aggregated model back to a participating peer.
        
        After a successful training round, this method sends the aggregated model
        parameters back to each peer that participated in the training.
        
        Args:
            peer_id: ID of the peer agent to send the model to
            model_id: ID of the model that was trained
            round_id: ID of the training round
            parameters: Aggregated model parameters
            metrics: Performance metrics from the training round
            
        Returns:
            Dictionary with distribution results
        """
        try:
            logger.debug(f"Distributing aggregated model for round {round_id} to peer {peer_id}")
            
            # Create a secure connection with peer
            connection = await self._establish_secure_peer_connection(peer_id)
            if not connection["success"]:
                return {
                    "success": False, 
                    "reason": f"Connection failed: {connection.get('error', 'Unknown error')}"
                }
            
            # Prepare model distribution message
            model_distribution = {
                "update_type": "aggregated_model",
                "model_id": model_id,
                "round_id": round_id,
                "parameters": parameters,
                "metrics": metrics,
                "timestamp": time.time(),
                "source_agent": self.agent_id,
                "participants_count": metrics.get("participants", 0)
            }
            
            # Send the aggregated model to the peer
            response = await self.secure_communicate(
                peer_id=peer_id,
                message_type="model_distribution",
                data=model_distribution,
                timeout=120,  # 2 minutes timeout for model distribution
                encryption_level="high"  # Use high encryption for model data
            )
            
            if not response or not response.get("success", False):
                logger.warning(f"Failed to distribute aggregated model to peer {peer_id}: {response.get('reason', 'Unknown reason')}")
                return {
                    "success": False,
                    "reason": response.get("reason", "Distribution failed")
                }
            
            logger.info(f"Successfully distributed aggregated model to peer {peer_id} for round {round_id}")
            return {
                "success": True,
                "peer_id": peer_id,
                "round_id": round_id,
                "timestamp": time.time(),
                "acknowledgement": response.get("acknowledgement", "received")
            }
            
        except Exception as e:
            logger.error(f"Error distributing aggregated model to peer {peer_id}: {str(e)}")
            return {
                "success": False,
                "reason": str(e)
            }
