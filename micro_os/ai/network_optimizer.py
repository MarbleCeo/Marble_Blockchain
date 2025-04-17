import logging
import time
import random
from typing import Dict, List, Tuple, Optional, Set
import asyncio
import numpy as np
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PeerInfo:
    """Data class to store information about peers in the network."""
    peer_id: str
    address: str
    latency: float = 0.0
    bandwidth: float = 0.0
    reliability: float = 1.0  # 0.0 to 1.0
    last_seen: float = 0.0
    connection_success_rate: float = 1.0
    supported_protocols: List[str] = None
    
    def __post_init__(self):
        if self.supported_protocols is None:
            self.supported_protocols = []
        self.last_seen = time.time()

@dataclass
class RouteInfo:
    """Data class to store routing information."""
    source: str
    destination: str
    path: List[str]
    estimated_latency: float
    bandwidth: float
    last_updated: float
    hops: int
    
    def __post_init__(self):
        self.last_updated = time.time()
        self.hops = len(self.path) - 2  # Exclude source and destination


class NetworkOptimizationError(Exception):
    """Base exception class for network optimization errors."""
    pass


class PeerDiscoveryError(NetworkOptimizationError):
    """Exception raised for errors in the peer discovery process."""
    pass


class RoutingError(NetworkOptimizationError):
    """Exception raised for errors in the routing process."""
    pass


class TrafficAnalysisError(NetworkOptimizationError):
    """Exception raised for errors in traffic analysis."""
    pass


class NetworkOptimizer:
    """
    Advanced network optimizer using AI techniques to improve routing, 
    peer discovery, and overall network performance in the micro OS.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the NetworkOptimizer with optional configuration.
        
        Args:
            config: Configuration dictionary with optimization parameters
        """
        self.config = config or {
            'routing_update_interval': 60,  # seconds
            'peer_discovery_interval': 300,  # seconds
            'traffic_analysis_interval': 120,  # seconds
            'max_peers': 50,
            'min_reliability_threshold': 0.7,
            'adaptive_routing_enabled': True,
            'ai_optimization_level': 2,  # 0-3, with 3 being most aggressive
            'bandwidth_weight': 0.3,
            'latency_weight': 0.4,
            'reliability_weight': 0.3,
        }
        
        self.peers: Dict[str, PeerInfo] = {}
        self.routes: Dict[Tuple[str, str], RouteInfo] = {}
        self.traffic_stats: Dict[str, Dict] = {}
        self.network_conditions: Dict[str, float] = {
            'congestion_level': 0.0,
            'average_latency': 0.0,
            'packet_loss_rate': 0.0,
        }
        
        self.running = False
        self.tasks = []
        
        logger.info("Network optimizer initialized with configuration: %s", self.config)
    
    async def start(self):
        """Start the network optimizer and its background tasks."""
        if self.running:
            logger.warning("Network optimizer is already running")
            return
        
        self.running = True
        logger.info("Starting network optimizer")
        
        # Start background tasks
        self.tasks = [
            asyncio.create_task(self._periodic_peer_discovery()),
            asyncio.create_task(self._periodic_route_optimization()),
            asyncio.create_task(self._periodic_traffic_analysis()),
            asyncio.create_task(self._adaptive_routing_monitor())
        ]
    
    async def stop(self):
        """Stop the network optimizer and all its background tasks."""
        if not self.running:
            logger.warning("Network optimizer is not running")
            return
        
        self.running = False
        logger.info("Stopping network optimizer")
        
        # Cancel all background tasks
        for task in self.tasks:
            task.cancel()
        
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks = []
    
    async def _periodic_peer_discovery(self):
        """Background task for periodic peer discovery."""
        try:
            while self.running:
                await self.discover_peers()
                await asyncio.sleep(self.config['peer_discovery_interval'])
        except asyncio.CancelledError:
            logger.info("Peer discovery task cancelled")
        except Exception as e:
            logger.error("Error in peer discovery task: %s", str(e), exc_info=True)
            if self.running:
                # Restart the task after a delay
                asyncio.create_task(self._delayed_restart(self._periodic_peer_discovery, 10))
    
    async def _periodic_route_optimization(self):
        """Background task for periodic route optimization."""
        try:
            while self.running:
                await self.optimize_routes()
                await asyncio.sleep(self.config['routing_update_interval'])
        except asyncio.CancelledError:
            logger.info("Route optimization task cancelled")
        except Exception as e:
            logger.error("Error in route optimization task: %s", str(e), exc_info=True)
            if self.running:
                # Restart the task after a delay
                asyncio.create_task(self._delayed_restart(self._periodic_route_optimization, 10))
    
    async def _periodic_traffic_analysis(self):
        """Background task for periodic traffic analysis."""
        try:
            while self.running:
                await self.analyze_traffic()
                await asyncio.sleep(self.config['traffic_analysis_interval'])
        except asyncio.CancelledError:
            logger.info("Traffic analysis task cancelled")
        except Exception as e:
            logger.error("Error in traffic analysis task: %s", str(e), exc_info=True)
            if self.running:
                # Restart the task after a delay
                asyncio.create_task(self._delayed_restart(self._periodic_traffic_analysis, 10))
    
    async def _adaptive_routing_monitor(self):
        """Background task for monitoring and adjusting to network conditions."""
        try:
            while self.running:
                if self.config['adaptive_routing_enabled']:
                    await self.adapt_to_network_conditions()
                await asyncio.sleep(30)  # Check network conditions every 30 seconds
        except asyncio.CancelledError:
            logger.info("Adaptive routing monitor task cancelled")
        except Exception as e:
            logger.error("Error in adaptive routing monitor: %s", str(e), exc_info=True)
            if self.running:
                # Restart the task after a delay
                asyncio.create_task(self._delayed_restart(self._adaptive_routing_monitor, 10))
    
    async def _delayed_restart(self, task_func, delay):
        """Restart a task after a delay period."""
        await asyncio.sleep(delay)
        if self.running:
            self.tasks.append(asyncio.create_task(task_func()))
    
    async def discover_peers(self):
        """
        Discover new peers in the network and update peer information.
        
        Raises:
            PeerDiscoveryError: If there's an issue with peer discovery
        """
        try:
            logger.info("Starting peer discovery")
            
            # Simulate peer discovery process
            # In a real implementation, this would interact with the VM protocol
            # to discover and connect to new peers
            newly_discovered = self._simulate_peer_discovery()
            
            # Update existing peers with new information
            current_time = time.time()
            for peer_id, peer_info in list(self.peers.items()):
                if current_time - peer_info.last_seen > 3600:  # 1 hour timeout
                    logger.info(f"Removing stale peer {peer_id}")
                    self.peers.pop(peer_id, None)
            
            # Prune peers if we exceed the maximum
            if len(self.peers) > self.config['max_peers']:
                self._prune_peers()
            
            logger.info(f"Peer discovery completed. Total peers: {len(self.peers)}")
            return newly_discovered
            
        except Exception as e:
            logger.error(f"Peer discovery failed: {str(e)}", exc_info=True)
            raise PeerDiscoveryError(f"Failed to discover peers: {str(e)}") from e
    
    def _simulate_peer_discovery(self) -> List[str]:
        """
        Simulate peer discovery for testing purposes.
        In a real implementation, this would connect to the actual network.
        
        Returns:
            List of newly discovered peer IDs
        """
        newly_discovered = []
        
        # Simulate finding 1-3 new peers
        for _ in range(random.randint(1, 3)):
            peer_id = f"peer_{random.randint(1000, 9999)}"
            
            # Skip if peer already exists
            if peer_id in self.peers:
                continue
                
            # Create new peer with simulated properties
            self.peers[peer_id] = PeerInfo(
                peer_id=peer_id,
                address=f"192.168.1.{random.randint(2, 254)}",
                latency=random.uniform(10, 200),
                bandwidth=random.uniform(1, 10),
                reliability=random.uniform(0.8, 1.0),
                supported_protocols=["libp2p", "ipfs"]
            )
            newly_discovered.append(peer_id)
            logger.debug(f"Discovered new peer: {peer_id}")
            
        return newly_discovered
    
    def _prune_peers(self):
        """
        Remove the least valuable peers to stay under the maximum peer limit.
        Uses a scoring system based on peer metrics.
        """
        if len(self.peers) <= self.config['max_peers']:
            return
            
        # Score peers based on latency, bandwidth, and reliability
        scored_peers = []
        for peer_id, info in self.peers.items():
            # Lower latency is better, higher bandwidth and reliability are better
            latency_score = 1000 / (info.latency + 10)  # Avoid division by zero
            score = (
                latency_score * self.config['latency_weight'] +
                info.bandwidth * self.config['bandwidth_weight'] +
                info.reliability * self.config['reliability_weight']
            )
            scored_peers.append((peer_id, score))
        
        # Sort by score (lowest first)
        scored_peers.sort(key=lambda x: x[1])
        
        # Remove lowest-scoring peers
        peers_to_remove = len(self.peers) - self.config['max_peers']
        for i in range(peers_to_remove):
            peer_id = scored_peers[i][0]
            self.peers.pop(peer_id, None)
            logger.info(f"Pruned low-scoring peer: {peer_id}")
    
    async def optimize_routes(self):
        """
        Optimize network routes based on peer information and network conditions.
        
        Raises:
            RoutingError: If there's an issue with route optimization
        """
        try:
            logger.info("Starting route optimization")
            
            if len(self.peers) < 2:
                logger.warning("Not enough peers for route optimization")
                return
            
            # Build a graph representation of the network
            graph = self._build_network_graph()
            
            # Find optimal routes between all peer pairs
            source_peers = list(self.peers.keys())
            destination_peers = list(self.peers.keys())
            
            # Limit the number of routes to optimize for large networks
            if len(source_peers) > 10:
                source_peers = random.sample(source_peers, 10)
            
            for source in source_peers:
                for destination in destination_peers:
                    if source == destination:
                        continue
                    
                    route = self._find_optimal_route(graph, source, destination)
                    if route:
                        key = (source, destination)
                        self.routes[key] = route
                        logger.debug(f"Optimized route: {source} -> {destination}")
            
            logger.info(f"Route optimization completed. Total routes: {len(self.routes)}")
            
        except Exception as e:
            logger.error(f"Route optimization failed: {str(e)}", exc_info=True)
            raise RoutingError(f"Failed to optimize routes: {str(e)}") from e
    
    def _build_network_graph(self) -> Dict:
        """
        Build a graph representation of the network for path finding.
        
        Returns:
            Dictionary representing the network graph
        """
        graph = {}
        
        # Create graph nodes for each peer
        for peer_id in self.peers:
            graph[peer_id] = {}
        
        # Create edges between peers with weights based on latency and bandwidth
        for source_id, source_info in self.peers.items():
            for dest_id, dest_info in self.peers.items():
                if source_id == dest_id:
                    continue
                
                # Calculate edge weight (lower is better)
                # Weight is influenced by latency, inverse of bandwidth, and reliability
                latency = random.uniform(10, 100)  # Simulated latency between peers
                bandwidth = min(source_info.bandwidth, dest_info.bandwidth)
                reliability = source_info.reliability * dest_info.reliability
                
                # Skip unreliable connections
                if reliability < self.config['min_reliability_threshold']:
                    continue
                
                # Calculate weighted score (lower is better)
                weight = (
                    latency * self.config['latency_weight'] + 
                    (1 / bandwidth) * self.config['bandwidth_weight'] + 
                    (1 / reliability) * self.config['reliability_weight']
                )
                
                graph[source_id][dest_id] = weight
        
        return graph
    
    def _find_optimal_route(self, graph: Dict, source: str, destination: str) -> Optional[RouteInfo]:
        """
        Find the optimal route from source to destination using Dijkstra's algorithm.
        
        Args:
            graph: The network graph
            source: Source peer ID
            destination: Destination peer ID
            
        Returns:
            RouteInfo object if a route is found, None otherwise
        """
        if source not in graph or destination not in graph:
            return None
        
        # Initialize distances and predecessors
        distances = {node: float('infinity') for node in graph}
        predecessors = {node: None for node in graph}
        distances[source] = 0
        unvisited = set(graph.keys())
        
        while unvisited:
            # Find the unvisited node with the smallest distance
            current = min(unvisited, key=lambda node: distances[node])
            
            # If we've reached the destination or if the smallest distance
            # among unvisited nodes is infinity, we're done
            if current == destination or distances[current] == float('infinity'):
                break
                
            # Remove the current node from unvisited
            unvisited.remove(current)
            
            # Check all neighbors of

