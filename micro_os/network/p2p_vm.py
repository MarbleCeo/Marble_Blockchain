from libp2p import new_host
from libp2p.peer.peerinfo import info_from_p2p_addr
from libp2p.pubsub.pubsub import Pubsub
from libp2p.pubsub.gossipsub import GossipSub
from libp2p.network.stream.net_stream_interface import INetStream

import asyncio
import json
from typing import Dict, List, Optional, Callable
import logging
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PeerMessage:
    msg_type: str
    payload: dict
    timestamp: float
    sender: str

class P2PNetworkManager:
    """
    Manages P2P networking between VMs using libp2p.
    Handles peer discovery, message routing, and network state.
    """
    def __init__(self, host_ip: str = "0.0.0.0", port: int = 8888):
        self.logger = logging.getLogger(__name__)
        self.host_ip = host_ip
        self.port = port
        self.host = None
        self.pubsub = None
        self.peers: Dict[str, dict] = {}
        self.message_handlers: Dict[str, List[Callable]] = {}
        
        # Initialize topics
        self.topics = {
            "vm_state": "vm_state_updates",
            "resource_request": "resource_requests",
            "peer_discovery": "peer_discovery"
        }

    async def start(self):
        """Initialize and start the P2P network node."""
        try:
            # Create libp2p host
            self.host = await new_host(
                transport_opt=[f"/ip4/{self.host_ip}/tcp/{self.port}"]
            )
            
            # Initialize GossipSub protocol
            self.pubsub = GossipSub(self.host)
            
            # Set up protocol handlers
            await self._setup_protocol_handlers()
            
            # Subscribe to topics
            await self._subscribe_to_topics()
            
            self.logger.info(f"P2P Network started on {self.host_ip}:{self.port}")
            self.logger.info(f"Node ID: {self.host.get_id()}")
            
            return {
                "node_id": self.host.get_id(),
                "address": f"{self.host_ip}:{self.port}",
                "status": "running"
            }
        
        except Exception as e:
            self.logger.error(f"Failed to start P2P network: {str(e)}")
            raise RuntimeError(f"P2P network start failed: {str(e)}")

    async def _setup_protocol_handlers(self):
        """Set up handlers for different P2P protocols."""
        # VM State Updates
        await self.host.set_stream_handler("/vm/state/1.0.0", self._handle_vm_state)
        
        # Resource Requests
        await self.host.set_stream_handler("/vm/resource/1.0.0", self._handle_resource_request)
        
        # Peer Discovery
        await self.host.set_stream_handler("/peer/discovery/1.0.0", self._handle_peer_discovery)

    async def _subscribe_to_topics(self):
        """Subscribe to all required pubsub topics."""
        for topic in self.topics.values():
            await self.pubsub.subscribe(topic)

    async def _handle_vm_state(self, stream: INetStream):
        """Handle incoming VM state updates."""
        try:
            data = await self._read_stream(stream)
            message = PeerMessage(**json.loads(data))
            
            # Update peer state
            peer_id = str(stream.muxed_conn.peer_id)
            self.peers[peer_id] = {
                "last_seen": datetime.now().timestamp(),
                "state": message.payload
            }
            
            # Notify handlers
            await self._notify_handlers("vm_state", message)
        
        except Exception as e:
            self.logger.error(f"Error handling VM state: {str(e)}")

    async def _handle_resource_request(self, stream: INetStream):
        """Handle incoming resource requests."""
        try:
            data = await self._read_stream(stream)
            message = PeerMessage(**json.loads(data))
            
            # Process resource request
            await self._notify_handlers("resource_request", message)
            
            # Send response
            response = {
                "status": "received",
                "request_id": message.payload.get("request_id")
            }
            await self._write_stream(stream, json.dumps(response))
        
        except Exception as e:
            self.logger.error(f"Error handling resource request: {str(e)}")

    async def _handle_peer_discovery(self, stream: INetStream):
        """Handle peer discovery requests."""
        try:
            # Send peer info
            peer_info = {
                "node_id": self.host.get_id(),
                "addresses": self.host.get_addrs(),
                "protocols": list(self.host.get_protocols())
            }
            await self._write_stream(stream, json.dumps(peer_info))
        
        except Exception as e:
            self.logger.error(f"Error handling peer discovery: {str(e)}")

    async def connect_to_peer(self, peer_addr: str) -> bool:
        """Connect to a peer using their multiaddr."""
        try:
            peer_info = info_from_p2p_addr(peer_addr)
            await self.host.connect(peer_info)
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to connect to peer {peer_addr}: {str(e)}")
            return False

    async def broadcast_vm_state(self, state: dict):
        """Broadcast VM state update to the network."""
        message = PeerMessage(
            msg_type="vm_state",
            payload=state,
            timestamp=datetime.now().timestamp(),
            sender=str(self.host.get_id())
        )
        
        await self.pubsub.publish(
            self.topics["vm_state"],
            json.dumps(message.__dict__).encode()
        )

    async def request_resources(self, requirements: dict) -> dict:
        """Request resources from the network."""
        message = PeerMessage(
            msg_type="resource_request",
            payload=requirements,
            timestamp=datetime.now().timestamp(),
            sender=str(self.host.get_id())
        )
        
        await self.pubsub.publish(
            self.topics["resource_request"],
            json.dumps(message.__dict__).encode()
        )
        
        # Wait for responses (implement actual response handling)
        return {"status": "requested", "request_id": requirements.get("request_id")}

    async def discover_peers(self) -> List[dict]:
        """Discover and return information about all peers."""
        peer_info_list = []
        for peer_id, peer_data in self.peers.items():
            if datetime.now().timestamp() - peer_data["last_seen"] < 300:  # 5 min timeout
                peer_info_list.append({
                    "peer_id": peer_id,
                    **peer_data
                })
        return peer_info_list

    def add_message_handler(self, msg_type: str, handler: Callable):
        """Add a message handler for a specific message type."""
        if msg_type not in self.message_handlers:
            self.message_handlers[msg_type] = []
        self.message_handlers[msg_type].append(handler)

    async def _notify_handlers(self, msg_type: str, message: PeerMessage):
        """Notify all handlers for a specific message type."""
        if msg_type in self.message_handlers:
            for handler in self.message_handlers[msg_type]:
                await handler(message)

    @staticmethod
    async def _read_stream(stream: INetStream) -> str:
        """Read data from a stream."""
        data = await stream.read()
        return data.decode()

    @staticmethod
    async def _write_stream(stream: INetStream, data: str):
        """Write data to a stream."""
        await stream.write(data.encode())

