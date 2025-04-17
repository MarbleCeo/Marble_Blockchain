#!/usr/bin/env python3
"""
VM Protocol - Secure VM-to-VM communication implementation using libp2p

This module provides secure communication between virtual machines in the micro OS
with support for encryption, NAT traversal, message signing, and message handlers for
VM synchronization, state management, resource discovery, and container orchestration.
"""

import asyncio
import json
import logging
import os
import time
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Callable, Tuple

# Libp2p imports
from libp2p import new_node
from libp2p.crypto.secp256k1 import create_new_key_pair
from libp2p.peer.peerinfo import info_from_p2p_addr
from libp2p.pubsub.pubsub import Pubsub
from libp2p.pubsub.gossipsub import GossipSub
from libp2p.security.secio import securio
from libp2p.security.noise import noise
from libp2p.network.stream.net_stream_interface import INetStream
from libp2p.tools.authentication import AuthenticationAPI
from libp2p.tools.nat_manager import NatManager

# Create a logger for this module
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages that can be exchanged between VMs"""
    VM_DISCOVERY = auto()
    VM_STATE = auto()
    RESOURCE_INFO = auto()
    CIRCUIT_SYNC = auto()
    CONTAINER_ORCHESTRATION = auto()
    ERROR = auto()
    PING = auto()
    PONG = auto()


class VMProtocolError(Exception):
    """Base exception for VM protocol errors"""
    pass


class ConnectionError(VMProtocolError):
    """Raised when there is an error connecting to peers"""
    pass


class MessageError(VMProtocolError):
    """Raised when there is an error processing messages"""
    pass


class SecurityError(VMProtocolError):
    """Raised when there is a security-related error"""
    pass


class VMProtocol:
    """
    Secure communication protocol for VMs using libp2p.
    
    This class implements a secure peer-to-peer communication protocol that
    enables VMs to discover each other, synchronize state, share resources,
    and orchestrate containers across the network.
    
    Features:
    - End-to-end encryption using libp2p's securio and noise protocols
    - NAT traversal for connecting through firewalls
    - Message signing for authenticity verification
    - Circuit state synchronization across VMs
    - Container orchestration support
    """
    
    def __init__(self, 
                 vm_id: str, 
                 bootstrap_peers: List[str] = None,
                 listen_port: int = 9000,
                 key_file: str = None):
        """
        Initialize the VM protocol.
        
        Args:
            vm_id: Unique identifier for this VM
            bootstrap_peers: List of multiaddrs for initial peers to connect to
            listen_port: Port to listen on
            key_file: Path to the key file for this VM
        """
        self.vm_id = vm_id
        self.bootstrap_peers = bootstrap_peers or []
        self.listen_port = listen_port
        self.key_file = key_file
        
        # Peer connections and state
        self.peers = {}  # peer_id -> peer_info
        self.node = None
        self.pubsub = None
        self.nat_manager = None
        self.message_handlers = {}
        self.resource_info = {}
        self.circuit_state = {}
        self.container_info = {}
        
        # Message queue for handling messages asynchronously
        self.message_queue = asyncio.Queue()
        
        # Register message handlers
        self._register_default_handlers()
    
    async def start(self):
        """
        Start the VM protocol node.
        
        This initializes the libp2p node with security protocols, sets up 
        pubsub for broadcasting messages, and connects to bootstrap peers.
        """
        try:
            # Initialize key pair
            if self.key_file and os.path.exists(self.key_file):
                # Load existing key
                with open(self.key_file, 'rb') as f:
                    private_key = f.read()
                key_pair = create_new_key_pair(private_key)
            else:
                # Create new key pair
                key_pair = create_new_key_pair()
                if self.key_file:
                    with open(self.key_file, 'wb') as f:
                        f.write(key_pair.private_key.serialize())
            
            # Security transports
            security_protocols = [securio, noise]
            
            # Create a new libp2p node
            self.node = await new_node(
                key_pair=key_pair,
                transport_opt=[f"/ip4/0.0.0.0/tcp/{self.listen_port}"],
                muxer_opt=["/mplex/6.7.0"],
                sec_opt=security_protocols,
                enable_pubsub=True,
                pubsub_protocol=GossipSub
            )
            
            # Get the pubsub interface
            self.pubsub = self.node.get_pubsub()
            
            # Setup NAT traversal
            self.nat_manager = NatManager(self.node)
            await self.nat_manager.start()
            
            # Setup authentication API
            self.auth_api = AuthenticationAPI(self.node)
            
            # Get our peer ID and addresses
            peer_id = self.node.get_id().pretty()
            listened_addrs = [str(addr) for addr in self.node.get_addrs()]
            logger.info(f"VM Protocol started with peer ID: {peer_id}")
            logger.info(f"Listening on: {listened_addrs}")
            
            # Subscribe to topics
            for msg_type in MessageType:
                topic = f"micro_os/{msg_type.name.lower()}"
                await self.pubsub.subscribe(topic)
            
            # Connect to bootstrap peers
            await self._connect_to_bootstrap_peers()
            
            # Start message handling loop
            asyncio.create_task(self._handle_messages())
            
            # Announce our presence
            await self._announce_presence()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start VM Protocol: {str(e)}")
            raise ConnectionError(f"Failed to start VM Protocol: {str(e)}")
    
    async def stop(self):
        """Stop the VM protocol node and clean up resources."""
        logger.info("Stopping VM Protocol...")
        if self.nat_manager:
            await self.nat_manager.stop()
        
        if self.node:
            await self.node.stop()
        
        logger.info("VM Protocol stopped")
    
    async def _connect_to_bootstrap_peers(self):
        """Connect to the bootstrap peers."""
        if not self.bootstrap_peers:
            logger.info("No bootstrap peers specified")
            return
        
        for peer_addr in self.bootstrap_peers:
            try:
                peer_info = info_from_p2p_addr(peer_addr)
                await self.node.connect(peer_info)
                logger.info(f"Connected to bootstrap peer: {peer_addr}")
            except Exception as e:
                logger.warning(f"Failed to connect to bootstrap peer {peer_addr}: {str(e)}")
    
    async def _announce_presence(self):
        """Announce presence to the network."""
        message = {
            "type": MessageType.VM_DISCOVERY.name,
            "vm_id": self.vm_id,
            "timestamp": time.time(),
            "addrs": [str(addr) for addr in self.node.get_addrs()],
            "resources": self._get_resource_info()
        }
        
        topic = f"micro_os/{MessageType.VM_DISCOVERY.name.lower()}"
        await self.publish_message(topic, message)
        logger.info("Announced presence to the network")
    
    def _get_resource_info(self) -> Dict[str, Any]:
        """Get resource information for this VM."""
        # This would be expanded to include actual resource metrics
        return {
            "cpu_cores": os.cpu_count(),
            "memory_available": "8GB",  # Placeholder
            "containers": len(self.container_info),
            "load": 0.5  # Placeholder
        }
    
    async def publish_message(self, topic: str, message: Dict[str, Any]):
        """
        Publish a message to a topic.
        
        Args:
            topic: The topic to publish to
            message: The message to publish (will be JSON encoded)
        """
        try:
            # Add sender information
            message["sender"] = self.vm_id
            message["timestamp"] = time.time()
            
            # Sign the message with our private key
            message_str = json.dumps(message)
            signature = self.node.get_private_key().sign(message_str.encode())
            
            # Add signature to message
            signed_message = {
                "data": message,
                "signature": signature.hex(),
                "peer_id": self.node.get_id().pretty()
            }
            
            # Publish the signed message
            encoded_message = json.dumps(signed_message).encode()
            await self.pubsub.publish(topic, encoded_message)
            
            logger.debug(f"Published message to topic {topic}: {message['type']}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to publish message: {str(e)}")
            return False
    
    async def _handle_messages(self):
        """
        Main message handling loop.
        
        This coroutine processes incoming messages from the pubsub system
        and dispatches them to the appropriate handlers.
        """
        # Set up pubsub message handler
        async def pubsub_handler(topic, message):
            try:
                message_data = json.loads(message.decode())
                # Verify signature
                if not self._verify_message(message_data):
                    logger.warning(f"Received message with invalid signature on topic {topic}")
                    return
                
                # Extract the actual message
                actual_message = message_data["data"]
                
                # Queue the message for processing
                await self.message_queue.put((topic, actual_message))
                
            except json.JSONDecodeError:
                logger.warning(f"Received malformed message on topic {topic}")
            except Exception as e:
                logger.error(f"Error handling pubsub message: {str(e)}")
        
        # Subscribe to all topics with our handler
        for msg_type in MessageType:
            topic = f"micro_os/{msg_type.name.lower()}"
            self.pubsub.subscribe_to_topic(topic, pubsub_handler)
        
        # Process messages from the queue
        while True:
            try:
                topic, message = await self.message_queue.get()
                
                # Extract message type
                message_type = message.get("type")
                if not message_type:
                    logger.warning(f"Received message without type field: {message}")
                    continue
                
                # Find and call the appropriate handler
                handler = self.message_handlers.get(message_type)
                if handler:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.error(f"Error in message handler for {message_type}: {str(e)}")
                else:
                    logger.warning(f"No handler for message type: {message_type}")
                
                # Mark task as complete
                self.message_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in message handling loop: {str(e)}")
    
    def _verify_message(self, message_data: Dict[str, Any]) -> bool:
        """
        Verify the signature of a message.
        
        Args:
            message_data: The message data containing signature, peer_id, and data
            
        Returns:
            bool: True if signature is valid, False otherwise
        """
        try:
            signature = bytes.fromhex(message_data["signature"])
            peer_id = message_data["peer_id"]
            data = json.dumps(message_data["data"]).encode()
            
            # Get the public key from the peer ID
            if peer_id in self.peers:
                public_key = self.peers[peer_id].public_key
                return public_key.verify(data, signature)
            else:
                # For now, return True for unknown peers to allow discovery
                # In a production environment, we'd want to verify through other means
                logger.warning(f"Received message from unknown peer: {peer_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error verifying message signature: {str(e)}")
            return False
    
    def _register_default_handlers(self):
        """Register default message handlers."""
        self.message_handlers = {
            MessageType.VM_DISCOVERY.name: self._handle_vm_discovery,
            MessageType.VM_STATE.name: self._handle_vm_state,
            MessageType.RESOURCE_INFO.name: self._handle_resource_info,
            MessageType.CIRCUIT_SYNC.name: self._handle_circuit_sync,
            MessageType.CONTAINER_ORCHESTRATION.name: self._handle_container_orchestration,
            MessageType.PING.name: self._handle_ping,
            MessageType.PONG.name: self._handle_pong,
            MessageType.ERROR.name: self._handle_error
        }
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """
        Register a custom message handler.
        
        Args:
            message_type: The type of message to handle
            handler: The handler function, which should accept a message dict
        """
        self.message_handlers[message_type.name] = handler
    
    async def _handle_vm_discovery(self, message: Dict[str, Any]):
        """
        Handle VM discovery messages.
        
        Args:
            message: The discovery message containing VM information
        """
        vm_id = message.get("vm_id")
        addrs = message.get("addrs", [])
        resources = message.get("resources", {})
        
        if not vm_id or vm_id == self.vm_id:
            return
        
        logger.info(f"Discovered VM: {vm_id}")
        
        # Store peer information
        self.peers[vm_id] = {
            "addrs": addrs,
            "resources": resources,
            "last_seen": time.time()
        }
        
        # Send our state information to the new peer
        await self._send_vm_state(vm_id)
    
    async def _send_vm_state(self, target_vm_id: Optional[str] = None):
        """
        Send VM state information to peers.
        
        Args:
            target_vm_id: If specified, send only to this VM, otherwise broadcast
        """
        message = {
            "type": MessageType.VM_STATE.name,
            "vm_id": self.vm_id,
            "timestamp": time.time(),
            "state": {
                "containers": list(self.container_info.keys()),
                "circuits

