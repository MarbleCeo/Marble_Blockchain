#!/usr/bin/env python3
"""
P2P Client implementation using libp2p for peer-to-peer communication.
This module provides a robust client that can connect to peers, send/receive messages,
and handle connection errors gracefully.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Union, cast

from libp2p import new_host
from libp2p.host.basic_host import BasicHost
from libp2p.network.stream.net_stream_interface import INetStream
from libp2p.peer.id import ID
from libp2p.peer.peerinfo import PeerInfo
from libp2p.typing import TProtocol
from multiaddr import Multiaddr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("P2PClient")

class MessageType(Enum):
    """Types of messages that can be exchanged between peers."""
    CHAT = auto()
    TRANSACTION = auto()
    NICKNAME = auto()
    DISCOVERY = auto()
    HEARTBEAT = auto()
    ERROR = auto()
    

@dataclass
class Message:
    """Represents a message in the P2P network."""
    msg_type: MessageType
    sender: str
    content: Any
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_json(self) -> str:
        """Convert message to JSON format for transmission."""
        return json.dumps({
            "msg_type": self.msg_type.name,
            "sender": self.sender,
            "content": self.content,
            "timestamp": self.timestamp,
            "message_id": self.message_id
        })
    
    @classmethod
    def from_json(cls, json_data: str) -> 'Message':
        """Create a Message object from JSON data."""
        try:
            data = json.loads(json_data)
            return cls(
                msg_type=MessageType[data["msg_type"]],
                sender=data["sender"],
                content=data["content"],
                timestamp=data["timestamp"],
                message_id=data["message_id"]
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse message: {e}")
            # Return an error message if parsing fails
            return cls(
                msg_type=MessageType.ERROR,
                sender="system",
                content=f"Invalid message format: {e}",
            )


class ConnectionState(Enum):
    """Possible states for a client connection."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    FAILED = auto()


class Client:
    """
    P2P Client that handles communication with other peers using libp2p.
    
    This client can:
    - Connect to peers
    - Send and receive messages
    - Handle connection errors
    - Queue messages when connection is unavailable
    - Automatically reconnect when connection is lost
    """
    
    def __init__(
        self, 
        nickname: str,
        protocol_id: str = "/p2p/1.0.0",
        bootstrap_peers: Optional[List[str]] = None,
        max_reconnect_attempts: int = 5,
        reconnect_delay: int = 5,
        message_handlers: Optional[Dict[MessageType, List[Callable]]] = None
    ):
        """
        Initialize a new P2P Client.
        
        Args:
            nickname: The user's nickname in the network
            protocol_id: Protocol identifier for libp2p
            bootstrap_peers: List of peer multiaddresses to connect to on startup
            max_reconnect_attempts: Maximum number of reconnection attempts
            reconnect_delay: Delay between reconnection attempts in seconds
            message_handlers: Dictionary of callbacks for different message types
        """
        self.nickname = nickname
        self.protocol_id = TProtocol(protocol_id)
        self.bootstrap_peers = bootstrap_peers or []
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        
        # Initialize connection state
        self.state = ConnectionState.DISCONNECTED
        self.host: Optional[BasicHost] = None
        self.peer_id: Optional[ID] = None
        self.connected_peers: Set[str] = set()
        
        # Message handling
        self.message_queue: List[Message] = []
        self.message_handlers = message_handlers or {msg_type: [] for msg_type in MessageType}
        
        # Default handlers
        self._register_default_handlers()
        
        # Event loop and tasks
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.background_tasks: List[asyncio.Task] = []

    def _register_default_handlers(self) -> None:
        """Register the default message handlers."""
        # Register system message handler
        if MessageType.ERROR not in self.message_handlers:
            self.message_handlers[MessageType.ERROR] = []
        self.message_handlers[MessageType.ERROR].append(self._handle_error_message)
        
        # Register heartbeat handler
        if MessageType.HEARTBEAT not in self.message_handlers:
            self.message_handlers[MessageType.HEARTBEAT] = []
        self.message_handlers[MessageType.HEARTBEAT].append(self._handle_heartbeat)
        
        # Register discovery handler
        if MessageType.DISCOVERY not in self.message_handlers:
            self.message_handlers[MessageType.DISCOVERY] = []
        self.message_handlers[MessageType.DISCOVERY].append(self._handle_discovery)

    def register_handler(self, msg_type: MessageType, handler: Callable) -> None:
        """
        Register a new message handler for a specific message type.
        
        Args:
            msg_type: The type of message to handle
            handler: The callback function to handle the message
        """
        if msg_type not in self.message_handlers:
            self.message_handlers[msg_type] = []
        self.message_handlers[msg_type].append(handler)
        logger.debug(f"Registered new handler for {msg_type.name}")

    async def _handle_error_message(self, message: Message) -> None:
        """Handle error messages."""
        logger.error(f"Received error: {message.content}")

    async def _handle_heartbeat(self, message: Message) -> None:
        """Handle heartbeat messages to keep connection alive."""
        # Respond to heartbeat with another heartbeat
        if message.sender != self.nickname:
            await self.send_message(
                MessageType.HEARTBEAT, 
                f"Heartbeat response from {self.nickname}",
                direct_peer=message.sender
            )

    async def _handle_discovery(self, message: Message) -> None:
        """Handle peer discovery messages."""
        if message.sender != self.nickname:
            # Add peer to connected peers list if not already there
            self.connected_peers.add(message.sender)
            # Respond with our own discovery message
            await self.send_message(
                MessageType.DISCOVERY,
                {
                    "nickname": self.nickname,
                    "peer_id": str(self.peer_id) if self.peer_id else None
                }
            )
            logger.info(f"Discovered new peer: {message.sender}")

    async def start(self) -> None:
        """Start the P2P client and connect to the network."""
        self.loop = asyncio.get_running_loop()
        self.state = ConnectionState.CONNECTING
        
        try:
            # Create a new libp2p host
            self.host = new_host()
            self.peer_id = self.host.get_id()
            
            # Set up stream handler for incoming connections
            self.host.set_stream_handler(self.protocol_id, self._handle_stream)
            
            # Connect to bootstrap peers
            self.background_tasks.append(
                self.loop.create_task(self._connect_to_bootstrap_peers())
            )
            
            # Start heartbeat task
            self.background_tasks.append(
                self.loop.create_task(self._send_periodic_heartbeats())
            )
            
            # Start message queue processor
            self.background_tasks.append(
                self.loop.create_task(self._process_message_queue())
            )
            
            # Announce ourselves to the network
            await self.send_message(
                MessageType.DISCOVERY,
                {
                    "nickname": self.nickname,
                    "peer_id": str(self.peer_id)
                }
            )
            
            self.state = ConnectionState.CONNECTED
            logger.info(f"P2P Client started with peer ID: {self.peer_id}")
            logger.info(f"Listening addresses: {[str(addr) for addr in self.host.get_addrs()]}")
            
        except Exception as e:
            self.state = ConnectionState.FAILED
            logger.error(f"Failed to start P2P client: {e}")
            raise
    
    async def stop(self) -> None:
        """
        Stop the P2P client and clean up resources.
        """
        logger.info("Stopping P2P client...")
        
        # Cancel all background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Close the host
        if self.host:
            await self.host.close()
        
        self.state = ConnectionState.DISCONNECTED
        logger.info("P2P client stopped")

    async def _connect_to_bootstrap_peers(self) -> None:
        """Connect to bootstrap peers to join the network."""
        for peer_addr in self.bootstrap_peers:
            try:
                multiaddr = Multiaddr(peer_addr)
                await self.host.connect(PeerInfo(ID.from_string(str(multiaddr)), [multiaddr]))
                logger.info(f"Connected to bootstrap peer: {peer_addr}")
            except Exception as e:
                logger.error(f"Failed to connect to bootstrap peer {peer_addr}: {e}")
    
    async def _handle_stream(self, stream: INetStream) -> None:
        """
        Handle incoming streams from other peers.
        
        Args:
            stream: The incoming network stream
        """
        peer_id = stream.muxed_conn.peer_id
        logger.debug(f"New stream from: {peer_id}")
        
        try:
            # Read from the stream until it's closed
            while True:
                read_bytes = await stream.read()
                if not read_bytes:
                    break
                
                # Process the received message
                message_str = read_bytes.decode('utf-8')
                message = Message.from_json(message_str)
                await self._process_message(message)
                
        except Exception as e:
            logger.error(f"Error handling stream from {peer_id}: {e}")
        finally:
            await stream.close()
    
    async def _process_message(self, message: Message) -> None:
        """
        Process a received message and dispatch it to the appropriate handlers.
        
        Args:
            message: The message to process
        """
        logger.debug(f"Processing message of type {message.msg_type.name} from {message.sender}")
        
        # Call all registered handlers for this message type
        if message.msg_type in self.message_handlers:
            for handler in self.message_handlers[message.msg_type]:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")
    
    async def send_message(
        self, 
        msg_type: MessageType, 
        content: Any, 
        direct_peer: Optional[str] = None
    ) -> None:
        """
        Send a message to peers in the network.
        
        Args:
            msg_type: The type of message to send
            content: The content of the message
            direct_peer: If specified, send only to this peer
        """
        message = Message(
            msg_type=msg_type,
            sender=self.nickname,
            content=content
        )
        
        # If we're not connected, queue the message for later
        if self.state != ConnectionState.CONNECTED:
            logger.warning(f"Not connected to network. Queuing message: {message.message_id}")
            self.message_queue.append(message)
            return
        
        # If the host is not initialized, queue the message
        if not self.host:
            logger.warning("Host not initialized. Queuing message.")
            self.message_queue.append(message)
            return
        
        message_json = message.to_json()
        message_bytes = message_json.encode('utf-8')
        
        sent_to_any = False
        
        # Try to send the message to all connected peers or a specific peer
        for peer_id in list(self.host.get_network().connections.keys()):
            if direct_peer and str(peer_id) != direct_peer:
                continue
                
            try:
                stream = await self.host.new_stream(peer_id, [self.protocol_id])
                await stream.write(message_bytes)
                await stream.close()
                sent_to_any = True
                logger.debug(f"Sent message to {peer_id}")
            except Exception as e:
                logger.error(f"Failed to send message to {peer_id}: {e}")
                # Remove the peer from our connected peers if we can't reach it
                if str(peer_id) in self.connected_peers:
                    self.connected_peers.remove(str(peer_id))
        
        if not sent_to_any:
            logger.warning("Couldn't send message to any peers. Queuing message.")
            self.message_queue.append(message)
    
    async def _process_message_queue(self) -> None:
        """Process queued messages when connection is restored."""
        while True:
            # Only process the queue when we're connected
            if self.state == ConnectionState.CONNECTED and self.message_queue:
                # Get the oldest message
                message = self.message_queue.pop(0)
                try:
                    # Try to send it
                    await self.send_message(message.msg_type, message.content)
                    logger.debug(f"Sent queued message: {message.message_id}")
                except Exception as e:
                    logger.error(f"Failed to send queued message: {e}")
                    # Put the message back at the end of the queue
                    self.message_queue.append(message)
            
            # Wait a bit before checking the queue again
            await asyncio.sleep(1)
    
    async def _send_periodic_heartbeats(self) -> None:
        """Send periodic heartbeats to keep connections alive."""
        while True:
            if self.state == ConnectionState.CONNECTED:
                try:
                    await self.send_message(
                        MessageType.HEARTBEAT,
                        f"Heartbeat from {self.nickname} at {time.time()}"
                    )
                    logger.debug("Sent heartbeat to peers")
                except Exception as e:
                    logger.error(f"Failed to send heartbeat: {e}")
            
            # Wait for some time before sending the next heartbeat
            # Typically heartbeats are sent every 30-60 seconds
            await asyncio.sleep(30)

# Connection configuration constants
DEFAULT_PROTOCOL = "/p2pchat/1.0.0"
DEFAULT_RECONNECT_ATTEMPTS = 5
DEFAULT_RECONNECT_DELAY = 5  # seconds

async def example_chat_handler(message: Message) -> None:
    """Example handler for chat messages."""
    print(f"[{message.sender}]: {message.content}")

async def example_transaction_handler(message: Message) -> None:
    """Example handler for transaction messages."""
    print(f"Transaction from {message.sender}: {message.content}")

async def run_client(nickname: str, bootstrap_peers: List[str]) -> None:
    """Run a P2P client with the given nickname and bootstrap peers."""
    # Create a new client
    client = Client(
        nickname=nickname,
        protocol_id=DEFAULT_PROTOCOL,
        bootstrap_peers=bootstrap_peers,
        max_reconnect_attempts=DEFAULT_RECONNECT_ATTEMPTS,
        reconnect_delay=DEFAULT_RECONNECT_DELAY
    )
    
    # Register message handlers
    client.register_handler(MessageType.CHAT, example_chat_handler)
    client.register_handler(MessageType.TRANSACTION, example_transaction_handler)
    
    try:
        # Start the client
        await client.start()
        
        # Example: send a chat message
        await client.send_message(
            MessageType.CHAT,
            "Hello, world! I've joined the network."
        )
        
        # Keep the client running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
    except Exception as e:
        logger.error(f"Error in client: {e}")
    finally:
        # Ensure the client is properly stopped
        await client.stop()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="P2P Client")
    parser.add_argument("--nickname", type=str, default=f"User-{uuid.uuid4().hex[:8]}", help="Your nickname in the network")
    parser.add_argument("--peers", type=str, nargs="+", help="Bootstrap peer multiaddresses")
    args = parser.parse_args()
    
    # Use default bootstrap peers if none provided
    bootstrap_peers = args.peers or [
        "/ip4/104.131.131.82/tcp/4001/p2p/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ",  # Example public bootstrap node
        "/dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN"   # Example DNS bootstrap node
    ]
    
    # Run the client
    try:
        asyncio.run(run_client(args.nickname, bootstrap_peers))
    except KeyboardInterrupt:
        print("Exiting...")
