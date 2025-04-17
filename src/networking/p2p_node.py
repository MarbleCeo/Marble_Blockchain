import socket
import threading
import pickle
import ssl
import time
import random
import logging
import ipaddress
import json
import queue
import hashlib
import uuid
from typing import List, Dict, Set, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from functools import wraps, lru_cache
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import upnpy
import stun
import requests

from blockchain import Blockchain
from src.ai.ia import RegenerativeDeepSeekAI
from src.blockchain.blockchain_core import Block, ProofOfHistory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("P2PNetwork")

# Constants
DEFAULT_STUN_SERVER = "stun.l.google.com:19302"
PEER_DISCOVERY_INTERVAL = 300  # seconds
HEALTH_CHECK_INTERVAL = 60  # seconds
MAX_RECONNECT_ATTEMPTS = 5
BACKOFF_FACTOR = 1.5
CONNECTION_TIMEOUT = 10  # seconds
MAX_CONNECTIONS = 50
SSL_CERT_FILE = "config/server.crt"
SSL_KEY_FILE = "config/server.key"
@dataclass
class PeerInfo:
    """Information about a connected peer"""
    address: Tuple[str, int]
    socket: Optional[socket.socket] = None
    last_seen: datetime = field(default_factory=datetime.now)
    trust_score: float = 0.5  # 0.0 to 1.0
    connection_attempts: int = 0
    is_active: bool = False
    capabilities: Dict[str, Any] = field(default_factory=dict)
    latency: float = 0.0  # in seconds
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __hash__(self) -> int:
        return hash(self.address)

class ConnectionPool:
    """Manages a pool of peer connections with limits and prioritization"""
    
    def __init__(self, max_connections: int = MAX_CONNECTIONS):
        self.max_connections: int = max_connections
        self._active_connections: Dict[Tuple[str, int], PeerInfo] = {}
        self._connection_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._lock: threading.RLock = threading.RLock()
        
    def add_connection(self, peer_info: PeerInfo) -> bool:
        """
        Add a connection to the pool
        
        Args:
            peer_info: Information about the peer to add
            
        Returns:
            bool: True if connection was added, False if pool is full
        """
        with self._lock:
            if len(self._active_connections) >= self.max_connections:
                # Queue connection based on trust score
                self._connection_queue.put((-peer_info.trust_score, peer_info))
                return False
                
            self._active_connections[peer_info.address] = peer_info
            return True
    
    def remove_connection(self, address: Tuple[str, int]) -> None:
        """Remove a connection from the pool"""
        with self._lock:
            if address in self._active_connections:
                del self._active_connections[address]
                
                # If queue has pending connections, add highest priority
                if not self._connection_queue.empty():
                    _, new_peer = self._connection_queue.get()
                    self._active_connections[new_peer.address] = new_peer
    
    def get_all_connections(self) -> List[PeerInfo]:
        """Get all active connections"""
        with self._lock:
            return list(self._active_connections.values())
    
    def get_connection(self, address: Tuple[str, int]) -> Optional[PeerInfo]:
        """Get a specific connection by address"""
        with self._lock:
            return self._active_connections.get(address)
    
    def update_connection(self, peer_info: PeerInfo) -> None:
        """Update information for an existing connection"""
        with self._lock:
            if peer_info.address in self._active_connections:
                self._active_connections[peer_info.address] = peer_info

class P2PNode:
    """
    Peer-to-peer network node with enhanced capabilities for blockchain networking.
    
    Features:
    - AI-driven peer management
    - Automatic IP/port detection and NAT traversal
    - Secure communication with SSL/TLS
    - Enhanced peer discovery and management
    - Connection pooling and backoff mechanisms
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 55555, 
                 auto_discover: bool = True, enable_ai: bool = True,
                 enable_ssl: bool = True, enable_upnp: bool = True):
        """
        Initialize a new P2P Node
        
        Args:
            host: Host address to bind to (0.0.0.0 for all interfaces)
            port: Port to listen on
            auto_discover: Whether to automatically discover peers
            enable_ai: Whether to use AI for peer management
            enable_ssl: Whether to use SSL/TLS for secure communication
            enable_upnp: Whether to use UPnP for port forwarding
        """
        self.host: str = host
        self.port: int = port
        self.public_ip: Optional[str] = None
        self.public_port: Optional[int] = None
        self.node_id: str = str(uuid.uuid4())
        self.blockchain = Blockchain()
        self.server_socket: Optional[socket.socket] = None
        self.ssl_context: Optional[ssl.SSLContext] = None
        
        # Connection management
        self.peers: List[socket.socket] = []  # For backward compatibility
        self.peer_pool: ConnectionPool = ConnectionPool(MAX_CONNECTIONS)
        self.known_peers: Set[Tuple[str, int]] = set()
        self.blacklisted_peers: Set[Tuple[str, int]] = set()
        
        # Thread management
        self.running: bool = False
        self.thread_pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=10)
        self._lock: threading.RLock = threading.RLock()
        
        # Features
        self.auto_discover: bool = auto_discover
        self.enable_ai: bool = enable_ai
        self.enable_ssl: bool = enable_ssl
        self.enable_upnp: bool = enable_upnp
        
        # AI integration
        self.ai_agent: Optional[RegenerativeDeepSeekAI] = None
        if enable_ai:
            self._setup_ai_agent()
            
        # Setup secure communication if enabled
        if enable_ssl:
            self._setup_ssl()
    def _setup_ai_agent(self) -> None:
        """Initialize the AI agent for peer management"""
        try:
            self.ai_agent = RegenerativeDeepSeekAI()
            logger.info("AI agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI agent: {e}")
            self.enable_ai = False
    
    def _setup_ssl(self) -> None:
        """Set up SSL/TLS context for secure communication"""
        try:
            self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            self.ssl_context.load_cert_chain(certfile=SSL_CERT_FILE, keyfile=SSL_KEY_FILE)
            logger.info("SSL context initialized successfully")
        except FileNotFoundError:
            logger.warning(f"SSL certificate files not found. Creating self-signed certificate...")
            self._create_self_signed_cert()
        except Exception as e:
            logger.error(f"Failed to initialize SSL: {e}")
            self.enable_ssl = False
    
    def _create_self_signed_cert(self) -> None:
        """Create a self-signed certificate for SSL/TLS"""
        try:
            from OpenSSL import crypto
            import os
            
            # Create directory if it doesn't exist
            os.makedirs("config", exist_ok=True)
            
            # Create a key pair
            k = crypto.PKey()
            k.generate_key(crypto.TYPE_RSA, 2048)
            
            # Create a self-signed cert
            cert = crypto.X509()
            cert.get_subject().C = "US"
            cert.get_subject().ST = "State"
            cert.get_subject().L = "City"
            cert.get_subject().O = "Organization"
            cert.get_subject().OU = "Organizational Unit"
            cert.get_subject().CN = "localhost"
            cert.set_serial_number(1000)
            cert.gmtime_adj_notBefore(0)
            cert.gmtime_adj_notAfter(10*365*24*60*60)  # 10 years
            cert.set_issuer(cert.get_subject())
            cert.set_pubkey(k)
            cert.sign(k, 'sha256')
            
            # Save cert and key
            with open(SSL_CERT_FILE, "wb") as f:
                f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
            with open(SSL_KEY_FILE, "wb") as f:
                f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
                
            # Reload SSL context
            self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            self.ssl_context.load_cert_chain(certfile=SSL_CERT_FILE, keyfile=SSL_KEY_FILE)
            logger.info("Self-signed certificate created successfully")
        except Exception as e:
            logger.error(f"Failed to create self-signed certificate: {e}")
            self.enable_ssl = False
    
    def _detect_public_ip(self) -> Tuple[Optional[str], Optional[int]]:
        """
        Detect the node's public IP address and port using STUN protocol
        
        Returns:
            Tuple[Optional[str], Optional[int]]: Public IP and port, or None if detection fails
        """
        try:
            nat_type, external_ip, external_port = stun.get_ip_info(
                source_ip=self.host if self.host != "0.0.0.0" else None,
                source_port=self.port,
                stun_host=DEFAULT_STUN_SERVER.split(':')[0],
                stun_port=int(DEFAULT_STUN_SERVER.split(':')[1])
            )
            logger.info(f"NAT type: {nat_type}, Public IP: {external_ip}, Public port: {external_port}")
            return external_ip, external_port
        except Exception as e:
            logger.error(f"Failed to detect public IP: {e}")
            # Fallback to other methods
            try:
                response = requests.get('https://api.ipify.org?format=json')
                external_ip = response.json()['ip']
                logger.info(f"Public IP detected via ipify: {external_ip}")
                return external_ip, None  # Still don't know the port
            except Exception as e:
                logger.error(f"Failed to detect public IP via ipify: {e}")
                return None, None
    
    def _setup_port_forwarding(self) -> bool:
        """
        Set up port forwarding using UPnP
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enable_upnp:
            return False
            
        try:
            upnp = upnpy.UPnP()
            devices = upnp.discover()
            
            if not devices:
                logger.warning("No UPnP devices found")
                return False
                
            # Get the first IGD (Internet Gateway Device)
            device = upnp.get_igd()
            service = device.get_service("WANIPConnection") or device.get_service("WANPPPConnection")
            
            if not service:
                logger.warning("No WANIPConnection or WANPPPConnection service found")
                return False
            
            # Add port mapping
            service.add_port_mapping(
                NewRemoteHost="",
                NewExternalPort=self.port,
                NewProtocol="TCP",
                NewInternalPort=self.port,
                NewInternalClient=self.host if self.host != "0.0.0.0" else socket.gethostbyname(socket.gethostname()),
                NewEnabled=1,
                NewPortMappingDescription=f"P2PNode-{self.node_id}",
                NewLeaseDuration=0
            )
            logger.info(f"UPnP port forwarding set up for port {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to set up UPnP port forwarding: {e}")
            return False

    def start(self) -> None:
        """
        Start the P2P node server and related services
        """
        try:
            # Mark as running
            self.running = True
            
            # Try to detect public IP and set up port forwarding
            if self.auto_discover:
                self.public_ip, self.public_port = self._detect_public_ip()
                self._setup_port_forwarding()
            
            # Initialize server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            logger.info(f"P2P Node server started on {self.host}:{self.port}")
            
            # Wrap server socket with SSL if enabled
            if self.enable_ssl and self.ssl_context:
                self.server_socket = self.ssl_context.wrap_socket(
                    self.server_socket, server_side=True
                )
                logger.info("SSL encryption enabled for connections")
            
            # Store start time for uptime calculation
            self._start_time = time.time()
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=self._network_monitoring_loop, daemon=True)
            monitor_thread.start()
            
            # Start auto peer discovery if enabled
            if self.auto_discover:
                discover_thread = threading.Thread(target=self.discover_peers, daemon=True)
                discover_thread.start()
            
            # Start accepting connections
            accept_thread = threading.Thread(target=self.accept_connections, daemon=True)
            accept_thread.start()
            
            logger.info(f"P2P Node with ID {self.node_id} is fully operational")
            
    def accept_connections(self) -> None:
        """Accept incoming connections from peers"""
        logger.info("Starting connection acceptance loop")
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                logger.info(f'Connected by {addr}')
                
                # Create peer info object
                peer_info = PeerInfo(
                    address=addr,
                    socket=client_socket,
                    last_seen=datetime.now(),
                    is_active=True
                )
                
                # Add to connection pool
                if addr not in self.blacklisted_peers and self.peer_pool.add_connection(peer_info):
                    # For backward compatibility
                    self.peers.append(client_socket)
                    
                    # Start client handler thread
                    self.thread_pool.submit(self.handle_client, client_socket, addr)
                else:
                    logger.warning(f"Connection from {addr} rejected (blacklisted or pool full)")
                    client_socket.close()
            except ssl.SSLError as e:
                logger.warning(f"SSL error during connection: {e}")
            except socket.error as e:
                if self.running:  # Only log if we're still supposed to be running
                    logger.error(f"Socket error in accept_connections: {e}")
            except Exception as e:
                logger.error(f"Error accepting connection: {e}")
                if not self.running:
                    break

    def handle_client(self, client_socket: socket.socket, addr: Tuple[str, int]) -> None:
        """
        Handle communication with a connected client
        
        Args:
            client_socket: Socket connection to the client
            addr: Client address (IP, port)
        """
        peer_info = self.peer_pool.get_connection(addr)
        
        while self.running and peer_info and peer_info.is_active:
            try:
                # Set a timeout to prevent hanging
                client_socket.settimeout(CONNECTION_TIMEOUT)
                
                # Receive data
                data = client_socket.recv(4096)
                if not data:
                    logger.info(f"Connection closed by {addr}")
                    break
                
                # Update last seen timestamp
                if peer_info:
                    peer_info.last_seen = datetime.now()
                    self.peer_pool.update_connection(peer_info)
                
                # Process encrypted data if SSL is enabled
                if self.enable_ssl:
                    try:
                        data = self._handle_secure_message(data, addr)
                    except Exception as e:
                        logger.error(f"Error handling secure message from {addr}: {e}")
                        continue
                
                # Deserialize the data
                try:
                    data = pickle.loads(data)
                    request_type = data.get('request_type')

                    if request_type == 'sync_chain':
                        # Synchronize blockchain with peer
                        self.sync_chain(client_socket)
                        
                    elif request_type == 'add_transaction':
                        transaction = data['transaction']
                        logger.info(f"Received transaction from {addr}")
                        self.blockchain.create_transaction(transaction)
                        
                    elif request_type == 'mine_block':
                        miner_address = data['miner_address']
                        logger.info(f"Received mine request from {addr} for {miner_address}")
                        # Use PoH+PoW for mining from blockchain_core
                        self.blockchain.mine_pending_transactions(miner_address)
                        
                    elif request_type == 'get_balance':
                        address = data['address']
                        balance = self.blockchain.get_balance_of_address(address)
                        self._send_secure_data(client_socket, {
                            'response_type': 'get_balance', 
                            'balance': balance
                        })
                        
                    elif request_type == 'discover_peers':
                        # Send known peers list
                        peer_list = [
                            {'host': p.address[0], 'port': p.address[1], 'node_id': p.node_id} 
                            for p in self.peer_pool.get_all_connections()
                        ]
                        self._send_secure_data(client_socket, {
                            'response_type': 'peer_list',
                            'peers': peer_list
                        })
                        
                    elif request_type == 'peer_health_check':
                        # Respond to health check
                        metrics = {
                            'uptime': time.time() - self._start_time,
                            'peers_count': len(self.peer_pool.get_all_connections()),
                            'blockchain_height': len(self.blockchain.chain),
                            'pending_tx': len(self.blockchain.pending_transactions)
                        }
                        self._send_secure_data(client_socket, {
                            'response_type': 'health_metrics',
                            'metrics': metrics,
                            'node_id': self.node_id
                        })
                except (pickle.PickleError, KeyError) as e:
                    logger.warning(f"Error processing message from {addr}: {e}")
                    continue
            except socket.timeout:
                logger.debug(f"Connection to {addr} timed out, checking if still active")
                # Check if we should keep the connection
                if peer_info and (datetime.now() - peer_info.last_seen) > timedelta(seconds=CONNECTION_TIMEOUT * 2):
                    logger.info(f"Connection to {addr} considered stale, closing")
                    break
            except ConnectionResetError:
                logger.info(f"Connection reset by {addr}")
                break
            except Exception as e:
                logger.error(f"Error handling client {addr}: {e}")
                break

        # Clean up connection
        try:
            client_socket.close()
            if addr in [p.address for p in self.peers]:
                self.peers = [p for p in self.peers if getattr(p, 'getpeername', lambda: None)() != addr]
            if peer_info:
                peer_info.is_active = False
                self.peer_pool.update_connection(peer_info)
                self.peer_pool.remove_connection(addr)
            logger.info(f"Connection to {addr} closed and removed from pool")
        except Exception as e:
            logger.error(f"Error cleaning up connection to {addr}: {e}")
    
    def _network_monitoring_loop(self) -> None:
        """
        Continuously monitor the health of the network and peers
        """
        self._start_time = time.time()
        logger.info("Starting network monitoring loop")
        
        while self.running:
            try:
                # Perform health checks on peers
                self._health_check_peers()
                
                # Log network statistics
                active_peers = self.peer_pool.get_all_connections()
                logger.info(f"Network stats: {len(active_peers)} active peers, " +
                           f"{len(self.blacklisted_peers)} blacklisted peers")
                
                # If using AI, analyze network data
                if self.enable_ai and self.ai_agent:
                    try:
                        peer_data = [{
                            'address': str(p.address),
                            'latency': p.latency,
                            'trust_score': p.trust_score,
                            'last_seen': (datetime.now() - p.last_seen).total_seconds(),
                            'capabilities': p.capabilities
                        } for p in active_peers]
                        
                        optimization = self.ai_agent.analyze_network_health(peer_data)
                        
                        # Apply AI recommendations
                        if optimization and 'blacklist' in optimization:
                            for peer_addr in optimization['blacklist']:
                                addr = tuple(peer_addr.split(':'))
                                self.blacklisted_peers.add(addr)
                                logger.info(f"AI recommended blacklisting peer {addr}")
                    except Exception as e:
                        logger.error(f"Error in AI network analysis: {e}")
                
                # Sleep between monitoring cycles
                time.sleep(HEALTH_CHECK_INTERVAL)
            except Exception as e:
                logger.error(f"Error in network monitoring loop: {e}")
                time.sleep(10)  # Shorter sleep on error
    
    def _health_check_peers(self) -> None:
        """
        Check the health of all connected peers
        """
        peers = self.peer_pool.get_all_connections()
        for peer in peers:
            # Skip inactive peers
            if not peer.is_active:
                continue
                
            try:
                # Measure latency
                start_time = time.time()
                
                # Send health check message
                if peer.socket:
                    self._send_secure_data(peer.socket, {
                        'request_type': 'peer_health_check',
                        'timestamp': time.time(),
                        'node_id': self.node_id
                    })
                    
                    # Wait for response with timeout
                    peer.socket.settimeout(5.0)
                    response = peer.socket.recv(4096)
                    
                    if response:
                        # Update latency
                        peer.latency = time.time() - start_time
                        
                        # Process response
                        if self.enable_ssl:
                            response = self._handle_secure_message(response, peer.address)
                        
                        response_data = pickle.loads(response)
                        if response_data.get('response_type') == 'health_metrics':
                            # Update peer capabilities based on metrics
                            peer.capabilities = response_data.get('metrics', {})
                            
                            # Update trust score based on response quality
                            if all(k in peer.capabilities for k in ['uptime', 'peers_count', 'blockchain_height']):
                                peer.trust_score = min(1.0, peer.trust_score + 0.05)
                            
                            # Update peer info
                            peer.last_seen = datetime.now()
                            self.peer_pool.update_connection(peer)
                            
                            logger.debug(f"Health check for {peer.address} successful. Latency: {peer.latency:.4f}s")
            except (socket.timeout, ConnectionResetError):
                # Failed health check
                peer.connection_attempts += 1
                peer.trust_score = max(0.0, peer.trust_score - 0.1)
                
                # Disconnect if too many failed attempts
                if peer.connection_attempts > MAX_RECONNECT_ATTEMPTS:
                    logger.warning(f"Peer {peer.address} failed too many health checks, disconnecting")
                    peer.is_active = False
                    if peer.socket:
                        try:
                            peer.socket.close()
                        except Exception as e:
                            logger.error(f"Error closing socket for peer {peer.address}: {e}")
                    self.peer_pool.remove_connection(peer.address)
                    
            except Exception as e:
                logger.error(f"Error during health check for peer {peer.address}: {e}")
                
                # Update trust score negatively on error
                peer.trust_score = max(0.0, peer.trust_score - 0.05)
                self.peer_pool.update_connection(peer)

    def sync_chain(self, client_socket):
        # Receba a cadeia do cliente
        data = client_socket.recv(4096)
        data = pickle.loads(data)

        # Compare as cadeias e atualize a cadeia local com a mais longa
        client_chain = data['chain']
        if len(client_chain) > len(self.blockchain.chain):
            self.blockchain.chain = client_chain

    def connect_to_peer(self, host, port):
        # Conecte-se a um par e sincronize a blockchain
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        client_socket.sendall(pickle.dumps({'request_type': 'sync_chain'}))
        data = client_socket.recv(4096)
        data = pickle.loads(data)
        self.sync_chain(client_socket)
        client_socket.close()

    def add_transaction(self, transaction):
        # Adicione uma transação à blockchain local e propague para os pares
        self.blockchain.create_transaction(transaction)
        for peer in self.peers:
            peer.sendall(pickle.dumps({'request_type': 'add_transaction', 'transaction': transaction}))

    def mine_block(self, miner_address):
        # Mine um bloco na blockchain local e propague para os pares
        self.blockchain.mine_pending_transactions(miner_address)
        for peer in self.peers:
            peer.sendall(pickle.dumps({'request_type': 'mine_block', 'miner_address': miner_address}))

    def get_balance(self, address):
        # Solicite o saldo à blockchain local e retorne o resultado
        balance = self.blockchain.get_balance_of_address(address)
        balance = self.blockchain.get_balance_of_address(address)
        return balance
        
    def _handle_secure_message(self, data: bytes, addr: Tuple[str, int]) -> bytes:
        """
        Handles secure message decryption and verification
        
        Args:
            data: Encrypted data received from a peer
            addr: Address (IP, port) of the sender
            
        Returns:
            bytes: Decrypted data
            
        Raises:
            ValueError: If the message fails verification
            ssl.SSLError: If decryption fails
        """
        try:
            # Check if we have a valid SSL context
            if not self.enable_ssl or not self.ssl_context:
                return data
                
            # Extract message parts
            # Format: [HMAC (32 bytes)][IV (16 bytes)][Encrypted Data]
            if len(data) < 48:  # Minimum size for a secure message
                raise ValueError("Message too short to be a valid secure message")
                
            hmac_received = data[:32]
            iv = data[32:48]
            ciphertext = data[48:]
            
            # Verify HMAC
            peer_info = self.peer_pool.get_connection(addr)
            if peer_info and hasattr(peer_info, 'session_key'):
                # Use the session key to verify HMAC
                h = hashlib.sha256()
                h.update(iv + ciphertext + peer_info.session_key)
                hmac_calculated = h.digest()
                
                if not hmac.compare_digest(hmac_received, hmac_calculated):
                    raise ValueError("HMAC verification failed")
                
            # Decrypt data using SSL context
            decrypted = self.ssl_context.unwrap_socket(io.BytesIO(ciphertext))
            return decrypted.read()
            
        except (ssl.SSLError, ValueError) as e:
            logger.warning(f"Secure message handling error from {addr}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in secure message handling from {addr}: {e}")
            raise ValueError(f"Message processing error: {str(e)}")
    
    def _send_secure_data(self, client_socket: socket.socket, data: Any) -> bool:
        """
        Securely send data to a peer
        
        Args:
            client_socket: Socket connection to the peer
            data: Data to send (will be pickled)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Serialize the data
            serialized_data = pickle.dumps(data)
            
            # If SSL is enabled, encrypt the data
            if self.enable_ssl and self.ssl_context:
                try:
                    # Get peer address for session key lookup
                    addr = client_socket.getpeername()
                    peer_info = self.peer_pool.get_connection(addr)
                    
                    # Generate random IV
                    iv = os.urandom(16)
                    
                    # Encrypt data
                    bio = io.BytesIO()
                    ssl_sock = self.ssl_context.wrap_bio(
                        io.BytesIO(serialized_data), io.BytesIO(), server_side=True
                    )
                    while True:
                        try:
                            bio.write(ssl_sock.read())
                        except ssl.SSLWantReadError:
                            break
                    
                    ciphertext = bio.getvalue()
                    
                    # Generate HMAC if we have a session key
                    if peer_info and hasattr(peer_info, 'session_key'):
                        h = hashlib.sha256()
                        h.update(iv + ciphertext + peer_info.session_key)
                        hmac_tag = h.digest()
                    else:
                        # Fallback to a simpler integrity check if no session key
                        h = hashlib.sha256()
                        h.update(iv + ciphertext)
                        hmac_tag = h.digest()
                    
                    # Combine HMAC, IV, and ciphertext
                    encrypted_data = hmac_tag + iv + ciphertext
                    
                    # Send the data
                    client_socket.sendall(encrypted_data)
                    return True
                    
                except Exception as e:
                    logger.error(f"Error in secure data transmission: {e}")
                    # Fall back to sending unencrypted if encryption fails
                    client_socket.sendall(serialized_data)
                    return True
            else:
                # Send unencrypted data
                client_socket.sendall(serialized_data)
                return True
                
        except (socket.error, pickle.PickleError) as e:
            logger.error(f"Socket or serialization error when sending data: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in sending data: {e}")
            return False
    
    def discover_peers(self) -> None:
        """
        Continuously discover new peers through multiple methods:
        1. Query known peers for their peer lists
        2. Use DNS discovery service if available
        3. Connect to seed nodes
        4. Use UPnP to discover local network nodes
        
        This method runs as a background thread.
        """
        logger.info("Starting peer discovery service")
        
        # Seed nodes - hardcoded bootstrap nodes to connect to initially
        seed_nodes = [
            ('bootstrap1.blockchain-network.org', 55555),
            ('bootstrap2.blockchain-network.org', 55555),
            ('bootstrap3.blockchain-network.org', 55555)
        ]
        
        # Discovery services
        discovery_services = [
            'https://discovery.blockchain-network.org/peers',
            'https://backup-discovery.blockchain-network.org/peers'
        ]
        
        while self.running:
            try:
                discovered_peers = set()
                
                # Method 1: Query our existing peers for their peer lists
                for peer_info in self.peer_pool.get_all_connections():
                    if not peer_info.is_active or not peer_info.socket:
                        continue
                        
                    try:
                        # Request peer list
                        self._send_secure_data(peer_info.socket, {
                            'request_type': 'discover_peers'
                        })
                        
                        # Set a short timeout for response
                        peer_info.socket.settimeout(5.0)
                        response = peer_info.socket.recv(4096)
                        
                        if response:
                            if self.enable_ssl:
                                response = self._handle_secure_message(response, peer_info.address)
                                
                            response_data = pickle.loads(response)
                            if response_data.get('response_type') == 'peer_list':
                                for peer in response_data.get('peers', []):
                                    if 'host' in peer and 'port' in peer:
                                        discovered_peers.add((peer['host'], peer['port']))
                    except Exception as e:
                        logger.debug(f"Error getting peer list from {peer_info.address}: {e}")
                
                # Method 2: Use discovery services
                if self.enable_ai:  # Advanced discovery uses AI capabilities
                    for service_url in discovery_services:
                        try:
                            response = requests.get(service_url, timeout=10)
                            if response.status_code == 200:
                                peers_data = response.json()
                                for peer in peers_data.get('peers', []):
                                    if 'host' in peer and 'port' in peer:
                                        discovered_peers.add((peer['host'], int(peer['port'])))
                        except Exception as e:
                            logger.debug(f"Error querying discovery service {service_url}: {e}")
                
                # Method 3: Try seed nodes if we have few peers
                if len(self.peer_pool.get_all_connections()) < 3:
                    for seed_host, seed_port in seed_nodes:
                        discovered_peers.add((seed_host, seed_port))
                
                # Method 4: Use UPnP for local network discovery
                if self.enable_upnp:
                    try:
                        upnp = upnpy.UPnP()
                        devices = upnp.discover(timeout=2)
                        
                        for device in devices:
                            if 'P2PNode' in device.friendly_name:
                                # Extract host and port from device info
                                device_url = device.presentation_url
                                parsed_url = urllib.parse.urlparse(device_url)
                                host = parsed_url.hostname
                                port = 55555  # Assume default port
                                
                                if host:
                                    discovered_peers.add((host, port))
                    except Exception as e:
                        logger.debug(f"UPnP discovery error: {e}")
                
                # Connect to all discovered peers
                for host, port in discovered_peers:
                    try:
                        # Skip if it's our own address
                        if (self.host == host or self.host == "0.0.0.0") and self.port == port:
                            continue
                            
                        # Skip already connected or blacklisted peers
                        addr = (host, port)
                        if addr in [(p.address[0], p.address[1]) for p in self.peer_pool.get_all_connections()] or addr in self.blacklisted_peers:
                            continue
                            
                        # Attempt to connect
                        logger.info(f"Discovered new peer: {host}:{port}, attempting to connect")
                        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        client_socket.settimeout(CONNECTION_TIMEOUT)
                        
                        # Wrap with SSL if enabled
                        if self.enable_ssl and self.ssl_context:
                            client_socket = self.ssl_context.wrap_socket(client_socket)
                            
                        client_socket.connect((host, port))
                        
                        # Create peer info and add to pool
                        peer_info = PeerInfo(
                            address=(host, port),
                            socket=client_socket,
                            last_seen=datetime.now(),
                            is_active=True
                        )
                        
                        if self.peer_pool.add_connection(peer_info):
                            # For backward compatibility
                            self.peers.append(client_socket)
                            
                            # Start client handler thread
                            self.thread_pool.submit(self.handle_client, client_socket, (host, port))
                            
                            # Sync blockchain
                            self._send_secure_data(client_socket, {'request_type': 'sync_chain'})
                    except Exception as e:
                        logger.debug(f"Failed to connect to discovered peer {host}:{port}: {e}")
                
                # Sleep between discovery cycles
                time.sleep(PEER_DISCOVERY_INTERVAL)
            except Exception as e:
                logger.error(f"Error in peer discovery: {e}")
                time.sleep(60)  # Wait longer after error
    
    def broadcast_message(self, message: Dict[str, Any], exclude: Optional[List[Tuple[str, int]]] = None) -> int:
        """
        Broadcast a message to all connected peers
        
        Args:
            message: Message to broadcast (will be pickled)
            exclude: List of addresses to exclude from broadcast
            
        Returns:
            int: Number of peers the message was successfully sent to
        """
        if exclude is None:
            exclude = []
            
        success_count = 0
        peers = self.peer_pool.get_all_connections()
        
        # Use AI to optimize broadcast if enabled
        if self.enable_ai and self.ai_agent and len(peers) > 10:
            try:
                peer_data = [{
                    'address': str(p.address),
                    'latency': p.latency,
    node.start()