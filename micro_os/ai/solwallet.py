"""
Solana Wallet Implementation for MicroOS

This module provides a secure wallet implementation for the Solana blockchain,
offering features like wallet creation, transaction handling, balance checking,
and key management. The implementation is designed to integrate with the
agent system defined in IA.py.

Classes:
    SolWallet: Core wallet functionality for Solana blockchain operations
    TransactionError: Custom exception for transaction-related errors
    WalletError: Custom exception for wallet-related errors

Functions:
    estimate_transaction_fee: Calculate the expected fee for a transaction
    get_network_status: Check Solana network health and parameters
    format_sol_amount: Convert lamports to SOL with proper formatting
"""

import asyncio
import base58
import base64
import json
import logging
import os
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

try:
    import nacl.signing
    from nacl.public import PrivateKey, PublicKey, Box
    HAS_NACL = True
except ImportError:
    HAS_NACL = False

try:
    from solana.rpc.api import Client
    from solana.rpc.types import TxOpts
    from solana.keypair import Keypair
    from solana.publickey import PublicKey as SolPublicKey
    from solana.transaction import Transaction, TransactionInstruction, AccountMeta
    from solana.system_program import TransferParams, transfer
    from spl.token.client import Token
    from spl.token.constants import TOKEN_PROGRAM_ID
    HAS_SOLANA = True
except ImportError:
    HAS_SOLANA = False

# Configure logging
logger = logging.getLogger(__name__)

# Constants
LAMPORTS_PER_SOL = 1_000_000_000  # 10^9 lamports = 1 SOL
DEFAULT_ENDPOINTS = {
    "mainnet": "https://api.mainnet-beta.solana.com",
    "testnet": "https://api.testnet.solana.com",
    "devnet": "https://api.devnet.solana.com",
    "localhost": "http://localhost:8899",
}
DEFAULT_NETWORK = "devnet"


class NetworkType(Enum):
    """Supported Solana network types."""
    MAINNET = "mainnet"
    TESTNET = "testnet"
    DEVNET = "devnet"
    LOCALHOST = "localhost"


class TransactionError(Exception):
    """Exception raised for transaction-related errors."""
    pass


class WalletError(Exception):
    """Exception raised for wallet-related errors."""
    pass


class TokenAccountInfo:
    """Information about a token account."""
    
    def __init__(self, mint: str, owner: str, amount: int, decimals: int = 9):
        self.mint = mint
        self.owner = owner
        self.amount = amount
        self.decimals = decimals
    
    @property
    def formatted_amount(self) -> str:
        """Return the token amount with proper decimal formatting."""
        return format_token_amount(self.amount, self.decimals)
    
    def __repr__(self) -> str:
        return f"TokenAccountInfo(mint={self.mint}, amount={self.formatted_amount})"


class SolWallet:
    """
    Solana wallet implementation providing key management, transactions,
    and balance operations integrated with the MicroOS agent system.
    
    This class handles the creation, loading, and management of Solana wallets,
    along with transaction functionality, balance checking, and integration
    with the IntelligentAgent system.
    
    Attributes:
        keypair (Keypair): The Solana keypair for this wallet
        network (NetworkType): The Solana network this wallet connects to
        client (Client): RPC client for Solana network communication
        transaction_history (List): Historical transactions for this wallet
        default_fee_payer (Optional[Keypair]): Keypair used to pay transaction fees
    """
    
    def __init__(
        self,
        keypair: Optional[Union[Keypair, str, bytes, List[int]]] = None,
        network: Union[NetworkType, str] = DEFAULT_NETWORK,
        endpoint: Optional[str] = None,
        wallet_path: Optional[Union[str, Path]] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize a Solana wallet instance.
        
        Args:
            keypair: Existing keypair or seed bytes for the wallet
            network: Solana network to connect to (mainnet, testnet, devnet, localhost)
            endpoint: Custom RPC endpoint URL (overrides network parameter)
            wallet_path: Path to encrypted wallet file for loading
            password: Password for wallet decryption
            
        Raises:
            ImportError: If required dependencies are missing
            WalletError: If wallet initialization fails
        """
        if not HAS_NACL or not HAS_SOLANA:
            raise ImportError(
                "Required dependencies not installed. Please install PyNaCl and Solana packages: "
                "pip install pynacl solana-sdk spl-token"
            )
        
        # Set up network connection
        if isinstance(network, str):
            try:
                self.network = NetworkType(network)
            except ValueError:
                self.network = NetworkType.DEVNET
                logger.warning(f"Invalid network '{network}', using {self.network.value} instead")
        else:
            self.network = network
            
        self.endpoint = endpoint or DEFAULT_ENDPOINTS[self.network.value]
        self.client = Client(self.endpoint)
        
        # Initialize wallet components
        self.default_fee_payer = None
        self.transaction_history = []
        
        # Set up wallet from parameters
        self._load_or_create_wallet(keypair, wallet_path, password)
        
        # Validate connection
        try:
            self.client.get_version()
            logger.info(f"Connected to Solana {self.network.value} at {self.endpoint}")
        except Exception as e:
            logger.warning(f"Unable to connect to Solana network: {str(e)}")
    
    def _load_or_create_wallet(
        self,
        keypair: Optional[Union[Keypair, str, bytes, List[int]]],
        wallet_path: Optional[Union[str, Path]],
        password: Optional[str],
    ) -> None:
        """
        Load a wallet from file or create a new one from provided keypair.
        
        Args:
            keypair: Keypair or seed for wallet creation
            wallet_path: Path to encrypted wallet file
            password: Password for wallet decryption
            
        Raises:
            WalletError: If wallet loading or creation fails
        """
        if wallet_path:
            self._load_wallet_from_file(wallet_path, password)
        elif keypair:
            self._initialize_from_keypair(keypair)
        else:
            # Create a new random keypair
            self.keypair = Keypair()
            logger.info(f"Created new wallet with address: {self.address}")
    
    def _initialize_from_keypair(self, keypair: Union[Keypair, str, bytes, List[int]]) -> None:
        """Initialize wallet from an existing keypair or seed."""
        try:
            if isinstance(keypair, Keypair):
                self.keypair = keypair
            elif isinstance(keypair, str):
                # Handle base58 encoded private key
                if len(keypair) == 88 and keypair.startswith('5'):
                    # This looks like a base58 encoded private key
                    seed = base58.b58decode(keypair)[:32]
                    self.keypair = Keypair.from_seed(seed)
                else:
                    # Try as seed phrase (not implemented - would use BIP39)
                    raise NotImplementedError("Seed phrase to keypair conversion is not implemented")
            elif isinstance(keypair, bytes):
                self.keypair = Keypair.from_seed(keypair[:32])
            elif isinstance(keypair, list) and all(isinstance(x, int) for x in keypair):
                self.keypair = Keypair.from_seed(bytes(keypair[:32]))
            else:
                raise WalletError(f"Invalid keypair format: {type(keypair)}")
            
            logger.info(f"Initialized wallet with address: {self.address}")
        
        except Exception as e:
            raise WalletError(f"Failed to initialize keypair: {str(e)}")

    def _load_wallet_from_file(self, wallet_path: Union[str, Path], password: Optional[str]) -> None:
        """
        Load wallet from an encrypted wallet file.
        
        Args:
            wallet_path: Path to the wallet file
            password: Decryption password
            
        Raises:
            WalletError: If wallet loading fails
        """
        try:
            path = Path(wallet_path)
            if not path.exists():
                raise WalletError(f"Wallet file not found: {path}")
                
            if not password:
                raise WalletError("Password is required to decrypt wallet")
                
            # Load encrypted wallet
            with open(path, 'r') as f:
                encrypted_data = json.load(f)
                
            # Decrypt wallet (simplified - in a real implementation, use a proper encryption scheme)
            salt = base64.b64decode(encrypted_data['salt'])
            nonce = base64.b64decode(encrypted_data['nonce'])
            encrypted_bytes = base64.b64decode(encrypted_data['encrypted_keypair'])
            
            # Derive key from password and salt (simplified)
            key = nacl.hash.sha256(password.encode() + salt).digest()
            box = nacl.secret.SecretBox(key)
            
            # Decrypt the keypair data
            keypair_bytes = box.decrypt(encrypted_bytes)
            
            # Create keypair from decrypted data
            self.keypair = Keypair.from_seed(keypair_bytes[:32])
            logger.info(f"Wallet loaded from file with address: {self.address}")
            
        except Exception as e:
            raise WalletError(f"Failed to load wallet from file: {str(e)}")

    def save_wallet(self, wallet_path: Union[str, Path], password: str) -> None:
        """
        Save wallet to encrypted file.
        
        Args:
            wallet_path: Path to save the wallet file
            password: Encryption password
            
        Raises:
            WalletError: If wallet saving fails
        """
        try:
            path = Path(wallet_path)
            os.makedirs(path.parent, exist_ok=True)
            
            # Create salt and derive key
            salt = os.urandom(16)
            key = nacl.hash.sha256(password.encode() + salt).digest()
            
            # Encrypt the keypair
            box = nacl.secret.SecretBox(key)
            nonce = os.urandom(box.NONCE_SIZE)
            encrypted = box.encrypt(bytes(self.keypair.seed), nonce)
            
            # Prepare data for storage
            wallet_data = {
                'version': 1,
                'salt': base64.b64encode(salt).decode('utf-8'),
                'nonce': base64.b64encode(nonce).decode('utf-8'),
                'encrypted_keypair': base64.b64encode(encrypted).decode('utf-8'),
                'public_key': str(self.address),
                'network': self.network.value,
                'created_at': datetime.now().isoformat(),
            }
            
            # Save to file
            with open(path, 'w') as f:
                json.dump(wallet_data, f, indent=2)
                
            logger.info(f"Wallet saved to {path}")
            
        except Exception as e:
            raise WalletError(f"Failed to save wallet: {str(e)}")
    
    @property
    def address(self) -> str:
        """Return the wallet's public key address."""
        return str(self.keypair.public_key)
    
    @property
    def public_key(self) -> SolPublicKey:
        """Return the wallet's public key object."""
        return self.keypair.public_key
    
    async def get_balance(self) -> int:
        """
        Get the wallet's SOL balance in lamports.
        
        Returns:
            int: Balance in lamports (1 SOL = 10^9 lamports)
            
        Raises:
            TransactionError: If balance check fails
        """
        try:
            response = await self.client.get_balance(self.public_key)
            if response["result"]["value"] is not None:
                return response["result"]["value"]
            else:
                raise TransactionError("Invalid balance response")
        except Exception as e:
            logger.error(f"Failed to get balance: {str(e)}")
            raise TransactionError(f"Balance check failed: {str(e)}")
    
    async def get_formatted_balance(self) -> str:
        """
        Get the wallet's SOL balance formatted with proper units.
        
        Returns:
            str: Formatted balance string (e.g., "1.234 SOL")
        """
        lamports = await self.get_balance()
        return format_sol_amount(lamports)
    
    async def get_token_accounts(self) -> List[TokenAccountInfo]:
        """
        Get all token accounts owned by this wallet.
        
        Returns:
            List[TokenAccountInfo]: List of token account information
            
        Raises:
            TransactionError: If token account fetching fails
        """
        try:
            response = await self.client.get_token_accounts_by_owner(
                self.public_key,
                {'programId': TOKEN_PROGRAM_ID}
            )
            
            accounts = []
            for item in response["result"]["value"]:
                account_data = base64.b64decode(item["account"]["data"][0])
                # Parse token account data according to SPL Token schema
                # This is simplified - actual implementation needs proper binary parsing
                mint = str(SolPublicKey(account_data[0:32]))
                owner = str(SolPublicKey(account_data[32:64]))
                amount = int.from_bytes(account_data[64:72], byteorder='little')
                decimals = account_data[72]
                
                accounts.append(TokenAccountInfo(mint, owner, amount, decimals))
                
            return accounts
            
        except Exception as e:
            logger.error(f"Failed to get token accounts: {str(e)}")
            raise TransactionError(f"Token account fetch failed: {str(e)}")
    
    async def transfer_sol(
        self,
        recipient: Union[str, SolPublicKey],
        amount: Union[int, float],
        memo: Optional[str] = None,
        skip_confirmation: bool = False,
    ) -> str:
        """
        Transfer SOL to another wallet.
        
        Args:
            recipient: Recipient's public key or address
            amount: Amount to send (in SOL if float, in lamports if int)
            memo: Optional memo to include with the transaction
            skip_confirmation: Whether to wait for transaction confirmation
            
        Returns:
            str: Transaction signature
            
        Raises:
            TransactionError: If the transfer fails
        """
        try:
            # Convert recipient to PublicKey if needed
            if isinstance(recipient, str):
                recipient_key = SolPublicKey(recipient)
            else:
                recipient_key = recipient
                
            # Convert amount to lamports if it's a float (SOL)
            lamports = int(amount * LAMPORTS_PER_SOL) if isinstance(amount, float) else amount
            
            # Create a transfer transaction
            transaction = Transaction()
            
            # Add transfer instruction
            transfer_instruction = transfer(
                TransferParams(
                    from_pubkey=self.public_key,
                    to_pubkey=recipient_key,
                    lamports=lamports
                )
            )
            transaction.add(transfer_instruction)
            
            # Add memo if provided
            if memo:
                memo_program_id = SolPublicKey("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr")
                memo_instruction = TransactionInstruction(
                    keys=[],
                    program_id=memo_program_id,
                    data=bytes(memo, 'utf-8')
                )
                transaction.add(memo_instruction)
            
            # Get recent blockhash
            response = await self.client.get_recent_blockhash()
            if "result" not in response or "value" not in response["result"]:
                raise TransactionError("Failed to get recent blockhash")
                
            blockhash = response["result"]["value"]["blockhash"]
            transaction.recent_blockhash = blockhash
            
            # Sign the transaction
            transaction.sign(self.keypair)
            
            # Send the transaction
            tx_opts = TxOpts(skip_preflight=False, skip_confirmation=skip_confirmation)
            response = await self.client.send_transaction(transaction, self.keypair, opts=tx_opts)
            
            if "result" not in response:
                error_msg = response.get("error", {}).get("message", "Unknown error")
                raise TransactionError(f"Transaction failed: {error_msg}")
                
            signature = response["result"]
            
            # Record in transaction history
            tx_record = {
                "type": "transfer",
                "signature": signature,
                "amount": lamports,
                "recipient": str(recipient_key),
                "timestamp": time.time(),
                "confirmed": not skip_confirmation,
                "memo": memo
            }
            self.transaction_history.append(tx_record)
            
            logger.info(f"Transferred {format_sol_amount(lamports)} to {recipient_key} (sig: {signature[:10]}...)")
            return signature
            
        except Exception as e:
            logger.error(f"SOL transfer failed: {str(e)}")
            raise TransactionError(f"Failed to transfer SOL: {str(e)}")
    
    async def transfer_token(
        self,
        token_mint: Union[str, SolPublicKey],
        recipient: Union[str, SolPublicKey],
        amount: Union[int, float],
        decimals: int = 9,
        skip_confirmation: bool = False,
    ) -> str:
        """
        Transfer SPL tokens to another wallet.
        
        Args:
            token_mint: Mint address of the token
            recipient: Recipient's public key or address
            amount: Amount to send (as decimal if float, raw amount if int)
            decimals: Token decimal places (default 9)
            skip_confirmation: Whether to wait for transaction confirmation
            
        Returns:
            str: Transaction signature
            
        Raises:
            TransactionError: If the transfer fails
        """
        try:
            # Convert inputs to appropriate types
            if isinstance(token_mint, str):
                token_mint_key = SolPublicKey(token_mint)
            else:
                token_mint_key = token_mint
                
            if isinstance(recipient, str):
                recipient_key = SolPublicKey(recipient)
            else:
                recipient_key = recipient
            
            # Convert amount to raw units if it's a float
            raw_amount = int(amount * (10 ** decimals)) if isinstance(amount, float) else amount
            
            # Initialize token client
            token = Token(
                conn=self.client,
                pubkey=token_mint_key,
                program_id=TOKEN_PROGRAM_ID,
                payer=self.keypair
            )
            
            # Get source token account
            source_accounts = await self.client.get_token_accounts_by_owner(
                self.public_key,
                {"mint": token_mint_key}
            )
            
            if not source_accounts["result"]["value"]:
                raise TransactionError(f"No token account found for mint {token_mint_key}")
            
            source_account = SolPublicKey(source_accounts["result"]["value"][0]["pubkey"])
            
            # Get or create destination token account
            dest_accounts = await self.client.get_token_accounts_by_owner(
                recipient_key,
                {"mint": token_mint_key}
            )
            
            if not dest_accounts["result"]["value"]:
                # Create associated token account for recipient
                create_tx = await token.create_associated_token_account(
                    owner=recipient_key
                )
                # Wait for confirmation
                await self.client.confirm_transaction(create_tx)
                
                # Get the new account
                dest_accounts = await self.client.get_token_accounts_by_owner(
                    recipient_key,
                    {"mint": token_mint_key}
                )
                
                if not dest_accounts["result"]["value"]:
                    raise TransactionError("Failed to create destination token account")
            
            destination_account = SolPublicKey(dest_accounts["result"]["value"][0]["pubkey"])
            
            # Transfer tokens
            tx_signature = await token.transfer(
                source=source_account,
                dest=destination_account,
                owner=self.keypair,
                amount=raw_amount,
                skip_preflight=False
            )
            
            # Record in transaction history
            tx_record = {
                "type": "token_transfer",
                "signature": tx_signature,
                "token_mint": str(token_mint_key),
                "amount": raw_amount,
                "decimals": decimals,
                "recipient": str(recipient_key),
                "timestamp": time.time(),
                "confirmed": not skip_confirmation,
            }
            self.transaction_history.append(tx_record)
            
            formatted_amount = format_token_amount(raw_amount, decimals)
            logger.info(f"Transferred {formatted_amount} tokens to {recipient_key} (sig: {tx_signature[:10]}...)")
            return tx_signature
            
        except Exception as e:
            logger.error(f"Token transfer failed: {str(e)}")
            raise TransactionError(f"Failed to transfer tokens: {str(e)}")
    
    async def create_token_account(
        self, 
        token_mint: Union[str, SolPublicKey]
    ) -> str:
        """
        Create a token account for a specific token mint.
        
        Args:
            token_mint: Mint address of the token
            
        Returns:
            str: The address of the created token account
            
        Raises:
            TransactionError: If account creation fails
        """
        try:
            # Convert mint to PublicKey if needed
            if isinstance(token_mint, str):
                token_mint_key = SolPublicKey(token_mint)
            else:
                token_mint_key = token_mint
                
            # Initialize token client
            token = Token(
                conn=self.client,
                pubkey=token_mint_key,
                program_id=TOKEN_PROGRAM_ID,
                payer=self.keypair
            )
            
            # Create associated token account
            account_address = await token.create_associated_token_account(
                owner=self.public_key
            )
            
            logger.info(f"Created token account {account_address} for mint {token_mint_key}")
            return account_address
            
        except Exception as e:
            logger.error(f"Token account creation failed: {str(e)}")
            raise TransactionError(f"Failed to create token account: {str(e)}")
    
    async def get_transaction_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent transactions involving this wallet.
        
        Args:
            limit: Maximum number of transactions to fetch
            
        Returns:
            List[Dict]: List of transaction details
            
        Raises:
            TransactionError: If history fetch fails
        """
        try:
            # Get transaction signatures for the address
            response = await self.client.get_signatures_for_address(
                self.public_key,
                limit=limit
            )
            
            if "result" not in response:
                raise TransactionError("Failed to fetch transaction signatures")
                
            signatures = [item["signature"] for item in response["result"]]
            transactions = []
            
            # Get transaction details for each signature
            for sig in signatures:
                tx_response = await self.client.get_transaction(sig)
                if "result" in tx_response and tx_response["result"]:
                    tx_data = tx_response["result"]
                    
                    # Extract basic transaction info
                    tx_info = {
                        "signature": sig,
                        "timestamp": tx_data.get("blockTime", 0),
                        "slot": tx_data.get("slot", 0),
                        "success": tx_data.get("meta", {}).get("err") is None,
                        "fee": tx_data.get("meta", {}).get("fee", 0),
                    }
                    
                    # Try to identify transaction type and details
                    instructions = tx_data.get("transaction", {}).get("message", {}).get("instructions", [])
                    if instructions:
                        # This is simplified and would need more logic for different instruction types
                        program_id = instructions[0].get("programId", "")
                        if program_id == "11111111111111111111111111111111":  # System program
                            tx_info["type"] = "system"
                        elif program_id == TOKEN_PROGRAM_ID:
                            tx_info["type"] = "token"
                        else:
                            tx_info["type"] = "other"
                    
                    transactions.append(tx_info)
            
            return transactions
            
        except Exception as e:
            logger.error(f"Failed to get transaction history: {str(e)}")
            raise TransactionError(f"Transaction history fetch failed: {str(e)}")

    async def airdrop(self, amount: int = LAMPORTS_PER_SOL) -> str:
        """
        Request an airdrop of SOL (only works on devnet/testnet).
        
        Args:
            amount: Amount of lamports to request (default: 1 SOL)
            
        Returns:
            str: Transaction signature if successful
            
        Raises:
            TransactionError: If airdrop fails
        """
        if self.network not in [NetworkType.DEVNET, NetworkType.TESTNET, NetworkType.LOCALHOST]:
            raise TransactionError("Airdrops are only available on devnet, testnet, or localhost")
            
        try:
            response = await self.client.request_airdrop(
                self.public_key,
                amount
            )
            
            if "result" not in response:
                error_msg = response.get("error", {}).get("message", "Unknown error")
                raise TransactionError(f"Airdrop failed: {error_msg}")
                
            signature = response["result"]
            
            # Wait for confirmation
            await self.client.confirm_transaction(signature)
            
            logger.info(f"Received airdrop of {format_sol_amount(amount)}, signature: {signature[:10]}...")
            return signature
            
        except Exception as e:
            logger.error(f"Airdrop failed: {str(e)}")
            raise TransactionError(f"Failed to request airdrop: {str(e)}")


async def estimate_transaction_fee(
    client: Client,
    num_signatures: int = 1,
    num_instructions: int = 1
) -> int:
    """
    Estimate the transaction fee for a Solana transaction.
    
    Args:
        client: Solana RPC client
        num_signatures: Number of signatures in the transaction
        num_instructions: Number of instructions in the transaction
        
    Returns:
        int: Estimated fee in lamports
        
    Raises:
        TransactionError: If fee estimation fails
    """
    try:
        # Get recent block fee metadata
        response = await client.get_recent_blockhash()
        if "result" not in response or "value" not in response["result"]:
            raise TransactionError("Failed to get recent blockhash")
            
        fee_calculator = response["result"]["value"].get("feeCalculator", {})
        lamports_per_signature = fee_calculator.get("lamportsPerSignature", 5000)
        
        # Basic fee calculation
        # This is simplified - actual Solana fees depend on more factors
        base_fee = lamports_per_signature * num_signatures
        instruction_fee = 1000 * num_instructions  # Approximation
        
        return base_fee + instruction_fee
        
    except Exception as e:
        logger.error(f"Fee estimation failed: {str(e)}")
        raise TransactionError(f"Failed to estimate transaction fee: {str(e)}")


async def get_network_status(client: Client) -> Dict[str, Any]:
    """
    Get the current status of the Solana network.
    
    Args:
        client: Solana RPC client
        
    Returns:
        Dict: Network status information
        
    Raises:
        TransactionError: If status check fails
    """
    try:
        # Collect various network metrics
        status = {}
        
        # Get version info
        version_response = await client.get_version()
        if "result" in version_response:
            status["version"] = version_response["result"]
            
        # Get cluster nodes
        nodes_response = await client.get_cluster_nodes()
        if "result" in nodes_response:
            status["node_count"] = len(nodes_response["result"])
            
        # Get recent performance samples
        performance_response = await client.get_recent_performance_samples(limit=1)
        if "result" in performance_response and performance_response["result"]:
            status["performance"] = performance_response["result"][0]
            
        # Get block height
        block_height_response = await client.get_block_height()
        if "result" in block_height_response:
            status["block_height"] = block_height_response["result"]
            
        # Get recent blockhash and fee
        blockhash_response = await client.get_recent_blockhash()
        if "result" in blockhash_response and "value" in blockhash_response["result"]:
            status["blockhash"] = blockhash_response["result"]["value"]["blockhash"]
            status["fee_calculator"] = blockhash_response["result"]["value"].get("feeCalculator", {})
            
        # Get epoch info
        epoch_info_response = await client.get_epoch_info()
        if "result" in epoch_info_response:
            status["epoch_info"] = epoch_info_response["result"]
            
        return status
        

def format_sol_amount(lamports: int) -> str:
    """
    Convert lamports to SOL with proper formatting.
    
    Args:
        lamports: Amount in lamports (10^9 lamports = 1 SOL)
        
    Returns:
        str: Formatted amount string (e.g., "1.234 SOL")
    """
    sol_amount = lamports / LAMPORTS_PER_SOL
    
    # For small amounts, show more decimal places
    if sol_amount < 0.001 and sol_amount > 0:
        return f"{sol_amount:.9f} SOL"
    elif sol_amount < 0.1:
        return f"{sol_amount:.6f} SOL"
    elif sol_amount < 1:
        return f"{sol_amount:.4f} SOL"
    else:
        # For larger amounts, limit to 2-3 decimal places
        return f"{sol_amount:.3f} SOL"


def format_token_amount(amount: int, decimals: int = 9) -> str:
    """
    Format token amount with proper decimal places.
    
    Args:
        amount: Raw token amount (integer)
        decimals: Number of decimal places for the token (default: 9)
        
    Returns:
        str: Formatted token amount as a string
    """
    if decimals == 0:
        return str(amount)
        
    # Convert raw amount to decimal representation
    decimal_divisor = 10 ** decimals
    decimal_amount = amount / decimal_divisor
    
    # Determine appropriate formatting based on size
    if decimal_amount < 0.001 and decimal_amount > 0:
        # Show more precision for very small amounts
        return f"{decimal_amount:.{min(decimals, 8)}f}"
    elif decimal_amount < 0.1:
        return f"{decimal_amount:.6f}"
    elif decimal_amount < 1:
        return f"{decimal_amount:.4f}"
    elif decimal_amount < 1000:
        return f"{decimal_amount:.2f}"
    else:
        # For large amounts, use fewer decimal places
        return f"{decimal_amount:,.2f}"
