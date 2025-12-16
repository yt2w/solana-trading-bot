"""
Async Wallet Management Module - Production Grade

Provides comprehensive async wallet management for Solana trading bot:
- Secure key generation and storage
- Encryption with PBKDF2 + Fernet
- RPC connection pooling with failover
- Thread-safe async operations
- Backup and restore functionality
"""

import asyncio
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import asynccontextmanager
import struct

# Cryptography imports
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

# Solana imports
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Commitment, Confirmed, Finalized
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.signature import Signature

# Local imports
from .config import config
from .logger import get_logger
from .exceptions import (
    WalletError,
    WalletNotFoundError,
    WalletExistsError,
    WalletLimitExceededError,
    EncryptionError,
    DecryptionError,
    InvalidPrivateKeyError,
    RPCError,
    RPCConnectionError,
    RPCTimeoutError,
    BackupError,
    RestoreError,
)


logger = get_logger(__name__)


# =============================================================================
# Constants
# =============================================================================

LAMPORTS_PER_SOL = 1_000_000_000
MAX_WALLETS_PER_USER = 10
PBKDF2_ITERATIONS = 600_000  # Strong security - OWASP recommended
BACKUP_VERSION = "1.0.0"
ENCRYPTION_SALT_SIZE = 32
WALLET_DATA_VERSION = 1


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class WalletInfo:
    """Wallet information container (never contains private key in plaintext)."""
    public_key: str
    wallet_name: str
    telegram_id: int
    created_at: datetime
    is_default: bool = False
    label: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "public_key": self.public_key,
            "wallet_name": self.wallet_name,
            "telegram_id": self.telegram_id,
            "created_at": self.created_at.isoformat(),
            "is_default": self.is_default,
            "label": self.label,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WalletInfo":
        """Create from dictionary."""
        return cls(
            public_key=data["public_key"],
            wallet_name=data["wallet_name"],
            telegram_id=data["telegram_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            is_default=data.get("is_default", False),
            label=data.get("label"),
        )


@dataclass
class TokenBalance:
    """Token balance information."""
    mint: str
    balance: int  # Raw amount
    decimals: int
    ui_amount: float  # Human-readable amount
    symbol: Optional[str] = None
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mint": self.mint,
            "balance": self.balance,
            "decimals": self.decimals,
            "ui_amount": self.ui_amount,
            "symbol": self.symbol,
            "name": self.name,
        }


@dataclass
class BalanceResult:
    """Complete balance result."""
    sol_balance: int  # In lamports
    sol_ui_amount: float  # In SOL
    token_balances: List[TokenBalance] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sol_balance": self.sol_balance,
            "sol_ui_amount": self.sol_ui_amount,
            "token_balances": [t.to_dict() for t in self.token_balances],
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RPCEndpoint:
    """RPC endpoint configuration."""
    url: str
    weight: int = 1
    is_healthy: bool = True
    last_check: Optional[datetime] = None
    latency_ms: Optional[float] = None
    consecutive_failures: int = 0
    

class ConnectionState(Enum):
    """Connection state enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


# =============================================================================
# Encryption Manager
# =============================================================================

class EncryptionManager:
    """
    Handles all encryption/decryption operations for wallet data.
    
    Uses PBKDF2 for key derivation and Fernet for symmetric encryption.
    Each user gets a unique encryption key derived from their telegram_id
    and the master secret.
    """
    
    def __init__(self, master_secret: str):
        """
        Initialize encryption manager.
        
        Args:
            master_secret: Master secret for key derivation
        """
        if not master_secret or len(master_secret) < 32:
            raise EncryptionError("Master secret must be at least 32 characters")
        
        self._master_secret = master_secret.encode()
        self._key_cache: Dict[int, bytes] = {}
        self._cache_lock = asyncio.Lock()
        
    def _derive_key(
        self,
        telegram_id: int,
        salt: Optional[bytes] = None,
        additional_password: Optional[str] = None,
    ) -> Tuple[bytes, bytes]:
        """
        Derive encryption key using PBKDF2.
        
        Args:
            telegram_id: User's telegram ID
            salt: Optional salt (generated if not provided)
            additional_password: Optional additional password layer
            
        Returns:
            Tuple of (derived_key, salt)
        """
        if salt is None:
            salt = secrets.token_bytes(ENCRYPTION_SALT_SIZE)
        
        # Combine master secret with telegram_id
        key_material = self._master_secret + str(telegram_id).encode()
        
        # Add additional password if provided
        if additional_password:
            key_material += additional_password.encode()
        
        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=PBKDF2_ITERATIONS,
            backend=default_backend(),
        )
        
        derived_key = base64.urlsafe_b64encode(kdf.derive(key_material))
        return derived_key, salt
    
    async def get_user_key(self, telegram_id: int) -> bytes:
        """
        Get or create encryption key for user.
        
        Args:
            telegram_id: User's telegram ID
            
        Returns:
            Encryption key for user
        """
        async with self._cache_lock:
            if telegram_id not in self._key_cache:
                # Use deterministic salt based on telegram_id for consistency
                deterministic_salt = hashlib.sha256(
                    self._master_secret + str(telegram_id).encode()
                ).digest()
                key, _ = self._derive_key(telegram_id, deterministic_salt)
                self._key_cache[telegram_id] = key
            
            return self._key_cache[telegram_id]
    
    async def encrypt(
        self,
        data: bytes,
        telegram_id: int,
        additional_password: Optional[str] = None,
    ) -> bytes:
        """
        Encrypt data for a user.
        
        Args:
            data: Data to encrypt
            telegram_id: User's telegram ID
            additional_password: Optional additional password
            
        Returns:
            Encrypted data with salt prepended
        """
        try:
            if additional_password:
                # Generate random salt for password-protected encryption
                key, salt = self._derive_key(telegram_id, additional_password=additional_password)
            else:
                key = await self.get_user_key(telegram_id)
                salt = b""  # No salt needed - using cached key
            
            fernet = Fernet(key)
            encrypted = fernet.encrypt(data)
            
            # Prepend salt length and salt if using additional password
            if additional_password:
                result = struct.pack(">I", len(salt)) + salt + encrypted
            else:
                result = struct.pack(">I", 0) + encrypted
            
            return result
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise EncryptionError(f"Failed to encrypt data: {e}")
    
    async def decrypt(
        self,
        encrypted_data: bytes,
        telegram_id: int,
        additional_password: Optional[str] = None,
    ) -> bytes:
        """
        Decrypt data for a user.
        
        Args:
            encrypted_data: Encrypted data (with salt prepended)
            telegram_id: User's telegram ID
            additional_password: Optional additional password
            
        Returns:
            Decrypted data
        """
        try:
            # Extract salt length
            salt_length = struct.unpack(">I", encrypted_data[:4])[0]
            
            if salt_length > 0:
                # Extract salt and encrypted portion
                salt = encrypted_data[4:4 + salt_length]
                encrypted = encrypted_data[4 + salt_length:]
                key, _ = self._derive_key(telegram_id, salt, additional_password)
            else:
                encrypted = encrypted_data[4:]
                key = await self.get_user_key(telegram_id)
            
            fernet = Fernet(key)
            return fernet.decrypt(encrypted)
            
        except InvalidToken:
            raise DecryptionError("Invalid password or corrupted data")
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise DecryptionError(f"Failed to decrypt data: {e}")
    
    async def rotate_key(self, telegram_id: int) -> None:
        """
        Rotate encryption key for user (invalidates cached key).
        
        Args:
            telegram_id: User's telegram ID
        """
        async with self._cache_lock:
            if telegram_id in self._key_cache:
                # Securely clear old key
                old_key = self._key_cache.pop(telegram_id)
                # Overwrite memory
                old_key_ba = bytearray(old_key)
                for i in range(len(old_key_ba)):
                    old_key_ba[i] = 0
    
    def clear_cache(self) -> None:
        """Clear all cached keys (for shutdown)."""
        for key in self._key_cache.values():
            # Attempt to overwrite
            key_ba = bytearray(key)
            for i in range(len(key_ba)):
                key_ba[i] = 0
        self._key_cache.clear()


# =============================================================================
# RPC Connection Pool
# =============================================================================

class RPCConnectionPool:
    """
    Manages pool of RPC connections with failover and load balancing.
    
    Features:
    - Multiple endpoint support
    - Health checking
    - Automatic failover
    - Round-robin load balancing
    - Connection reuse
    """
    
    def __init__(
        self,
        endpoints: Optional[List[str]] = None,
        max_connections_per_endpoint: int = 5,
        health_check_interval: float = 30.0,
        connection_timeout: float = 10.0,
    ):
        """
        Initialize connection pool.
        
        Args:
            endpoints: List of RPC endpoint URLs
            max_connections_per_endpoint: Max connections per endpoint
            health_check_interval: Seconds between health checks
            connection_timeout: Connection timeout in seconds
        """
        self._endpoints: List[RPCEndpoint] = []
        self._connections: Dict[str, List[AsyncClient]] = {}
        self._connection_semaphores: Dict[str, asyncio.Semaphore] = {}
        self._max_connections = max_connections_per_endpoint
        self._health_check_interval = health_check_interval
        self._connection_timeout = connection_timeout
        self._current_endpoint_idx = 0
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        self._state = ConnectionState.DISCONNECTED
        self._initialized = False
        
        # Setup endpoints
        endpoint_urls = endpoints or [config.get("solana.rpc_url", "https://api.mainnet-beta.solana.com")]
        for url in endpoint_urls:
            self._endpoints.append(RPCEndpoint(url=url))
            self._connections[url] = []
            self._connection_semaphores[url] = asyncio.Semaphore(max_connections_per_endpoint)
    
    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state
    
    @property
    def healthy_endpoints(self) -> List[RPCEndpoint]:
        """Get list of healthy endpoints."""
        return [ep for ep in self._endpoints if ep.is_healthy]
    
    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._initialized:
            return
        
        self._state = ConnectionState.CONNECTING
        logger.info("Initializing RPC connection pool...")
        
        try:
            # Check initial health of all endpoints
            await self._check_all_health()
            
            if not self.healthy_endpoints:
                raise RPCConnectionError("No healthy RPC endpoints available")
            
            # Start health check background task
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            self._state = ConnectionState.CONNECTED
            self._initialized = True
            logger.info(f"RPC connection pool initialized with {len(self.healthy_endpoints)} healthy endpoints")
            
        except Exception as e:
            self._state = ConnectionState.FAILED
            logger.error(f"Failed to initialize connection pool: {e}")
            raise RPCConnectionError(f"Failed to initialize: {e}")
    
    async def close(self) -> None:
        """Close all connections and cleanup."""
        logger.info("Closing RPC connection pool...")
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        for url, clients in self._connections.items():
            for client in clients:
                try:
                    await client.close()
                except Exception as e:
                    logger.warning(f"Error closing client for {url}: {e}")
            clients.clear()
        
        self._state = ConnectionState.DISCONNECTED
        self._initialized = False
        logger.info("RPC connection pool closed")
    
    async def _check_all_health(self) -> None:
        """Check health of all endpoints."""
        tasks = [self._check_endpoint_health(ep) for ep in self._endpoints]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_endpoint_health(self, endpoint: RPCEndpoint) -> None:
        """Check health of a single endpoint."""
        try:
            start = time.monotonic()
            async with AsyncClient(endpoint.url) as client:
                # Simple health check - get slot
                response = await asyncio.wait_for(
                    client.get_slot(),
                    timeout=self._connection_timeout
                )
                
                if hasattr(response, 'value') and response.value:
                    endpoint.is_healthy = True
                    endpoint.consecutive_failures = 0
                    endpoint.latency_ms = (time.monotonic() - start) * 1000
                else:
                    raise RPCError("Invalid response from endpoint")
                    
        except Exception as e:
            endpoint.consecutive_failures += 1
            if endpoint.consecutive_failures >= 3:
                endpoint.is_healthy = False
            logger.warning(f"Health check failed for {endpoint.url}: {e}")
        
        endpoint.last_check = datetime.now(timezone.utc)
    
    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self._check_all_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
    
    def _get_next_endpoint(self) -> RPCEndpoint:
        """Get next healthy endpoint using round-robin."""
        healthy = self.healthy_endpoints
        if not healthy:
            # Try to use any endpoint if none healthy
            if self._endpoints:
                return self._endpoints[0]
            raise RPCConnectionError("No RPC endpoints configured")
        
        # Round-robin selection
        self._current_endpoint_idx = (self._current_endpoint_idx + 1) % len(healthy)
        return healthy[self._current_endpoint_idx]
    
    @asynccontextmanager
    async def get_client(self) -> AsyncClient:
        """
        Get a client from the pool.
        
        Yields:
            AsyncClient instance
            
        Usage:
            async with pool.get_client() as client:
                result = await client.get_balance(pubkey)
        """
        if not self._initialized:
            await self.initialize()
        
        endpoint = self._get_next_endpoint()
        
        async with self._connection_semaphores[endpoint.url]:
            # Try to reuse existing connection
            async with self._lock:
                if self._connections[endpoint.url]:
                    client = self._connections[endpoint.url].pop()
                else:
                    client = None
            
            # Create new connection if needed
            if client is None:
                client = AsyncClient(
                    endpoint.url,
                    timeout=self._connection_timeout,
                )
            
            try:
                yield client
                
                # Return to pool on success
                async with self._lock:
                    if len(self._connections[endpoint.url]) < self._max_connections:
                        self._connections[endpoint.url].append(client)
                    else:
                        await client.close()
                        
            except Exception as e:
                # Don't return failed connection to pool
                try:
                    await client.close()
                except:
                    pass
                raise
    
    async def execute_with_retry(
        self,
        operation: str,
        func,
        *args,
        max_retries: int = 3,
        **kwargs
    ) -> Any:
        """
        Execute an RPC operation with automatic retry and failover.
        
        Args:
            operation: Operation name for logging
            func: Async function to execute (receives client as first arg)
            *args: Additional arguments for func
            max_retries: Maximum retry attempts
            **kwargs: Additional keyword arguments for func
            
        Returns:
            Result of the operation
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                async with self.get_client() as client:
                    result = await asyncio.wait_for(
                        func(client, *args, **kwargs),
                        timeout=self._connection_timeout * 2
                    )
                    return result
                    
            except asyncio.TimeoutError as e:
                last_error = RPCTimeoutError(f"{operation} timed out")
                logger.warning(f"{operation} attempt {attempt + 1} timed out")
                
            except Exception as e:
                last_error = e
                logger.warning(f"{operation} attempt {attempt + 1} failed: {e}")
            
            # Wait before retry with exponential backoff
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (2 ** attempt))
        
        raise RPCError(f"{operation} failed after {max_retries} attempts: {last_error}")


# =============================================================================
# Async Wallet Manager
# =============================================================================

class AsyncWalletManager:
    """
    Production-grade async wallet management.
    
    Features:
    - Secure key generation and storage
    - Encrypted private key storage
    - RPC connection pooling
    - Balance queries (SOL and SPL tokens)
    - Backup and restore
    - Thread-safe operations
    """
    
    def __init__(
        self,
        encryption_manager: Optional[EncryptionManager] = None,
        connection_pool: Optional[RPCConnectionPool] = None,
        storage_backend: Optional[Any] = None,
    ):
        """
        Initialize wallet manager.
        
        Args:
            encryption_manager: Encryption manager instance
            connection_pool: RPC connection pool instance
            storage_backend: Storage backend for wallet data (optional)
        """
        # Get master secret from config
        master_secret = config.get("security.encryption_key", "")
        if not master_secret:
            master_secret = config.get("security.master_secret", "")
        if not master_secret:
            # Generate a deterministic secret for development only
            logger.warning("No encryption key configured - using development key")
            master_secret = "development_key_do_not_use_in_production_" + "x" * 32
        
        self._encryption = encryption_manager or EncryptionManager(master_secret)
        self._pool = connection_pool or RPCConnectionPool()
        self._storage = storage_backend
        
        # In-memory wallet storage (replace with database in production)
        self._wallets: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self._wallet_lock = asyncio.Lock()
        
        # Operation locks per user
        self._user_locks: Dict[int, asyncio.Lock] = {}
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize wallet manager and connections."""
        if self._initialized:
            return
        
        logger.info("Initializing AsyncWalletManager...")
        await self._pool.initialize()
        self._initialized = True
        logger.info("AsyncWalletManager initialized")
    
    async def close(self) -> None:
        """Close wallet manager and cleanup resources."""
        logger.info("Closing AsyncWalletManager...")
        await self._pool.close()
        self._encryption.clear_cache()
        self._initialized = False
        logger.info("AsyncWalletManager closed")
    
    async def _get_user_lock(self, telegram_id: int) -> asyncio.Lock:
        """Get or create lock for user operations."""
        async with self._wallet_lock:
            if telegram_id not in self._user_locks:
                self._user_locks[telegram_id] = asyncio.Lock()
            return self._user_locks[telegram_id]
    
    def _validate_wallet_name(self, wallet_name: str) -> None:
        """Validate wallet name format."""
        if not wallet_name or not isinstance(wallet_name, str):
            raise WalletError("Wallet name is required")
        
        if len(wallet_name) > 32:
            raise WalletError("Wallet name must be 32 characters or less")
        
        if not wallet_name.replace("_", "").replace("-", "").isalnum():
            raise WalletError("Wallet name can only contain letters, numbers, underscores, and hyphens")
    
    async def _count_user_wallets(self, telegram_id: int) -> int:
        """Count user's wallets."""
        if telegram_id not in self._wallets:
            return 0
        return len(self._wallets[telegram_id])
    
    # =========================================================================
    # Wallet Operations
    # =========================================================================
    
    async def generate_wallet(
        self,
        telegram_id: int,
        wallet_name: str,
        label: Optional[str] = None,
        set_as_default: bool = False,
    ) -> WalletInfo:
        """
        Generate a new wallet for user.
        
        Args:
            telegram_id: User's Telegram ID
            wallet_name: Name for the wallet
            label: Optional human-readable label
            set_as_default: Set as default wallet
            
        Returns:
            WalletInfo for the new wallet
        """
        self._validate_wallet_name(wallet_name)
        
        user_lock = await self._get_user_lock(telegram_id)
        async with user_lock:
            # Check wallet limit
            if await self._count_user_wallets(telegram_id) >= MAX_WALLETS_PER_USER:
                raise WalletLimitExceededError(
                    f"Maximum {MAX_WALLETS_PER_USER} wallets per user"
                )
            
            # Check if wallet name exists
            if telegram_id in self._wallets and wallet_name in self._wallets[telegram_id]:
                raise WalletExistsError(f"Wallet '{wallet_name}' already exists")
            
            # Generate new keypair
            keypair = Keypair()
            public_key = str(keypair.pubkey())
            private_key = bytes(keypair)
            
            try:
                # Encrypt private key
                encrypted_key = await self._encryption.encrypt(
                    private_key,
                    telegram_id
                )
                
                # Create wallet info
                now = datetime.now(timezone.utc)
                wallet_info = WalletInfo(
                    public_key=public_key,
                    wallet_name=wallet_name,
                    telegram_id=telegram_id,
                    created_at=now,
                    is_default=set_as_default,
                    label=label,
                )
                
                # Store wallet
                if telegram_id not in self._wallets:
                    self._wallets[telegram_id] = {}
                
                self._wallets[telegram_id][wallet_name] = {
                    "info": wallet_info.to_dict(),
                    "encrypted_key": base64.b64encode(encrypted_key).decode(),
                    "version": WALLET_DATA_VERSION,
                }
                
                # Update default if needed
                if set_as_default:
                    await self._set_default_internal(telegram_id, wallet_name)
                
                logger.info(f"Generated wallet '{wallet_name}' for user {telegram_id}")
                return wallet_info
                
            finally:
                # Clear private key from memory
                private_key_ba = bytearray(private_key)
                for i in range(len(private_key_ba)):
                    private_key_ba[i] = 0

    
    async def import_wallet(
        self,
        telegram_id: int,
        private_key: Union[str, bytes, List[int]],
        wallet_name: str,
        label: Optional[str] = None,
        set_as_default: bool = False,
    ) -> WalletInfo:
        """
        Import wallet from private key.
        
        Args:
            telegram_id: User's Telegram ID
            private_key: Private key (base58, bytes, or list of ints)
            wallet_name: Name for the wallet
            label: Optional label
            set_as_default: Set as default wallet
            
        Returns:
            WalletInfo for imported wallet
        """
        self._validate_wallet_name(wallet_name)
        
        # Convert private key to bytes
        key_bytes: bytes
        try:
            if isinstance(private_key, str):
                # Assume base58 encoded
                import base58
                key_bytes = base58.b58decode(private_key)
            elif isinstance(private_key, list):
                key_bytes = bytes(private_key)
            else:
                key_bytes = private_key
            
            # Validate by creating keypair
            keypair = Keypair.from_bytes(key_bytes)
            public_key = str(keypair.pubkey())
            
        except Exception as e:
            raise InvalidPrivateKeyError(f"Invalid private key format: {e}")
        
        user_lock = await self._get_user_lock(telegram_id)
        async with user_lock:
            # Check wallet limit
            if await self._count_user_wallets(telegram_id) >= MAX_WALLETS_PER_USER:
                raise WalletLimitExceededError(
                    f"Maximum {MAX_WALLETS_PER_USER} wallets per user"
                )
            
            # Check if wallet name exists
            if telegram_id in self._wallets and wallet_name in self._wallets[telegram_id]:
                raise WalletExistsError(f"Wallet '{wallet_name}' already exists")
            
            # Check if public key already imported
            if telegram_id in self._wallets:
                for existing_wallet in self._wallets[telegram_id].values():
                    if existing_wallet["info"]["public_key"] == public_key:
                        raise WalletExistsError(
                            f"Wallet with this public key already exists"
                        )
            
            try:
                # Encrypt private key
                encrypted_key = await self._encryption.encrypt(
                    key_bytes,
                    telegram_id
                )
                
                # Create wallet info
                now = datetime.now(timezone.utc)
                wallet_info = WalletInfo(
                    public_key=public_key,
                    wallet_name=wallet_name,
                    telegram_id=telegram_id,
                    created_at=now,
                    is_default=set_as_default,
                    label=label,
                )
                
                # Store wallet
                if telegram_id not in self._wallets:
                    self._wallets[telegram_id] = {}
                
                self._wallets[telegram_id][wallet_name] = {
                    "info": wallet_info.to_dict(),
                    "encrypted_key": base64.b64encode(encrypted_key).decode(),
                    "version": WALLET_DATA_VERSION,
                }
                
                # Update default if needed
                if set_as_default:
                    await self._set_default_internal(telegram_id, wallet_name)
                
                logger.info(f"Imported wallet '{wallet_name}' for user {telegram_id}")
                return wallet_info
                
            finally:
                # Clear key from memory
                key_bytes_ba = bytearray(key_bytes)
                for i in range(len(key_bytes_ba)):
                    key_bytes_ba[i] = 0
    
    async def export_wallet(
        self,
        telegram_id: int,
        wallet_name: str,
        export_password: Optional[str] = None,
    ) -> bytes:
        """
        Export wallet with optional additional password protection.
        
        SECURITY: Returns encrypted data, never plaintext private key.
        
        Args:
            telegram_id: User's Telegram ID
            wallet_name: Wallet name to export
            export_password: Optional additional password for export
            
        Returns:
            Encrypted export data
        """
        user_lock = await self._get_user_lock(telegram_id)
        async with user_lock:
            if telegram_id not in self._wallets or wallet_name not in self._wallets[telegram_id]:
                raise WalletNotFoundError(f"Wallet '{wallet_name}' not found")
            
            wallet_data = self._wallets[telegram_id][wallet_name]
            
            # Get encrypted key
            encrypted_key = base64.b64decode(wallet_data["encrypted_key"])
            
            # Decrypt with user key
            private_key = await self._encryption.decrypt(encrypted_key, telegram_id)
            
            try:
                # Re-encrypt with export password if provided
                if export_password:
                    export_data = await self._encryption.encrypt(
                        private_key,
                        telegram_id,
                        additional_password=export_password
                    )
                else:
                    export_data = encrypted_key
                
                # Create export package
                export_package = {
                    "version": BACKUP_VERSION,
                    "wallet_name": wallet_name,
                    "public_key": wallet_data["info"]["public_key"],
                    "encrypted_key": base64.b64encode(export_data).decode(),
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "password_protected": export_password is not None,
                }
                
                logger.info(f"Exported wallet '{wallet_name}' for user {telegram_id}")
                return json.dumps(export_package).encode()
                
            finally:
                # Clear private key
                private_key_ba = bytearray(private_key)
                for i in range(len(private_key_ba)):
                    private_key_ba[i] = 0

    
    async def get_wallet(
        self,
        telegram_id: int,
        wallet_name: Optional[str] = None,
    ) -> Optional[WalletInfo]:
        """
        Get wallet info (never returns private key).
        
        Args:
            telegram_id: User's Telegram ID
            wallet_name: Wallet name (or None for default)
            
        Returns:
            WalletInfo or None
        """
        if telegram_id not in self._wallets:
            return None
        
        if wallet_name is None:
            # Get default wallet
            for data in self._wallets[telegram_id].values():
                if data["info"].get("is_default"):
                    return WalletInfo.from_dict(data["info"])
            
            # Return first wallet if no default
            if self._wallets[telegram_id]:
                first_wallet = next(iter(self._wallets[telegram_id].values()))
                return WalletInfo.from_dict(first_wallet["info"])
            return None
        
        if wallet_name not in self._wallets[telegram_id]:
            return None
        
        return WalletInfo.from_dict(self._wallets[telegram_id][wallet_name]["info"])
    
    async def list_wallets(self, telegram_id: int) -> List[WalletInfo]:
        """
        List all wallets for user.
        
        Args:
            telegram_id: User's Telegram ID
            
        Returns:
            List of WalletInfo objects
        """
        if telegram_id not in self._wallets:
            return []
        
        return [
            WalletInfo.from_dict(data["info"])
            for data in self._wallets[telegram_id].values()
        ]
    
    async def delete_wallet(
        self,
        telegram_id: int,
        wallet_name: str,
        confirm: bool = False,
    ) -> bool:
        """
        Delete a wallet.
        
        Args:
            telegram_id: User's Telegram ID
            wallet_name: Wallet name to delete
            confirm: Must be True to confirm deletion
            
        Returns:
            True if deleted
        """
        if not confirm:
            raise WalletError("Must confirm wallet deletion")
        
        user_lock = await self._get_user_lock(telegram_id)
        async with user_lock:
            if telegram_id not in self._wallets or wallet_name not in self._wallets[telegram_id]:
                raise WalletNotFoundError(f"Wallet '{wallet_name}' not found")
            
            # Get wallet data for secure clearing
            wallet_data = self._wallets[telegram_id].pop(wallet_name)
            
            # Clear encrypted key from memory
            if "encrypted_key" in wallet_data:
                encrypted_key = wallet_data["encrypted_key"]
                # Overwrite string (limited effectiveness due to string immutability)
                wallet_data["encrypted_key"] = "0" * len(encrypted_key)
            
            logger.info(f"Deleted wallet '{wallet_name}' for user {telegram_id}")
            
            # Clean up empty user entry
            if not self._wallets[telegram_id]:
                del self._wallets[telegram_id]
            
            return True
    
    async def set_default_wallet(
        self,
        telegram_id: int,
        wallet_name: str,
    ) -> WalletInfo:
        """
        Set wallet as default.
        
        Args:
            telegram_id: User's Telegram ID
            wallet_name: Wallet name to set as default
            
        Returns:
            Updated WalletInfo
        """
        user_lock = await self._get_user_lock(telegram_id)
        async with user_lock:
            if telegram_id not in self._wallets or wallet_name not in self._wallets[telegram_id]:
                raise WalletNotFoundError(f"Wallet '{wallet_name}' not found")
            
            await self._set_default_internal(telegram_id, wallet_name)
            return WalletInfo.from_dict(self._wallets[telegram_id][wallet_name]["info"])
    
    async def _set_default_internal(self, telegram_id: int, wallet_name: str) -> None:
        """Internal method to set default wallet (must hold lock)."""
        # Clear existing default
        for data in self._wallets[telegram_id].values():
            data["info"]["is_default"] = False
        
        # Set new default
        self._wallets[telegram_id][wallet_name]["info"]["is_default"] = True
    
    # =========================================================================
    # Keypair Access (Internal Use Only)
    # =========================================================================
    
    async def _get_keypair(
        self,
        telegram_id: int,
        wallet_name: Optional[str] = None,
    ) -> Keypair:
        """
        Get decrypted keypair for signing.
        
        INTERNAL USE ONLY - never expose to external callers.
        Caller MUST clear the keypair after use.
        
        Args:
            telegram_id: User's Telegram ID
            wallet_name: Wallet name (or None for default)
            
        Returns:
            Keypair instance
        """
        wallet_info = await self.get_wallet(telegram_id, wallet_name)
        if not wallet_info:
            raise WalletNotFoundError(
                f"Wallet '{wallet_name or 'default'}' not found"
            )
        
        wallet_data = self._wallets[telegram_id][wallet_info.wallet_name]
        encrypted_key = base64.b64decode(wallet_data["encrypted_key"])
        
        # Decrypt private key
        private_key = await self._encryption.decrypt(encrypted_key, telegram_id)
        
        try:
            return Keypair.from_bytes(private_key)
        finally:
            # Clear private key bytes
            private_key_ba = bytearray(private_key)
            for i in range(len(private_key_ba)):
                private_key_ba[i] = 0

    
    # =========================================================================
    # Balance Operations
    # =========================================================================
    
    async def get_sol_balance(
        self,
        public_key: Union[str, Pubkey],
    ) -> Tuple[int, float]:
        """
        Get SOL balance for an address.
        
        Args:
            public_key: Public key (string or Pubkey)
            
        Returns:
            Tuple of (lamports, sol_amount)
        """
        if not self._initialized:
            await self.initialize()
        
        pubkey = Pubkey.from_string(str(public_key)) if isinstance(public_key, str) else public_key
        
        async def _get_balance(client: AsyncClient) -> Tuple[int, float]:
            response = await client.get_balance(pubkey, commitment=Confirmed)
            lamports = response.value
            sol_amount = lamports / LAMPORTS_PER_SOL
            return lamports, sol_amount
        
        return await self._pool.execute_with_retry(
            "get_sol_balance",
            _get_balance
        )
    
    async def get_token_balances(
        self,
        public_key: Union[str, Pubkey],
    ) -> List[TokenBalance]:
        """
        Get all SPL token balances for an address.
        
        Args:
            public_key: Public key (string or Pubkey)
            
        Returns:
            List of TokenBalance objects
        """
        if not self._initialized:
            await self.initialize()
        
        pubkey = Pubkey.from_string(str(public_key)) if isinstance(public_key, str) else public_key
        
        async def _get_token_accounts(client: AsyncClient) -> List[TokenBalance]:
            from solana.rpc.types import TokenAccountOpts
            from solders.pubkey import Pubkey as SoldersPubkey
            
            # Token program ID
            token_program = SoldersPubkey.from_string(
                "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
            )
            
            response = await client.get_token_accounts_by_owner(
                pubkey,
                TokenAccountOpts(program_id=token_program),
                commitment=Confirmed,
            )
            
            balances = []
            if response.value:
                for account in response.value:
                    try:
                        account_data = account.account.data
                        # Parse token account data
                        # SPL Token account layout: 165 bytes
                        # Offset 0-32: mint
                        # Offset 32-64: owner
                        # Offset 64-72: amount (u64)
                        
                        if hasattr(account_data, 'parsed'):
                            parsed = account_data.parsed
                            info = parsed.get("info", {})
                            token_amount = info.get("tokenAmount", {})
                            
                            balances.append(TokenBalance(
                                mint=info.get("mint", ""),
                                balance=int(token_amount.get("amount", 0)),
                                decimals=token_amount.get("decimals", 0),
                                ui_amount=float(token_amount.get("uiAmount", 0)),
                            ))
                    except Exception as e:
                        logger.warning(f"Failed to parse token account: {e}")
                        continue
            
            return balances
        
        return await self._pool.execute_with_retry(
            "get_token_balances",
            _get_token_accounts
        )
    
    async def get_token_balance(
        self,
        public_key: Union[str, Pubkey],
        mint: Union[str, Pubkey],
    ) -> Optional[TokenBalance]:
        """
        Get balance for a specific token.
        
        Args:
            public_key: Wallet public key
            mint: Token mint address
            
        Returns:
            TokenBalance or None if not found
        """
        balances = await self.get_token_balances(public_key)
        mint_str = str(mint)
        
        for balance in balances:
            if balance.mint == mint_str:
                return balance
        
        return None
    
    async def get_full_balance(
        self,
        public_key: Union[str, Pubkey],
    ) -> BalanceResult:
        """
        Get complete balance (SOL + all tokens).
        
        Args:
            public_key: Public key
            
        Returns:
            BalanceResult with all balances
        """
        # Run both queries concurrently
        sol_task = self.get_sol_balance(public_key)
        tokens_task = self.get_token_balances(public_key)
        
        (lamports, sol_amount), token_balances = await asyncio.gather(
            sol_task,
            tokens_task
        )
        
        return BalanceResult(
            sol_balance=lamports,
            sol_ui_amount=sol_amount,
            token_balances=token_balances,
        )

    
    # =========================================================================
    # Backup and Restore
    # =========================================================================
    
    async def create_encrypted_backup(
        self,
        telegram_id: int,
        backup_password: str,
    ) -> bytes:
        """
        Create encrypted backup of all user wallets.
        
        Args:
            telegram_id: User's Telegram ID
            backup_password: Password for backup encryption
            
        Returns:
            Encrypted backup data
        """
        if not backup_password or len(backup_password) < 8:
            raise BackupError("Backup password must be at least 8 characters")
        
        user_lock = await self._get_user_lock(telegram_id)
        async with user_lock:
            if telegram_id not in self._wallets or not self._wallets[telegram_id]:
                raise BackupError("No wallets to backup")
            
            # Collect wallet data
            wallets_data = []
            for wallet_name, wallet_data in self._wallets[telegram_id].items():
                # Get encrypted key
                encrypted_key = base64.b64decode(wallet_data["encrypted_key"])
                
                # Decrypt with user key
                private_key = await self._encryption.decrypt(encrypted_key, telegram_id)
                
                try:
                    wallets_data.append({
                        "wallet_name": wallet_name,
                        "public_key": wallet_data["info"]["public_key"],
                        "private_key": base64.b64encode(private_key).decode(),
                        "label": wallet_data["info"].get("label"),
                        "is_default": wallet_data["info"].get("is_default", False),
                        "created_at": wallet_data["info"]["created_at"],
                    })
                finally:
                    # Clear private key
                    private_key_ba = bytearray(private_key)
                    for i in range(len(private_key_ba)):
                        private_key_ba[i] = 0
            
            # Create backup structure
            backup = {
                "version": BACKUP_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "telegram_id": telegram_id,
                "wallet_count": len(wallets_data),
                "wallets": wallets_data,
            }
            
            # Serialize and encrypt
            backup_json = json.dumps(backup).encode()
            
            try:
                encrypted_backup = await self._encryption.encrypt(
                    backup_json,
                    telegram_id,
                    additional_password=backup_password
                )
                
                # Add header for identification
                header = b"SOLBOT_BACKUP_V1"
                return header + encrypted_backup
                
            finally:
                # Clear sensitive data
                for wallet in wallets_data:
                    if "private_key" in wallet:
                        wallet["private_key"] = "0" * len(wallet["private_key"])
    
    async def restore_from_backup(
        self,
        telegram_id: int,
        backup_data: bytes,
        backup_password: str,
        overwrite: bool = False,
    ) -> List[WalletInfo]:
        """
        Restore wallets from encrypted backup.
        
        Args:
            telegram_id: User's Telegram ID
            backup_data: Encrypted backup data
            backup_password: Password for decryption
            overwrite: Overwrite existing wallets with same names
            
        Returns:
            List of restored WalletInfo objects
        """
        # Verify header
        header = b"SOLBOT_BACKUP_V1"
        if not backup_data.startswith(header):
            raise RestoreError("Invalid backup format")
        
        encrypted_data = backup_data[len(header):]
        
        try:
            # Decrypt backup
            backup_json = await self._encryption.decrypt(
                encrypted_data,
                telegram_id,
                additional_password=backup_password
            )
            
            backup = json.loads(backup_json.decode())
            
        except DecryptionError:
            raise RestoreError("Invalid backup password")
        except json.JSONDecodeError:
            raise RestoreError("Corrupted backup data")
        
        # Validate backup structure
        if backup.get("version") != BACKUP_VERSION:
            raise RestoreError(f"Unsupported backup version: {backup.get('version')}")
        
        restored = []
        user_lock = await self._get_user_lock(telegram_id)
        
        async with user_lock:
            for wallet_data in backup.get("wallets", []):
                wallet_name = wallet_data["wallet_name"]
                
                # Check if exists
                exists = (
                    telegram_id in self._wallets and 
                    wallet_name in self._wallets[telegram_id]
                )
                
                if exists and not overwrite:
                    logger.warning(f"Skipping existing wallet: {wallet_name}")
                    continue
                
                # Decode private key
                private_key = base64.b64decode(wallet_data["private_key"])
                
                try:
                    # Encrypt with user key
                    encrypted_key = await self._encryption.encrypt(
                        private_key,
                        telegram_id
                    )
                    
                    # Create wallet info
                    wallet_info = WalletInfo(
                        public_key=wallet_data["public_key"],
                        wallet_name=wallet_name,
                        telegram_id=telegram_id,
                        created_at=datetime.fromisoformat(wallet_data["created_at"]),
                        is_default=wallet_data.get("is_default", False),
                        label=wallet_data.get("label"),
                    )
                    
                    # Store wallet
                    if telegram_id not in self._wallets:
                        self._wallets[telegram_id] = {}
                    
                    self._wallets[telegram_id][wallet_name] = {
                        "info": wallet_info.to_dict(),
                        "encrypted_key": base64.b64encode(encrypted_key).decode(),
                        "version": WALLET_DATA_VERSION,
                    }
                    
                    restored.append(wallet_info)
                    
                finally:
                    # Clear private key
                    private_key_ba = bytearray(private_key)
                    for i in range(len(private_key_ba)):
                        private_key_ba[i] = 0
        
        logger.info(f"Restored {len(restored)} wallets for user {telegram_id}")
        return restored


# =============================================================================
# Factory Function
# =============================================================================

_wallet_manager: Optional[AsyncWalletManager] = None
_wallet_manager_lock = asyncio.Lock()


async def get_wallet_manager() -> AsyncWalletManager:
    """
    Get or create the global wallet manager instance.
    
    Returns:
        AsyncWalletManager singleton instance
    """
    global _wallet_manager
    
    async with _wallet_manager_lock:
        if _wallet_manager is None:
            _wallet_manager = AsyncWalletManager()
            await _wallet_manager.initialize()
        
        return _wallet_manager


async def close_wallet_manager() -> None:
    """Close the global wallet manager instance."""
    global _wallet_manager
    
    async with _wallet_manager_lock:
        if _wallet_manager is not None:
            await _wallet_manager.close()
            _wallet_manager = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Classes
    "AsyncWalletManager",
    "EncryptionManager",
    "RPCConnectionPool",
    "WalletInfo",
    "TokenBalance",
    "BalanceResult",
    "RPCEndpoint",
    "ConnectionState",
    # Functions
    "get_wallet_manager",
    "close_wallet_manager",
    # Constants
    "LAMPORTS_PER_SOL",
    "MAX_WALLETS_PER_USER",
]
