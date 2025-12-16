"""
Comprehensive Exception Hierarchy for Solana Trading Bot.

This module defines a complete exception hierarchy for handling all error
conditions in the trading bot, from wallet operations to RPC communication.

Each exception includes:
- Unique error code for logging and debugging
- Descriptive message
- Optional context dictionary for additional debugging info
- is_recoverable flag indicating if operation can be retried
- retry_after for rate-limited operations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime, timezone


@dataclass
class SolanaTraderError(Exception):
    """
    Base exception for all Solana Trading Bot errors.

    All custom exceptions in the trading bot inherit from this class,
    providing a consistent interface for error handling and logging.

    Attributes:
        message: Human-readable error description
        error_code: Unique identifier for the error type (e.g., "WALLET_001")
        context: Optional dictionary with debugging information
        is_recoverable: Whether the operation can be retried
        retry_after: Seconds to wait before retry (for rate limits)
        timestamp: When the error occurred
    """
    message: str
    error_code: str = "GENERAL_001"
    context: dict[str, Any] = field(default_factory=dict)
    is_recoverable: bool = False
    retry_after: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Initialize the exception with the formatted message."""
        super().__init__(self.format_message())

    def format_message(self) -> str:
        """Format the error message with code and context."""
        base = f"[{self.error_code}] {self.message}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base += f" | Context: {context_str}"
        return base

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "is_recoverable": self.is_recoverable,
            "retry_after": self.retry_after,
            "timestamp": self.timestamp.isoformat(),
            "exception_type": self.__class__.__name__,
        }

    def __str__(self) -> str:
        return self.format_message()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"error_code={self.error_code!r}, "
            f"message={self.message!r}, "
            f"is_recoverable={self.is_recoverable})"
        )


@dataclass
class ConfigurationError(SolanaTraderError):
    """Error in bot configuration or settings."""
    error_code: str = "CONFIG_001"
    is_recoverable: bool = False


@dataclass
class InitializationError(SolanaTraderError):
    """Error during component initialization."""
    error_code: str = "INIT_001"
    is_recoverable: bool = False


@dataclass
class WalletError(SolanaTraderError):
    """Base exception for wallet-related errors."""
    error_code: str = "WALLET_000"


@dataclass
class KeyGenerationError(WalletError):
    """
    Failed to generate cryptographic keys.

    Raised when keypair generation fails due to insufficient entropy,
    cryptographic library errors, or system resource issues.
    """
    error_code: str = "WALLET_001"
    is_recoverable: bool = True


@dataclass
class EncryptionError(WalletError):
    """
    Failed to encrypt wallet data.

    Raised when wallet encryption fails, typically due to invalid
    encryption parameters or cryptographic library errors.
    """
    error_code: str = "WALLET_002"
    is_recoverable: bool = False


@dataclass
class DecryptionError(WalletError):
    """
    Failed to decrypt wallet data.

    Raised when wallet decryption fails, usually due to incorrect
    password, corrupted data, or incompatible encryption format.
    """
    error_code: str = "WALLET_003"
    is_recoverable: bool = False


@dataclass
class WalletNotFoundError(WalletError):
    """
    Wallet file or data not found.

    Raised when attempting to load a wallet that doesn't exist
    at the specified path or in the expected storage location.
    """
    error_code: str = "WALLET_004"
    is_recoverable: bool = False


@dataclass
class WalletLockedError(WalletError):
    """
    Wallet is locked and requires unlocking.

    Raised when attempting to perform operations on a locked wallet
    that requires the private key.
    """
    error_code: str = "WALLET_005"
    is_recoverable: bool = True


@dataclass
class WalletAlreadyExistsError(WalletError):
    """
    Wallet already exists at the specified location.

    Raised when attempting to create a new wallet at a path
    where a wallet already exists.
    """
    error_code: str = "WALLET_006"
    is_recoverable: bool = False


@dataclass
class InvalidMnemonicError(WalletError):
    """
    Invalid mnemonic phrase provided.

    Raised when the provided mnemonic phrase is invalid,
    has incorrect checksum, or wrong word count.
    """
    error_code: str = "WALLET_007"
    is_recoverable: bool = False


@dataclass
class InvalidPrivateKeyError(WalletError):
    """
    Invalid private key format or value.

    Raised when the provided private key is malformed,
    wrong length, or contains invalid characters.
    """
    error_code: str = "WALLET_008"
    is_recoverable: bool = False


@dataclass
class InsufficientBalanceError(WalletError):
    """
    Insufficient balance for the requested operation.

    Raised when the wallet doesn't have enough SOL or tokens
    to complete the requested transaction.
    """
    error_code: str = "WALLET_009"
    is_recoverable: bool = True
    required_amount: Optional[float] = None
    available_amount: Optional[float] = None
    token_mint: Optional[str] = None


@dataclass
class WalletCorruptedError(WalletError):
    """
    Wallet data is corrupted or invalid.

    Raised when wallet file exists but contains invalid
    or corrupted data that cannot be parsed.
    """
    error_code: str = "WALLET_010"
    is_recoverable: bool = False


@dataclass
class TransactionError(SolanaTraderError):
    """Base exception for transaction-related errors."""
    error_code: str = "TX_000"
    transaction_signature: Optional[str] = None


@dataclass
class TransactionBuildError(TransactionError):
    """
    Failed to build transaction.

    Raised when transaction construction fails due to invalid
    instructions, missing accounts, or serialization errors.
    """
    error_code: str = "TX_001"
    is_recoverable: bool = True


@dataclass
class TransactionSignError(TransactionError):
    """
    Failed to sign transaction.

    Raised when transaction signing fails due to locked wallet,
    missing signers, or cryptographic errors.
    """
    error_code: str = "TX_002"
    is_recoverable: bool = True


@dataclass
class TransactionSendError(TransactionError):
    """
    Failed to send transaction to network.

    Raised when transaction broadcast fails due to network
    issues, RPC errors, or node rejection.
    """
    error_code: str = "TX_003"
    is_recoverable: bool = True


@dataclass
class TransactionConfirmationError(TransactionError):
    """
    Transaction confirmation failed or timed out.

    Raised when a sent transaction fails to confirm within
    the expected timeframe or is dropped from the network.
    """
    error_code: str = "TX_004"
    is_recoverable: bool = True
    confirmation_timeout: Optional[float] = None


@dataclass
class TransactionSimulationError(TransactionError):
    """
    Transaction simulation failed.

    Raised when transaction simulation indicates the transaction
    would fail if executed, with details about the failure reason.
    """
    error_code: str = "TX_005"
    is_recoverable: bool = False
    simulation_logs: list[str] = field(default_factory=list)


@dataclass
class TransactionExpiredError(TransactionError):
    """
    Transaction blockhash expired.

    Raised when a transaction's blockhash becomes too old
    and the transaction can no longer be processed.
    """
    error_code: str = "TX_006"
    is_recoverable: bool = True


@dataclass
class TransactionAlreadyProcessedError(TransactionError):
    """
    Transaction was already processed.

    Raised when attempting to send a duplicate transaction
    that has already been confirmed on-chain.
    """
    error_code: str = "TX_007"
    is_recoverable: bool = False


@dataclass
class InstructionError(TransactionError):
    """
    Program instruction execution failed.

    Raised when a specific instruction within a transaction
    fails during execution, with program-specific error details.
    """
    error_code: str = "TX_008"
    is_recoverable: bool = False
    program_id: Optional[str] = None
    instruction_index: Optional[int] = None
    program_error: Optional[str] = None


@dataclass
class PriorityFeeError(TransactionError):
    """
    Failed to calculate or apply priority fee.

    Raised when priority fee estimation fails or the
    calculated fee is outside acceptable bounds.
    """
    error_code: str = "TX_009"
    is_recoverable: bool = True


@dataclass
class JupiterError(SolanaTraderError):
    """Base exception for Jupiter DEX aggregator errors."""
    error_code: str = "JUP_000"


@dataclass
class QuoteError(JupiterError):
    """
    Failed to get swap quote from Jupiter.

    Raised when quote request fails due to API errors,
    invalid parameters, or no available routes.
    """
    error_code: str = "JUP_001"
    is_recoverable: bool = True
    input_mint: Optional[str] = None
    output_mint: Optional[str] = None
    amount: Optional[int] = None


@dataclass
class SwapError(JupiterError):
    """
    Swap execution failed.

    Raised when swap transaction fails after quote was obtained,
    due to price movement, slippage, or execution errors.
    """
    error_code: str = "JUP_002"
    is_recoverable: bool = True


@dataclass
class TokenNotFoundError(JupiterError):
    """
    Token not found or not supported by Jupiter.

    Raised when the requested token mint address is not
    recognized or supported by Jupiter's routing.
    """
    error_code: str = "JUP_003"
    is_recoverable: bool = False
    token_mint: Optional[str] = None


@dataclass
class InsufficientLiquidityError(JupiterError):
    """
    Insufficient liquidity for the requested swap.

    Raised when there isn't enough liquidity in the market
    to execute the swap at acceptable price levels.
    """
    error_code: str = "JUP_004"
    is_recoverable: bool = True
    available_liquidity: Optional[float] = None
    required_liquidity: Optional[float] = None


@dataclass
class SlippageExceededError(JupiterError):
    """
    Price slippage exceeded tolerance.

    Raised when the actual execution price differs from
    the quoted price by more than the allowed slippage.
    """
    error_code: str = "JUP_005"
    is_recoverable: bool = True
    expected_amount: Optional[int] = None
    actual_amount: Optional[int] = None
    slippage_bps: Optional[int] = None


@dataclass
class RouteNotFoundError(JupiterError):
    """
    No swap route found between tokens.

    Raised when Jupiter cannot find any viable route
    between the input and output tokens.
    """
    error_code: str = "JUP_006"
    is_recoverable: bool = True


@dataclass
class JupiterAPIError(JupiterError):
    """
    Jupiter API returned an error response.

    Raised when Jupiter API returns an error status code
    or error message in the response body.
    """
    error_code: str = "JUP_007"
    is_recoverable: bool = True
    status_code: Optional[int] = None
    api_error_message: Optional[str] = None


@dataclass
class PriceImpactTooHighError(JupiterError):
    """
    Price impact of swap is too high.

    Raised when the swap would cause excessive price impact,
    indicating potential loss or manipulation risk.
    """
    error_code: str = "JUP_008"
    is_recoverable: bool = True
    price_impact_percent: Optional[float] = None
    max_allowed_impact: Optional[float] = None


@dataclass
class RPCError(SolanaTraderError):
    """Base exception for RPC-related errors."""
    error_code: str = "RPC_000"
    rpc_endpoint: Optional[str] = None


@dataclass
class RPCConnectionError(RPCError):
    """
    Failed to connect to RPC endpoint.

    Raised when connection to the Solana RPC node cannot
    be established due to network or server issues.
    """
    error_code: str = "RPC_001"
    is_recoverable: bool = True


@dataclass
class RPCTimeoutError(RPCError):
    """
    RPC request timed out.

    Raised when an RPC request does not receive a response
    within the configured timeout period.
    """
    error_code: str = "RPC_002"
    is_recoverable: bool = True
    timeout_seconds: Optional[float] = None


@dataclass
class RPCRateLimitError(RPCError):
    """
    RPC rate limit exceeded.

    Raised when too many requests have been sent to the RPC
    endpoint and rate limiting has been triggered.
    """
    error_code: str = "RPC_003"
    is_recoverable: bool = True
    retry_after: Optional[float] = 60.0


@dataclass
class RPCResponseError(RPCError):
    """
    RPC returned an error response.

    Raised when the RPC endpoint returns a JSON-RPC error
    in the response, indicating request or processing failure.
    """
    error_code: str = "RPC_004"
    is_recoverable: bool = True
    rpc_error_code: Optional[int] = None
    rpc_error_message: Optional[str] = None


@dataclass
class RPCNodeUnhealthyError(RPCError):
    """
    RPC node is unhealthy or behind.

    Raised when the RPC node reports unhealthy status
    or is significantly behind the network slot.
    """
    error_code: str = "RPC_005"
    is_recoverable: bool = True
    node_slot: Optional[int] = None
    network_slot: Optional[int] = None


@dataclass
class RPCMethodNotFoundError(RPCError):
    """
    RPC method not supported by endpoint.

    Raised when calling an RPC method that is not supported
    by the connected endpoint (e.g., disabled methods).
    """
    error_code: str = "RPC_006"
    is_recoverable: bool = False
    method_name: Optional[str] = None


@dataclass
class WebSocketError(RPCError):
    """
    WebSocket connection error.

    Raised when WebSocket connection fails or is unexpectedly
    closed during subscription operations.
    """
    error_code: str = "RPC_007"
    is_recoverable: bool = True


@dataclass
class BlockhashNotFoundError(RPCError):
    """
    Recent blockhash not available.

    Raised when the RPC cannot provide a recent blockhash,
    typically during network congestion or node issues.
    """
    error_code: str = "RPC_008"
    is_recoverable: bool = True


@dataclass
class ValidationError(SolanaTraderError):
    """Base exception for validation errors."""
    error_code: str = "VAL_000"
    is_recoverable: bool = False


@dataclass
class InvalidAddressError(ValidationError):
    """
    Invalid Solana address format.

    Raised when a provided address is not a valid base58-encoded
    Solana public key of the correct length.
    """
    error_code: str = "VAL_001"
    invalid_address: Optional[str] = None


@dataclass
class InvalidAmountError(ValidationError):
    """
    Invalid amount specified.

    Raised when an amount is negative, zero when not allowed,
    exceeds maximum, or has invalid decimal precision.
    """
    error_code: str = "VAL_002"
    amount: Optional[float] = None
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None


@dataclass
class InvalidSlippageError(ValidationError):
    """
    Invalid slippage tolerance specified.

    Raised when slippage is outside acceptable bounds
    (typically 0-100% or 0-10000 bps).
    """
    error_code: str = "VAL_003"
    slippage_bps: Optional[int] = None
    min_slippage: Optional[int] = None
    max_slippage: Optional[int] = None


@dataclass
class InvalidTokenError(ValidationError):
    """
    Invalid token specification.

    Raised when a token symbol, mint address, or configuration
    is invalid or not recognized.
    """
    error_code: str = "VAL_004"
    token: Optional[str] = None


@dataclass
class InvalidSignatureError(ValidationError):
    """
    Invalid transaction signature format.

    Raised when a transaction signature is not a valid
    base58-encoded 64-byte signature.
    """
    error_code: str = "VAL_005"
    signature: Optional[str] = None


@dataclass
class InvalidConfigError(ValidationError):
    """
    Invalid configuration value.

    Raised when a configuration parameter is invalid,
    out of range, or of the wrong type.
    """
    error_code: str = "VAL_006"
    config_key: Optional[str] = None
    config_value: Optional[Any] = None
    expected_type: Optional[str] = None


@dataclass
class InvalidTimestampError(ValidationError):
    """
    Invalid timestamp value.

    Raised when a timestamp is invalid, in the past when
    not allowed, or too far in the future.
    """
    error_code: str = "VAL_007"
    timestamp: Optional[float] = None


@dataclass
class RateLimitError(SolanaTraderError):
    """Base exception for rate limiting errors."""
    error_code: str = "RATE_000"
    is_recoverable: bool = True
    retry_after: Optional[float] = 1.0


@dataclass
class RateLimitExceededError(RateLimitError):
    """
    Internal rate limit exceeded.

    Raised when the bot's internal rate limiter blocks
    a request to prevent exceeding external limits.
    """
    error_code: str = "RATE_001"
    limit_name: Optional[str] = None
    current_rate: Optional[float] = None
    max_rate: Optional[float] = None


@dataclass
class BurstLimitExceededError(RateLimitError):
    """
    Burst rate limit exceeded.

    Raised when too many requests are made in a short
    time window, exceeding burst capacity.
    """
    error_code: str = "RATE_002"
    burst_count: Optional[int] = None
    burst_limit: Optional[int] = None
    window_seconds: Optional[float] = None


@dataclass
class DailyLimitExceededError(RateLimitError):
    """
    Daily request/operation limit exceeded.

    Raised when daily limits for API calls or operations
    have been exhausted.
    """
    error_code: str = "RATE_003"
    retry_after: Optional[float] = 86400.0
    daily_count: Optional[int] = None
    daily_limit: Optional[int] = None


@dataclass
class ConcurrencyLimitError(RateLimitError):
    """
    Maximum concurrent operations exceeded.

    Raised when attempting to start a new operation while
    at maximum concurrent operation capacity.
    """
    error_code: str = "RATE_004"
    current_concurrent: Optional[int] = None
    max_concurrent: Optional[int] = None


@dataclass
class SecurityError(SolanaTraderError):
    """Base exception for security-related errors."""
    error_code: str = "SEC_000"
    is_recoverable: bool = False


@dataclass
class SecurityViolationError(SecurityError):
    """
    Security policy violation detected.

    Raised when an operation violates security policies,
    such as suspicious transaction patterns or unauthorized access.
    """
    error_code: str = "SEC_001"
    violation_type: Optional[str] = None
    blocked_action: Optional[str] = None


@dataclass
class AuditIntegrityError(SecurityError):
    """
    Audit log integrity check failed.

    Raised when audit log verification fails, indicating
    potential tampering or corruption.
    """
    error_code: str = "SEC_002"
    expected_hash: Optional[str] = None
    actual_hash: Optional[str] = None


@dataclass
class UnauthorizedAccessError(SecurityError):
    """
    Unauthorized access attempt detected.

    Raised when an operation is attempted without proper
    authorization or from an unauthorized context.
    """
    error_code: str = "SEC_003"
    resource: Optional[str] = None
    required_permission: Optional[str] = None


@dataclass
class SuspiciousActivityError(SecurityError):
    """
    Suspicious activity pattern detected.

    Raised when unusual patterns are detected that may
    indicate compromise, attack, or malfunction.
    """
    error_code: str = "SEC_004"
    activity_type: Optional[str] = None
    details: Optional[str] = None


@dataclass
class TokenValidationError(SecurityError):
    """
    Token security validation failed.

    Raised when a token fails security checks such as
    honeypot detection, rugpull indicators, etc.
    """
    error_code: str = "SEC_005"
    token_mint: Optional[str] = None
    validation_failure: Optional[str] = None


@dataclass
class MaxExposureExceededError(SecurityError):
    """
    Maximum exposure limit exceeded.

    Raised when a trade would exceed configured maximum
    exposure limits for risk management.
    """
    error_code: str = "SEC_006"
    is_recoverable: bool = True
    current_exposure: Optional[float] = None
    max_exposure: Optional[float] = None
    requested_amount: Optional[float] = None


@dataclass
class SandboxViolationError(SecurityError):
    """
    Sandbox/isolation violation detected.

    Raised when an operation attempts to break out of
    configured security boundaries or isolation.
    """
    error_code: str = "SEC_007"
    attempted_action: Optional[str] = None


@dataclass
class NetworkError(SolanaTraderError):
    """Base exception for network-related errors."""
    error_code: str = "NET_000"
    is_recoverable: bool = True


@dataclass
class HTTPError(NetworkError):
    """
    HTTP request failed.

    Raised when an HTTP request to an external service
    fails with an error status code.
    """
    error_code: str = "NET_001"
    status_code: Optional[int] = None
    url: Optional[str] = None


@dataclass
class ConnectionPoolExhaustedError(NetworkError):
    """
    Connection pool exhausted.

    Raised when all connections in the pool are in use
    and no connection is available for a new request.
    """
    error_code: str = "NET_002"
    pool_size: Optional[int] = None


@dataclass
class DNSResolutionError(NetworkError):
    """
    DNS resolution failed.

    Raised when hostname cannot be resolved to IP address.
    """
    error_code: str = "NET_003"
    hostname: Optional[str] = None


@dataclass
class SSLError(NetworkError):
    """
    SSL/TLS handshake or verification failed.

    Raised when secure connection cannot be established
    due to certificate or protocol issues.
    """
    error_code: str = "NET_004"


@dataclass
class DataError(SolanaTraderError):
    """Base exception for data-related errors."""
    error_code: str = "DATA_000"


@dataclass
class SerializationError(DataError):
    """
    Data serialization failed.

    Raised when data cannot be serialized to the target
    format (JSON, binary, etc.).
    """
    error_code: str = "DATA_001"
    is_recoverable: bool = False
    data_type: Optional[str] = None


@dataclass
class DeserializationError(DataError):
    """
    Data deserialization failed.

    Raised when data cannot be parsed from its stored
    or transmitted format.
    """
    error_code: str = "DATA_002"
    is_recoverable: bool = False
    data_type: Optional[str] = None


@dataclass
class CacheError(DataError):
    """
    Cache operation failed.

    Raised when read/write to cache fails or cache
    is in an invalid state.
    """
    error_code: str = "DATA_003"
    is_recoverable: bool = True
    cache_key: Optional[str] = None


@dataclass
class StorageError(DataError):
    """
    Persistent storage operation failed.

    Raised when file system or database operations fail.
    """
    error_code: str = "DATA_004"
    is_recoverable: bool = True
    storage_path: Optional[str] = None


def is_retryable(error: Exception) -> bool:
    """
    Check if an error is retryable.

    Args:
        error: The exception to check

    Returns:
        True if the error is recoverable and can be retried
    """
    if isinstance(error, SolanaTraderError):
        return error.is_recoverable
    return False


def get_retry_delay(error: Exception, default: float = 1.0) -> float:
    """
    Get the recommended retry delay for an error.

    Args:
        error: The exception to get delay for
        default: Default delay if not specified

    Returns:
        Recommended delay in seconds before retry
    """
    if isinstance(error, SolanaTraderError) and error.retry_after is not None:
        return error.retry_after
    return default


def wrap_exception(
    original: Exception,
    wrapper_class: type[SolanaTraderError],
    message: Optional[str] = None,
    **kwargs: Any
) -> SolanaTraderError:
    """
    Wrap a generic exception in a SolanaTraderError subclass.

    Args:
        original: The original exception to wrap
        wrapper_class: The SolanaTraderError subclass to use
        message: Optional custom message (defaults to str(original))
        **kwargs: Additional arguments for the wrapper class

    Returns:
        New exception wrapping the original
    """
    msg = message or str(original)
    context = kwargs.pop("context", {})
    context["original_error"] = type(original).__name__
    context["original_message"] = str(original)

    return wrapper_class(
        message=msg,
        context=context,
        **kwargs
    )


ERROR_CODE_MAP: dict[str, type[SolanaTraderError]] = {
    "GENERAL_001": SolanaTraderError,
    "CONFIG_001": ConfigurationError,
    "INIT_001": InitializationError,

    "WALLET_000": WalletError,
    "WALLET_001": KeyGenerationError,
    "WALLET_002": EncryptionError,
    "WALLET_003": DecryptionError,
    "WALLET_004": WalletNotFoundError,
    "WALLET_005": WalletLockedError,
    "WALLET_006": WalletAlreadyExistsError,
    "WALLET_007": InvalidMnemonicError,
    "WALLET_008": InvalidPrivateKeyError,
    "WALLET_009": InsufficientBalanceError,
    "WALLET_010": WalletCorruptedError,

    "TX_000": TransactionError,
    "TX_001": TransactionBuildError,
    "TX_002": TransactionSignError,
    "TX_003": TransactionSendError,
    "TX_004": TransactionConfirmationError,
    "TX_005": TransactionSimulationError,
    "TX_006": TransactionExpiredError,
    "TX_007": TransactionAlreadyProcessedError,
    "TX_008": InstructionError,
    "TX_009": PriorityFeeError,

    "JUP_000": JupiterError,
    "JUP_001": QuoteError,
    "JUP_002": SwapError,
    "JUP_003": TokenNotFoundError,
    "JUP_004": InsufficientLiquidityError,
    "JUP_005": SlippageExceededError,
    "JUP_006": RouteNotFoundError,
    "JUP_007": JupiterAPIError,
    "JUP_008": PriceImpactTooHighError,

    "RPC_000": RPCError,
    "RPC_001": RPCConnectionError,
    "RPC_002": RPCTimeoutError,
    "RPC_003": RPCRateLimitError,
    "RPC_004": RPCResponseError,
    "RPC_005": RPCNodeUnhealthyError,
    "RPC_006": RPCMethodNotFoundError,
    "RPC_007": WebSocketError,
    "RPC_008": BlockhashNotFoundError,

    "VAL_000": ValidationError,
    "VAL_001": InvalidAddressError,
    "VAL_002": InvalidAmountError,
    "VAL_003": InvalidSlippageError,
    "VAL_004": InvalidTokenError,
    "VAL_005": InvalidSignatureError,
    "VAL_006": InvalidConfigError,
    "VAL_007": InvalidTimestampError,

    "RATE_000": RateLimitError,
    "RATE_001": RateLimitExceededError,
    "RATE_002": BurstLimitExceededError,
    "RATE_003": DailyLimitExceededError,
    "RATE_004": ConcurrencyLimitError,

    "SEC_000": SecurityError,
    "SEC_001": SecurityViolationError,
    "SEC_002": AuditIntegrityError,
    "SEC_003": UnauthorizedAccessError,
    "SEC_004": SuspiciousActivityError,
    "SEC_005": TokenValidationError,
    "SEC_006": MaxExposureExceededError,
    "SEC_007": SandboxViolationError,

    "NET_000": NetworkError,
    "NET_001": HTTPError,
    "NET_002": ConnectionPoolExhaustedError,
    "NET_003": DNSResolutionError,
    "NET_004": SSLError,

    "DATA_000": DataError,
    "DATA_001": SerializationError,
    "DATA_002": DeserializationError,
    "DATA_003": CacheError,
    "DATA_004": StorageError,
}


__all__ = [

    "SolanaTraderError",
    "ConfigurationError",
    "InitializationError",

    "WalletError",
    "KeyGenerationError",
    "EncryptionError",
    "DecryptionError",
    "WalletNotFoundError",
    "WalletLockedError",
    "WalletAlreadyExistsError",
    "InvalidMnemonicError",
    "InvalidPrivateKeyError",
    "InsufficientBalanceError",
    "WalletCorruptedError",

    "TransactionError",
    "TransactionBuildError",
    "TransactionSignError",
    "TransactionSendError",
    "TransactionConfirmationError",
    "TransactionSimulationError",
    "TransactionExpiredError",
    "TransactionAlreadyProcessedError",
    "InstructionError",
    "PriorityFeeError",

    "JupiterError",
    "QuoteError",
    "SwapError",
    "TokenNotFoundError",
    "InsufficientLiquidityError",
    "SlippageExceededError",
    "RouteNotFoundError",
    "JupiterAPIError",
    "PriceImpactTooHighError",

    "RPCError",
    "RPCConnectionError",
    "RPCTimeoutError",
    "RPCRateLimitError",
    "RPCResponseError",
    "RPCNodeUnhealthyError",
    "RPCMethodNotFoundError",
    "WebSocketError",
    "BlockhashNotFoundError",

    "ValidationError",
    "InvalidAddressError",
    "InvalidAmountError",
    "InvalidSlippageError",
    "InvalidTokenError",
    "InvalidSignatureError",
    "InvalidConfigError",
    "InvalidTimestampError",

    "RateLimitError",
    "RateLimitExceededError",
    "BurstLimitExceededError",
    "DailyLimitExceededError",
    "ConcurrencyLimitError",

    "SecurityError",
    "SecurityViolationError",
    "AuditIntegrityError",
    "UnauthorizedAccessError",
    "SuspiciousActivityError",
    "TokenValidationError",
    "MaxExposureExceededError",
    "SandboxViolationError",

    "NetworkError",
    "HTTPError",
    "ConnectionPoolExhaustedError",
    "DNSResolutionError",
    "SSLError",

    "DataError",
    "SerializationError",
    "DeserializationError",
    "CacheError",
    "StorageError",

    "is_retryable",
    "get_retry_delay",
    "wrap_exception",
    "ERROR_CODE_MAP",
]
