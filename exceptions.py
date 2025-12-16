
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime, timezone

@dataclass
class SolanaTraderError(Exception):
    message: str
    error_code: str = "GENERAL_001"
    context: dict[str, Any] = field(default_factory=dict)
    is_recoverable: bool = False
    retry_after: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self) -> None:
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        base = f"[{self.error_code}] {self.message}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base += f" | Context: {context_str}"
        return base
    
    def to_dict(self) -> dict[str, Any]:
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
    error_code: str = "CONFIG_001"
    is_recoverable: bool = False

@dataclass  
class InitializationError(SolanaTraderError):
    error_code: str = "INIT_001"
    is_recoverable: bool = False

@dataclass
class WalletError(SolanaTraderError):
    error_code: str = "WALLET_000"

@dataclass
class KeyGenerationError(WalletError):
    error_code: str = "WALLET_001"
    is_recoverable: bool = True

@dataclass
class EncryptionError(WalletError):
    error_code: str = "WALLET_002"
    is_recoverable: bool = False

@dataclass
class DecryptionError(WalletError):
    error_code: str = "WALLET_003"
    is_recoverable: bool = False

@dataclass
class WalletNotFoundError(WalletError):
    error_code: str = "WALLET_004"
    is_recoverable: bool = False

@dataclass
class WalletLockedError(WalletError):
    error_code: str = "WALLET_005"
    is_recoverable: bool = True

@dataclass
class WalletAlreadyExistsError(WalletError):
    error_code: str = "WALLET_006"
    is_recoverable: bool = False

@dataclass
class InvalidMnemonicError(WalletError):
    error_code: str = "WALLET_007"
    is_recoverable: bool = False

@dataclass
class InvalidPrivateKeyError(WalletError):
    error_code: str = "WALLET_008"
    is_recoverable: bool = False

@dataclass
class InsufficientBalanceError(WalletError):
    error_code: str = "WALLET_009"
    is_recoverable: bool = True
    required_amount: Optional[float] = None
    available_amount: Optional[float] = None
    token_mint: Optional[str] = None

@dataclass
class WalletCorruptedError(WalletError):
    error_code: str = "WALLET_010"
    is_recoverable: bool = False

@dataclass
class TransactionError(SolanaTraderError):
    error_code: str = "TX_000"
    transaction_signature: Optional[str] = None

@dataclass
class TransactionBuildError(TransactionError):
    error_code: str = "TX_001"
    is_recoverable: bool = True

@dataclass
class TransactionSignError(TransactionError):
    error_code: str = "TX_002"
    is_recoverable: bool = True

@dataclass
class TransactionSendError(TransactionError):
    error_code: str = "TX_003"
    is_recoverable: bool = True

@dataclass
class TransactionConfirmationError(TransactionError):
    error_code: str = "TX_004"
    is_recoverable: bool = True
    confirmation_timeout: Optional[float] = None

@dataclass
class TransactionSimulationError(TransactionError):
    error_code: str = "TX_005"
    is_recoverable: bool = False
    simulation_logs: list[str] = field(default_factory=list)

@dataclass
class TransactionExpiredError(TransactionError):
    error_code: str = "TX_006"
    is_recoverable: bool = True

@dataclass
class TransactionAlreadyProcessedError(TransactionError):
    error_code: str = "TX_007"
    is_recoverable: bool = False

@dataclass
class InstructionError(TransactionError):
    error_code: str = "TX_008"
    is_recoverable: bool = False
    program_id: Optional[str] = None
    instruction_index: Optional[int] = None
    program_error: Optional[str] = None

@dataclass
class PriorityFeeError(TransactionError):
    error_code: str = "TX_009"
    is_recoverable: bool = True

@dataclass
class JupiterError(SolanaTraderError):
    error_code: str = "JUP_000"

@dataclass
class QuoteError(JupiterError):
    error_code: str = "JUP_001"
    is_recoverable: bool = True
    input_mint: Optional[str] = None
    output_mint: Optional[str] = None
    amount: Optional[int] = None

@dataclass
class SwapError(JupiterError):
    error_code: str = "JUP_002"
    is_recoverable: bool = True

@dataclass
class TokenNotFoundError(JupiterError):
    error_code: str = "JUP_003"
    is_recoverable: bool = False
    token_mint: Optional[str] = None

@dataclass
class InsufficientLiquidityError(JupiterError):
    error_code: str = "JUP_004"
    is_recoverable: bool = True
    available_liquidity: Optional[float] = None
    required_liquidity: Optional[float] = None

@dataclass
class SlippageExceededError(JupiterError):
    error_code: str = "JUP_005"
    is_recoverable: bool = True
    expected_amount: Optional[int] = None
    actual_amount: Optional[int] = None
    slippage_bps: Optional[int] = None

@dataclass
class RouteNotFoundError(JupiterError):
    error_code: str = "JUP_006"
    is_recoverable: bool = True

@dataclass
class JupiterAPIError(JupiterError):
    error_code: str = "JUP_007"
    is_recoverable: bool = True
    status_code: Optional[int] = None
    api_error_message: Optional[str] = None

@dataclass
class PriceImpactTooHighError(JupiterError):
    error_code: str = "JUP_008"
    is_recoverable: bool = True
    price_impact_percent: Optional[float] = None
    max_allowed_impact: Optional[float] = None

@dataclass
class RPCError(SolanaTraderError):
    error_code: str = "RPC_000"
    rpc_endpoint: Optional[str] = None

@dataclass
class RPCConnectionError(RPCError):
    error_code: str = "RPC_001"
    is_recoverable: bool = True

@dataclass
class RPCTimeoutError(RPCError):
    error_code: str = "RPC_002"
    is_recoverable: bool = True
    timeout_seconds: Optional[float] = None

@dataclass
class RPCRateLimitError(RPCError):
    error_code: str = "RPC_003"
    is_recoverable: bool = True
    retry_after: Optional[float] = 60.0

@dataclass
class RPCResponseError(RPCError):
    error_code: str = "RPC_004"
    is_recoverable: bool = True
    rpc_error_code: Optional[int] = None
    rpc_error_message: Optional[str] = None

@dataclass
class RPCNodeUnhealthyError(RPCError):
    error_code: str = "RPC_005"
    is_recoverable: bool = True
    node_slot: Optional[int] = None
    network_slot: Optional[int] = None

@dataclass
class RPCMethodNotFoundError(RPCError):
    error_code: str = "RPC_006"
    is_recoverable: bool = False
    method_name: Optional[str] = None

@dataclass
class WebSocketError(RPCError):
    error_code: str = "RPC_007"
    is_recoverable: bool = True

@dataclass
class BlockhashNotFoundError(RPCError):
    error_code: str = "RPC_008"
    is_recoverable: bool = True

@dataclass
class ValidationError(SolanaTraderError):
    error_code: str = "VAL_000"
    is_recoverable: bool = False

@dataclass
class InvalidAddressError(ValidationError):
    error_code: str = "VAL_001"
    invalid_address: Optional[str] = None

@dataclass
class InvalidAmountError(ValidationError):
    error_code: str = "VAL_002"
    amount: Optional[float] = None
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None

@dataclass
class InvalidSlippageError(ValidationError):
    error_code: str = "VAL_003"
    slippage_bps: Optional[int] = None
    min_slippage: Optional[int] = None
    max_slippage: Optional[int] = None

@dataclass
class InvalidTokenError(ValidationError):
    error_code: str = "VAL_004"
    token: Optional[str] = None

@dataclass
class InvalidSignatureError(ValidationError):
    error_code: str = "VAL_005"
    signature: Optional[str] = None

@dataclass
class InvalidConfigError(ValidationError):
    error_code: str = "VAL_006"
    config_key: Optional[str] = None
    config_value: Optional[Any] = None
    expected_type: Optional[str] = None

@dataclass
class InvalidTimestampError(ValidationError):
    error_code: str = "VAL_007"
    timestamp: Optional[float] = None

@dataclass
class RateLimitError(SolanaTraderError):
    error_code: str = "RATE_000"
    is_recoverable: bool = True
    retry_after: Optional[float] = 1.0

@dataclass
class RateLimitExceededError(RateLimitError):
    error_code: str = "RATE_001"
    limit_name: Optional[str] = None
    current_rate: Optional[float] = None
    max_rate: Optional[float] = None

@dataclass
class BurstLimitExceededError(RateLimitError):
    error_code: str = "RATE_002"
    burst_count: Optional[int] = None
    burst_limit: Optional[int] = None
    window_seconds: Optional[float] = None

@dataclass
class DailyLimitExceededError(RateLimitError):
    error_code: str = "RATE_003"
    retry_after: Optional[float] = 86400.0
    daily_count: Optional[int] = None
    daily_limit: Optional[int] = None

@dataclass
class ConcurrencyLimitError(RateLimitError):
    error_code: str = "RATE_004"
    current_concurrent: Optional[int] = None
    max_concurrent: Optional[int] = None

@dataclass
class SecurityError(SolanaTraderError):
    error_code: str = "SEC_000"
    is_recoverable: bool = False

@dataclass
class SecurityViolationError(SecurityError):
    error_code: str = "SEC_001"
    violation_type: Optional[str] = None
    blocked_action: Optional[str] = None

@dataclass
class AuditIntegrityError(SecurityError):
    error_code: str = "SEC_002"
    expected_hash: Optional[str] = None
    actual_hash: Optional[str] = None

@dataclass
class UnauthorizedAccessError(SecurityError):
    error_code: str = "SEC_003"
    resource: Optional[str] = None
    required_permission: Optional[str] = None

@dataclass
class SuspiciousActivityError(SecurityError):
    error_code: str = "SEC_004"
    activity_type: Optional[str] = None
    details: Optional[str] = None

@dataclass
class TokenValidationError(SecurityError):
    error_code: str = "SEC_005"
    token_mint: Optional[str] = None
    validation_failure: Optional[str] = None

@dataclass
class MaxExposureExceededError(SecurityError):
    error_code: str = "SEC_006"
    is_recoverable: bool = True
    current_exposure: Optional[float] = None
    max_exposure: Optional[float] = None
    requested_amount: Optional[float] = None

@dataclass
class SandboxViolationError(SecurityError):
    error_code: str = "SEC_007"
    attempted_action: Optional[str] = None

@dataclass
class NetworkError(SolanaTraderError):
    error_code: str = "NET_000"
    is_recoverable: bool = True

@dataclass
class HTTPError(NetworkError):
    error_code: str = "NET_001"
    status_code: Optional[int] = None
    url: Optional[str] = None

@dataclass
class ConnectionPoolExhaustedError(NetworkError):
    error_code: str = "NET_002"
    pool_size: Optional[int] = None

@dataclass
class DNSResolutionError(NetworkError):
    error_code: str = "NET_003"
    hostname: Optional[str] = None

@dataclass
class SSLError(NetworkError):
    error_code: str = "NET_004"

@dataclass
class DataError(SolanaTraderError):
    error_code: str = "DATA_000"

@dataclass
class SerializationError(DataError):
    error_code: str = "DATA_001"
    is_recoverable: bool = False
    data_type: Optional[str] = None

@dataclass
class DeserializationError(DataError):
    error_code: str = "DATA_002"
    is_recoverable: bool = False
    data_type: Optional[str] = None

@dataclass
class CacheError(DataError):
    error_code: str = "DATA_003"
    is_recoverable: bool = True
    cache_key: Optional[str] = None

@dataclass
class StorageError(DataError):
    error_code: str = "DATA_004"
    is_recoverable: bool = True
    storage_path: Optional[str] = None

def is_retryable(error: Exception) -> bool:
    if isinstance(error, SolanaTraderError):
        return error.is_recoverable
    return False

def get_retry_delay(error: Exception, default: float = 1.0) -> float:
    if isinstance(error, SolanaTraderError) and error.retry_after is not None:
        return error.retry_after
    return default

def wrap_exception(
    original: Exception,
    wrapper_class: type[SolanaTraderError],
    message: Optional[str] = None,
    **kwargs: Any
) -> SolanaTraderError:
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
