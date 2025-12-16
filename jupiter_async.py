"""
Jupiter DEX Async Client - Production-Grade Implementation
Full async integration with Jupiter aggregator for Solana token swaps.
"""

import asyncio
import aiohttp
import time
import logging
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, TypeVar, Generic
from enum import Enum
from collections import OrderedDict
from functools import wraps
import base64
import json

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

JUPITER_API_BASE = "https://quote-api.jup.ag/v6"
JUPITER_PRICE_API = "https://price.jup.ag/v6"
JUPITER_TOKEN_API = "https://token.jup.ag"

# Well-known token mints
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"

# Default configuration
DEFAULT_SLIPPAGE_BPS = 50  # 0.5%
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0

# Cache TTLs (seconds)
TOKEN_LIST_CACHE_TTL = 300  # 5 minutes
PRICE_CACHE_TTL = 30  # 30 seconds
QUOTE_CACHE_TTL = 5  # 5 seconds


# ============================================================================
# EXCEPTIONS
# ============================================================================

class JupiterError(Exception):
    """Base exception for Jupiter client errors."""
    
    def __init__(self, message: str, code: Optional[str] = None, details: Optional[Dict] = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class JupiterAPIError(JupiterError):
    """Error returned by Jupiter API."""
    
    def __init__(self, message: str, status_code: int, response_body: Optional[str] = None):
        super().__init__(message, code=f"HTTP_{status_code}")
        self.status_code = status_code
        self.response_body = response_body


class JupiterQuoteError(JupiterError):
    """Error getting quote from Jupiter."""
    pass


class JupiterSwapError(JupiterError):
    """Error executing swap through Jupiter."""
    pass


class JupiterRateLimitError(JupiterError):
    """Rate limit exceeded."""
    
    def __init__(self, retry_after: Optional[float] = None):
        super().__init__("Rate limit exceeded", code="RATE_LIMITED")
        self.retry_after = retry_after


class JupiterTimeoutError(JupiterError):
    """Request timeout."""
    pass


class JupiterConnectionError(JupiterError):
    """Connection error."""
    pass


class CircuitBreakerOpenError(JupiterError):
    """Circuit breaker is open due to repeated failures."""
    pass


# ============================================================================
# ENUMS
# ============================================================================

class SwapMode(str, Enum):
    """Swap mode for Jupiter quotes."""
    EXACT_IN = "ExactIn"
    EXACT_OUT = "ExactOut"


class DexId(str, Enum):
    """Supported DEX IDs on Jupiter."""
    RAYDIUM = "Raydium"
    RAYDIUM_CLMM = "Raydium CLMM"
    ORCA = "Orca"
    ORCA_WHIRLPOOL = "Whirlpool"
    METEORA = "Meteora"
    METEORA_DLMM = "Meteora DLMM"
    PHOENIX = "Phoenix"
    LIFINITY = "Lifinity"
    OPENBOOK = "OpenBook"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TokenInfo:
    """Information about a token."""
    address: str
    symbol: str
    name: str
    decimals: int
    logo_uri: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    extensions: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenInfo":
        return cls(
            address=data.get("address", ""),
            symbol=data.get("symbol", ""),
            name=data.get("name", ""),
            decimals=data.get("decimals", 0),
            logo_uri=data.get("logoURI"),
            tags=data.get("tags", []),
            extensions=data.get("extensions", {})
        )


@dataclass
class TokenPrice:
    """Token price information."""
    mint: str
    price_usd: float
    price_sol: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    confidence: Optional[str] = None
    
    @classmethod
    def from_dict(cls, mint: str, data: Dict[str, Any]) -> "TokenPrice":
        return cls(
            mint=mint,
            price_usd=float(data.get("price", 0)),
            confidence=data.get("confidence")
        )


@dataclass
class RoutePlan:
    """Single route plan in a swap."""
    swap_info: Dict[str, Any]
    percent: int
    
    @property
    def amm_key(self) -> str:
        return self.swap_info.get("ammKey", "")
    
    @property
    def label(self) -> str:
        return self.swap_info.get("label", "")
    
    @property
    def input_mint(self) -> str:
        return self.swap_info.get("inputMint", "")
    
    @property
    def output_mint(self) -> str:
        return self.swap_info.get("outputMint", "")
    
    @property
    def in_amount(self) -> int:
        return int(self.swap_info.get("inAmount", 0))
    
    @property
    def out_amount(self) -> int:
        return int(self.swap_info.get("outAmount", 0))
    
    @property
    def fee_amount(self) -> int:
        return int(self.swap_info.get("feeAmount", 0))
    
    @property
    def fee_mint(self) -> str:
        return self.swap_info.get("feeMint", "")


@dataclass
class SwapQuote:
    """Quote response from Jupiter."""
    input_mint: str
    output_mint: str
    in_amount: int
    out_amount: int
    other_amount_threshold: int
    swap_mode: SwapMode
    slippage_bps: int
    price_impact_pct: float
    route_plan: List[RoutePlan]
    context_slot: Optional[int] = None
    time_taken: Optional[float] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def price(self) -> float:
        """Calculate effective price (output per input)."""
        if self.in_amount == 0:
            return 0.0
        return self.out_amount / self.in_amount
    
    @property
    def minimum_received(self) -> int:
        """Minimum amount received after slippage."""
        return self.other_amount_threshold
    
    @property
    def num_hops(self) -> int:
        """Number of hops in the route."""
        return len(self.route_plan)
    
    @property
    def dexes_used(self) -> List[str]:
        """List of DEXes used in this route."""
        return list(set(r.label for r in self.route_plan))
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SwapQuote":
        route_plan = [
            RoutePlan(
                swap_info=r.get("swapInfo", {}),
                percent=r.get("percent", 100)
            )
            for r in data.get("routePlan", [])
        ]
        
        return cls(
            input_mint=data.get("inputMint", ""),
            output_mint=data.get("outputMint", ""),
            in_amount=int(data.get("inAmount", 0)),
            out_amount=int(data.get("outAmount", 0)),
            other_amount_threshold=int(data.get("otherAmountThreshold", 0)),
            swap_mode=SwapMode(data.get("swapMode", "ExactIn")),
            slippage_bps=int(data.get("slippageBps", DEFAULT_SLIPPAGE_BPS)),
            price_impact_pct=float(data.get("priceImpactPct", 0)),
            route_plan=route_plan,
            context_slot=data.get("contextSlot"),
            time_taken=data.get("timeTaken"),
            raw_response=data
        )


@dataclass
class SwapTransaction:
    """Swap transaction ready for signing."""
    swap_transaction: str  # Base64 encoded transaction
    last_valid_block_height: int
    priority_fee_lamports: int = 0
    compute_unit_limit: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SwapTransaction":
        return cls(
            swap_transaction=data.get("swapTransaction", ""),
            last_valid_block_height=int(data.get("lastValidBlockHeight", 0)),
            priority_fee_lamports=int(data.get("prioritizationFeeLamports", 0)),
            compute_unit_limit=data.get("computeUnitLimit")
        )


@dataclass
class SwapResult:
    """Result of a swap execution."""
    signature: str
    input_mint: str
    output_mint: str
    in_amount: int
    out_amount: int
    fee_amount: int
    slot: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    error: Optional[str] = None


@dataclass
class RouteAnalysis:
    """Analysis of a swap route."""
    quote: SwapQuote
    effective_price: float
    price_impact_percent: float
    total_fees_usd: Optional[float] = None
    route_efficiency: float = 0.0
    num_hops: int = 0
    dexes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @classmethod
    def from_quote(cls, quote: SwapQuote, input_price_usd: float = 0, output_price_usd: float = 0) -> "RouteAnalysis":
        warnings = []
        
        # Check price impact
        if quote.price_impact_pct > 1.0:
            warnings.append(f"High price impact: {quote.price_impact_pct:.2f}%")
        if quote.price_impact_pct > 5.0:
            warnings.append("CRITICAL: Very high price impact!")
        
        # Check number of hops
        if quote.num_hops > 3:
            warnings.append(f"Complex route with {quote.num_hops} hops")
        
        # Calculate fees in USD if prices available
        total_fees_usd = None
        if input_price_usd > 0:
            total_fees = sum(r.fee_amount for r in quote.route_plan)
            total_fees_usd = (total_fees / 1e9) * input_price_usd
        
        return cls(
            quote=quote,
            effective_price=quote.price,
            price_impact_percent=quote.price_impact_pct,
            total_fees_usd=total_fees_usd,
            num_hops=quote.num_hops,
            dexes=quote.dexes_used,
            warnings=warnings
        )


@dataclass
class JupiterMetrics:
    """Metrics for monitoring Jupiter client."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    total_latency_ms: float = 0.0
    quotes_fetched: int = 0
    swaps_executed: int = 0
    swaps_succeeded: int = 0
    swaps_failed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    circuit_breaker_trips: int = 0
    
    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "rate_limited_requests": self.rate_limited_requests,
            "avg_latency_ms": self.avg_latency_ms,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "quotes_fetched": self.quotes_fetched,
            "swaps_executed": self.swaps_executed,
            "swaps_succeeded": self.swaps_succeeded,
            "swaps_failed": self.swaps_failed,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "circuit_breaker_trips": self.circuit_breaker_trips
        }


# ============================================================================
# CACHING
# ============================================================================

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """Entry in the cache with TTL."""
    value: T
    expires_at: float
    
    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class LRUCache(Generic[T]):
    """Memory-bounded LRU cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 60.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[T]:
        """Get value from cache if exists and not expired."""
        async with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            if entry.is_expired:
                del self._cache[key]
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return entry.value
    
    async def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Set value in cache with optional TTL override."""
        ttl = ttl if ttl is not None else self.default_ttl
        async with self._lock:
            # Remove oldest if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            
            self._cache[key] = CacheEntry(
                value=value,
                expires_at=time.time() + ttl
            )
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all entries."""
        async with self._lock:
            self._cache.clear()
    
    async def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        async with self._lock:
            expired_keys = [
                k for k, v in self._cache.items() if v.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)
    
    @property
    def size(self) -> int:
        return len(self._cache)


class JupiterCache:
    """Manages all caches for Jupiter client."""
    
    def __init__(
        self,
        token_list_ttl: float = TOKEN_LIST_CACHE_TTL,
        price_ttl: float = PRICE_CACHE_TTL,
        quote_ttl: float = QUOTE_CACHE_TTL,
        max_entries: int = 10000
    ):
        self.token_list_cache: LRUCache[List[TokenInfo]] = LRUCache(
            max_size=10, default_ttl=token_list_ttl
        )
        self.token_info_cache: LRUCache[TokenInfo] = LRUCache(
            max_size=max_entries, default_ttl=token_list_ttl
        )
        self.price_cache: LRUCache[TokenPrice] = LRUCache(
            max_size=max_entries, default_ttl=price_ttl
        )
        self.quote_cache: LRUCache[SwapQuote] = LRUCache(
            max_size=1000, default_ttl=quote_ttl
        )
        
        self._cleanup_task: Optional[asyncio.Task] = None
    
    def start_cleanup_task(self, interval: float = 60.0) -> None:
        """Start background task to cleanup expired entries."""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(interval)
                await self.cleanup_all()
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    def stop_cleanup_task(self) -> None:
        """Stop the cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None
    
    async def cleanup_all(self) -> Dict[str, int]:
        """Cleanup all caches. Returns counts of removed entries."""
        return {
            "token_list": await self.token_list_cache.cleanup_expired(),
            "token_info": await self.token_info_cache.cleanup_expired(),
            "price": await self.price_cache.cleanup_expired(),
            "quote": await self.quote_cache.cleanup_expired()
        }
    
    async def clear_all(self) -> None:
        """Clear all caches."""
        await self.token_list_cache.clear()
        await self.token_info_cache.clear()
        await self.price_cache.clear()
        await self.quote_cache.clear()
    
    @staticmethod
    def make_quote_key(
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int
    ) -> str:
        """Generate cache key for a quote request."""
        data = f"{input_mint}:{output_mint}:{amount}:{slippage_bps}"
        return hashlib.md5(data.encode()).hexdigest()


# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """Token bucket rate limiter with burst support."""
    
    def __init__(
        self,
        requests_per_second: float = 10.0,
        burst_size: int = 20
    ):
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.tokens = float(burst_size)
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a token. Returns True if acquired, False if timeout.
        
        Args:
            timeout: Maximum time to wait for a token (None = wait forever)
        """
        start_time = time.time()
        
        while True:
            async with self._lock:
                now = time.time()
                # Replenish tokens
                time_passed = now - self.last_update
                self.tokens = min(
                    self.burst_size,
                    self.tokens + time_passed * self.requests_per_second
                )
                self.last_update = now
                
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True
            
            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False
            
            # Wait for token replenishment
            wait_time = (1.0 - self.tokens) / self.requests_per_second
            await asyncio.sleep(min(wait_time, 0.1))
    
    async def wait(self) -> None:
        """Wait until a token is available."""
        await self.acquire(timeout=None)


class AdaptiveRateLimiter:
    """Rate limiter that adapts based on API response headers."""
    
    def __init__(
        self,
        initial_rps: float = 10.0,
        min_rps: float = 1.0,
        max_rps: float = 50.0
    ):
        self.current_rps = initial_rps
        self.min_rps = min_rps
        self.max_rps = max_rps
        self._limiter = RateLimiter(initial_rps, int(initial_rps * 2))
        self._lock = asyncio.Lock()
        
        # Track rate limit responses
        self._rate_limit_hits = 0
        self._successful_requests = 0
        self._last_adjustment = time.time()
    
    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire a token."""
        return await self._limiter.acquire(timeout)
    
    async def wait(self) -> None:
        """Wait for a token."""
        await self._limiter.wait()
    
    async def on_success(self) -> None:
        """Called when a request succeeds."""
        async with self._lock:
            self._successful_requests += 1
            await self._maybe_increase_rate()
    
    async def on_rate_limit(self, retry_after: Optional[float] = None) -> None:
        """Called when rate limited."""
        async with self._lock:
            self._rate_limit_hits += 1
            # Immediately reduce rate
            new_rps = max(self.min_rps, self.current_rps * 0.5)
            await self._set_rate(new_rps)
            
            if retry_after:
                await asyncio.sleep(retry_after)
    
    async def _maybe_increase_rate(self) -> None:
        """Gradually increase rate if successful."""
        now = time.time()
        # Only adjust every 10 seconds
        if now - self._last_adjustment < 10.0:
            return
        
        # If no rate limits in recent requests, increase
        if self._rate_limit_hits == 0 and self._successful_requests > 10:
            new_rps = min(self.max_rps, self.current_rps * 1.1)
            await self._set_rate(new_rps)
        
        self._rate_limit_hits = 0
        self._successful_requests = 0
        self._last_adjustment = now
    
    async def _set_rate(self, rps: float) -> None:
        """Update the rate limit."""
        self.current_rps = rps
        self._limiter = RateLimiter(rps, int(rps * 2))


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker for handling persistent failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> CircuitState:
        return self._state
    
    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED
    
    async def can_execute(self) -> bool:
        """Check if request can proceed."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout passed
                if self._last_failure_time is None:
                    return False
                
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info("Circuit breaker: OPEN -> HALF_OPEN")
                    return True
                return False
            
            # HALF_OPEN state
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False
    
    async def record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                if self._half_open_calls >= self.half_open_max_calls:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info("Circuit breaker: HALF_OPEN -> CLOSED")
            else:
                self._failure_count = 0
    
    async def record_failure(self) -> None:
        """Record a failed call."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning("Circuit breaker: HALF_OPEN -> OPEN (failure during recovery)")
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit breaker: CLOSED -> OPEN (threshold reached: {self._failure_count})")
    
    async def reset(self) -> None:
        """Reset the circuit breaker."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0


# ============================================================================
# JUPITER CLIENT
# ============================================================================

class JupiterClient:
    """
    Production-grade async Jupiter DEX client.
    
    Features:
    - Full async implementation with aiohttp
    - Connection pooling and session management
    - Intelligent caching with TTL
    - Rate limiting with adaptive backoff
    - Circuit breaker for fault tolerance
    - Comprehensive error handling
    - Detailed metrics and monitoring
    
    Example:
        async with JupiterClient() as jupiter:
            quote = await jupiter.get_quote(
                input_mint=SOL_MINT,
                output_mint=USDC_MINT,
                amount=1_000_000_000,  # 1 SOL in lamports
                slippage_bps=50
            )
            print(f"Expected output: {quote.out_amount / 1e6} USDC")
    """
    
    def __init__(
        self,
        api_base: str = JUPITER_API_BASE,
        price_api_base: str = JUPITER_PRICE_API,
        token_api_base: str = JUPITER_TOKEN_API,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        rate_limit_rps: float = 10.0,
        enable_cache: bool = True,
        enable_circuit_breaker: bool = True,
        referral_account: Optional[str] = None,
        referral_fee_bps: int = 0
    ):
        """
        Initialize Jupiter client.
        
        Args:
            api_base: Base URL for Jupiter quote/swap API
            price_api_base: Base URL for Jupiter price API
            token_api_base: Base URL for Jupiter token API
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Initial delay between retries
            rate_limit_rps: Requests per second limit
            enable_cache: Enable response caching
            enable_circuit_breaker: Enable circuit breaker
            referral_account: Referral account for fee sharing
            referral_fee_bps: Referral fee in basis points
        """
        self.api_base = api_base.rstrip("/")
        self.price_api_base = price_api_base.rstrip("/")
        self.token_api_base = token_api_base.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.referral_account = referral_account
        self.referral_fee_bps = referral_fee_bps
        
        # Session (created on first use or __aenter__)
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        
        # Components
        self._rate_limiter = AdaptiveRateLimiter(rate_limit_rps)
        self._circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None
        self._cache = JupiterCache() if enable_cache else None
        
        # Metrics
        self.metrics = JupiterMetrics()
        
        # Flags
        self._closed = False
    
    async def __aenter__(self) -> "JupiterClient":
        """Async context manager entry."""
        await self._ensure_session()
        if self._cache:
            self._cache.start_cleanup_task()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists."""
        async with self._session_lock:
            if self._session is None or self._session.closed:
                connector = aiohttp.TCPConnector(
                    limit=100,  # Connection pool size
                    limit_per_host=30,
                    ttl_dns_cache=300,
                    enable_cleanup_closed=True
                )
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                self._session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                        "User-Agent": "JupiterAsyncClient/1.0"
                    }
                )
            return self._session
    
    async def close(self) -> None:
        """Close the client and cleanup resources."""
        if self._closed:
            return
        
        self._closed = True
        
        if self._cache:
            self._cache.stop_cleanup_task()
        
        if self._session and not self._session.closed:
            await self._session.close()
            # Wait for graceful close
            await asyncio.sleep(0.25)
        
        logger.info("JupiterClient closed")
    
    async def _request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        retry_on_rate_limit: bool = True
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic, rate limiting, and circuit breaker.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full URL to request
            params: Query parameters
            json_data: JSON body for POST requests
            retry_on_rate_limit: Whether to retry on 429
            
        Returns:
            Parsed JSON response
            
        Raises:
            JupiterAPIError: On API errors
            JupiterRateLimitError: On rate limit (if not retrying)
            JupiterTimeoutError: On timeout
            JupiterConnectionError: On connection errors
            CircuitBreakerOpenError: If circuit breaker is open
        """
        session = await self._ensure_session()
        
        # Circuit breaker check
        if self._circuit_breaker and not await self._circuit_breaker.can_execute():
            self.metrics.circuit_breaker_trips += 1
            raise CircuitBreakerOpenError("Circuit breaker is open")
        
        last_error: Optional[Exception] = None
        
        for attempt in range(self.max_retries):
            # Rate limiting
            await self._rate_limiter.wait()
            
            start_time = time.time()
            self.metrics.total_requests += 1
            
            try:
                # Sanitized logging
                log_params = {k: v for k, v in (params or {}).items() if k not in ["private_key"]}
                logger.debug(f"Request {method} {url} params={log_params} attempt={attempt + 1}")
                
                async with session.request(
                    method,
                    url,
                    params=params,
                    json=json_data
                ) as response:
                    latency = (time.time() - start_time) * 1000
                    self.metrics.total_latency_ms += latency
                    
                    # Handle rate limiting
                    if response.status == 429:
                        self.metrics.rate_limited_requests += 1
                        retry_after = float(response.headers.get("Retry-After", self.retry_delay * (2 ** attempt)))
                        
                        await self._rate_limiter.on_rate_limit(retry_after)
                        
                        if retry_on_rate_limit and attempt < self.max_retries - 1:
                            logger.warning(f"Rate limited, retrying after {retry_after}s")
                            continue
                        raise JupiterRateLimitError(retry_after)
                    
                    # Parse response
                    try:
                        data = await response.json()
                    except Exception:
                        data = {"raw": await response.text()}
                    
                    # Handle errors
                    if response.status >= 400:
                        self.metrics.failed_requests += 1
                        if self._circuit_breaker:
                            await self._circuit_breaker.record_failure()
                        
                        error_msg = data.get("error", data.get("message", str(data)))
                        raise JupiterAPIError(
                            error_msg,
                            response.status,
                            await response.text() if isinstance(data, dict) and "raw" in data else None
                        )
                    
                    # Success
                    self.metrics.successful_requests += 1
                    await self._rate_limiter.on_success()
                    if self._circuit_breaker:
                        await self._circuit_breaker.record_success()
                    
                    logger.debug(f"Request completed in {latency:.1f}ms")
                    return data
            
            except aiohttp.ClientError as e:
                self.metrics.failed_requests += 1
                last_error = JupiterConnectionError(str(e))
                if self._circuit_breaker:
                    await self._circuit_breaker.record_failure()
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Connection error: {e}, retrying in {delay}s")
                    await asyncio.sleep(delay)
                    continue
            
            except asyncio.TimeoutError:
                self.metrics.failed_requests += 1
                last_error = JupiterTimeoutError(f"Request timed out after {self.timeout}s")
                if self._circuit_breaker:
                    await self._circuit_breaker.record_failure()
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Timeout, retrying in {delay}s")
                    await asyncio.sleep(delay)
                    continue
            
            except JupiterAPIError:
                raise
        
        if last_error:
            raise last_error
        raise JupiterError("Max retries exceeded")


    # ========================================================================
    # QUOTE OPERATIONS
    # ========================================================================
    
    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = DEFAULT_SLIPPAGE_BPS,
        swap_mode: SwapMode = SwapMode.EXACT_IN,
        only_direct_routes: bool = False,
        as_legacy_transaction: bool = False,
        max_accounts: Optional[int] = None,
        dexes: Optional[List[str]] = None,
        exclude_dexes: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> SwapQuote:
        """
        Get a swap quote from Jupiter.
        
        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address
            amount: Amount in smallest unit (lamports for SOL, etc.)
            slippage_bps: Slippage tolerance in basis points (50 = 0.5%)
            swap_mode: EXACT_IN or EXACT_OUT
            only_direct_routes: Only use direct routes (no hops)
            as_legacy_transaction: Return legacy transaction format
            max_accounts: Maximum accounts in transaction
            dexes: Only use these DEXes
            exclude_dexes: Exclude these DEXes
            use_cache: Use cached quote if available
            
        Returns:
            SwapQuote with route information
            
        Raises:
            JupiterQuoteError: If quote cannot be obtained
        """
        # Check cache
        if use_cache and self._cache:
            cache_key = JupiterCache.make_quote_key(input_mint, output_mint, amount, slippage_bps)
            cached = await self._cache.quote_cache.get(cache_key)
            if cached:
                self.metrics.cache_hits += 1
                return cached
            self.metrics.cache_misses += 1
        
        # Build params
        params: Dict[str, Any] = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "slippageBps": str(slippage_bps),
            "swapMode": swap_mode.value,
            "onlyDirectRoutes": str(only_direct_routes).lower(),
            "asLegacyTransaction": str(as_legacy_transaction).lower()
        }
        
        if max_accounts is not None:
            params["maxAccounts"] = str(max_accounts)
        
        if dexes:
            params["dexes"] = ",".join(dexes)
        
        if exclude_dexes:
            params["excludeDexes"] = ",".join(exclude_dexes)
        
        try:
            data = await self._request("GET", f"{self.api_base}/quote", params=params)
            quote = SwapQuote.from_dict(data)
            self.metrics.quotes_fetched += 1
            
            # Cache result
            if self._cache:
                await self._cache.quote_cache.set(cache_key, quote)
            
            return quote
        
        except JupiterAPIError as e:
            raise JupiterQuoteError(
                f"Failed to get quote: {e}",
                details={"input": input_mint, "output": output_mint, "amount": amount}
            ) from e
    
    async def get_quotes(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = DEFAULT_SLIPPAGE_BPS,
        count: int = 3,
        **kwargs
    ) -> List[SwapQuote]:
        """
        Get multiple quote options by trying different DEX combinations.
        
        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address
            amount: Amount in smallest unit
            slippage_bps: Slippage tolerance
            count: Number of quotes to try to get
            **kwargs: Additional arguments passed to get_quote
            
        Returns:
            List of SwapQuote sorted by best output amount
        """
        quotes = []
        
        # Get main quote
        try:
            main_quote = await self.get_quote(
                input_mint, output_mint, amount, slippage_bps, **kwargs
            )
            quotes.append(main_quote)
        except JupiterQuoteError:
            pass
        
        # Try direct routes only
        if len(quotes) < count:
            try:
                direct_quote = await self.get_quote(
                    input_mint, output_mint, amount, slippage_bps,
                    only_direct_routes=True, use_cache=False, **kwargs
                )
                if direct_quote.out_amount not in [q.out_amount for q in quotes]:
                    quotes.append(direct_quote)
            except JupiterQuoteError:
                pass
        
        # Try specific DEXes
        if len(quotes) < count:
            for dex in [DexId.RAYDIUM.value, DexId.ORCA_WHIRLPOOL.value]:
                if len(quotes) >= count:
                    break
                try:
                    dex_quote = await self.get_quote(
                        input_mint, output_mint, amount, slippage_bps,
                        dexes=[dex], use_cache=False, **kwargs
                    )
                    if dex_quote.out_amount not in [q.out_amount for q in quotes]:
                        quotes.append(dex_quote)
                except JupiterQuoteError:
                    continue
        
        # Sort by output amount (best first)
        quotes.sort(key=lambda q: q.out_amount, reverse=True)
        return quotes[:count]
    
    def select_best_quote(self, quotes: List[SwapQuote]) -> Optional[SwapQuote]:
        """
        Select the best quote from a list.
        
        Considers:
        - Output amount (higher is better)
        - Price impact (lower is better)
        - Number of hops (fewer is generally better)
        
        Args:
            quotes: List of quotes to compare
            
        Returns:
            Best quote or None if list is empty
        """
        if not quotes:
            return None
        
        def score(q: SwapQuote) -> float:
            # Primary: output amount
            output_score = q.out_amount
            # Penalty for high price impact
            impact_penalty = q.price_impact_pct * 0.01 * q.out_amount
            # Small penalty for complex routes
            hop_penalty = (q.num_hops - 1) * 0.001 * q.out_amount
            return output_score - impact_penalty - hop_penalty
        
        return max(quotes, key=score)


    # ========================================================================
    # SWAP OPERATIONS
    # ========================================================================
    
    async def get_swap_transaction(
        self,
        quote: SwapQuote,
        user_pubkey: str,
        wrap_unwrap_sol: bool = True,
        fee_account: Optional[str] = None,
        compute_unit_price_micro_lamports: Optional[int] = None,
        priority_level: Optional[str] = None,
        as_legacy_transaction: bool = False,
        use_shared_accounts: bool = True,
        destination_token_account: Optional[str] = None
    ) -> SwapTransaction:
        """
        Get a serialized swap transaction ready for signing.
        
        Args:
            quote: Quote from get_quote()
            user_pubkey: User's wallet public key
            wrap_unwrap_sol: Auto wrap/unwrap SOL
            fee_account: Token account to receive referral fees
            compute_unit_price_micro_lamports: Priority fee in micro-lamports
            priority_level: Priority level (min, low, medium, high, veryHigh, auto)
            as_legacy_transaction: Use legacy transaction format
            use_shared_accounts: Use shared intermediate token accounts
            destination_token_account: Specific destination token account
            
        Returns:
            SwapTransaction ready for signing
            
        Raises:
            JupiterSwapError: If transaction cannot be created
        """
        body: Dict[str, Any] = {
            "quoteResponse": quote.raw_response,
            "userPublicKey": user_pubkey,
            "wrapAndUnwrapSol": wrap_unwrap_sol,
            "useSharedAccounts": use_shared_accounts,
            "asLegacyTransaction": as_legacy_transaction
        }
        
        if fee_account or self.referral_account:
            body["feeAccount"] = fee_account or self.referral_account
        
        if compute_unit_price_micro_lamports is not None:
            body["computeUnitPriceMicroLamports"] = compute_unit_price_micro_lamports
        elif priority_level:
            body["prioritizationFeeLamports"] = {
                "priorityLevelWithMaxLamports": {
                    "maxLamports": 10000000,  # 0.01 SOL max
                    "priorityLevel": priority_level
                }
            }
        
        if destination_token_account:
            body["destinationTokenAccount"] = destination_token_account
        
        try:
            data = await self._request("POST", f"{self.api_base}/swap", json_data=body)
            return SwapTransaction.from_dict(data)
        
        except JupiterAPIError as e:
            raise JupiterSwapError(f"Failed to get swap transaction: {e}") from e
    
    async def execute_swap(
        self,
        quote: SwapQuote,
        wallet_keypair: Any,  # solders.Keypair or equivalent
        rpc_client: Any,  # AsyncClient
        priority_fee_lamports: Optional[int] = None,
        skip_preflight: bool = False,
        max_retries: int = 3
    ) -> SwapResult:
        """
        Execute a complete swap operation.
        
        Args:
            quote: Quote from get_quote()
            wallet_keypair: Wallet keypair for signing
            rpc_client: Solana RPC client
            priority_fee_lamports: Priority fee to add
            skip_preflight: Skip preflight simulation
            max_retries: Max transaction submission retries
            
        Returns:
            SwapResult with transaction details
            
        Raises:
            JupiterSwapError: If swap fails
            
        Note:
            Requires solders and solana-py to be installed.
        """
        self.metrics.swaps_executed += 1
        
        try:
            # Import solana libraries (optional dependency)
            from solders.transaction import VersionedTransaction
            from solders.signature import Signature
            from solana.rpc.commitment import Confirmed
            
            # Get user pubkey
            user_pubkey = str(wallet_keypair.pubkey())
            
            # Get swap transaction
            swap_tx = await self.get_swap_transaction(
                quote=quote,
                user_pubkey=user_pubkey,
                compute_unit_price_micro_lamports=priority_fee_lamports
            )
            
            # Deserialize and sign
            tx_bytes = base64.b64decode(swap_tx.swap_transaction)
            transaction = VersionedTransaction.from_bytes(tx_bytes)
            
            # Sign transaction
            signed_tx = VersionedTransaction(
                transaction.message,
                [wallet_keypair]
            )
            
            # Submit transaction
            for attempt in range(max_retries):
                try:
                    result = await rpc_client.send_transaction(
                        signed_tx,
                        opts={
                            "skip_preflight": skip_preflight,
                            "preflight_commitment": Confirmed,
                            "max_retries": 0  # We handle retries
                        }
                    )
                    
                    signature = str(result.value)
                    
                    # Confirm transaction
                    confirmation = await rpc_client.confirm_transaction(
                        Signature.from_string(signature),
                        commitment=Confirmed
                    )
                    
                    if confirmation.value[0].err:
                        raise JupiterSwapError(f"Transaction failed: {confirmation.value[0].err}")
                    
                    self.metrics.swaps_succeeded += 1
                    
                    return SwapResult(
                        signature=signature,
                        input_mint=quote.input_mint,
                        output_mint=quote.output_mint,
                        in_amount=quote.in_amount,
                        out_amount=quote.out_amount,
                        fee_amount=sum(r.fee_amount for r in quote.route_plan),
                        success=True
                    )
                
                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1 * (attempt + 1))
                        continue
                    raise
        
        except ImportError as e:
            raise JupiterSwapError(
                "solders and solana-py required for swap execution. "
                "Install with: pip install solders solana"
            ) from e
        
        except Exception as e:
            self.metrics.swaps_failed += 1
            return SwapResult(
                signature="",
                input_mint=quote.input_mint,
                output_mint=quote.output_mint,
                in_amount=quote.in_amount,
                out_amount=0,
                fee_amount=0,
                success=False,
                error=str(e)
            )


    # ========================================================================
    # TOKEN OPERATIONS
    # ========================================================================
    
    async def get_token_list(self, use_cache: bool = True) -> List[TokenInfo]:
        """
        Get list of all tokens supported by Jupiter.
        
        Args:
            use_cache: Use cached token list if available
            
        Returns:
            List of TokenInfo
        """
        cache_key = "all_tokens"
        
        if use_cache and self._cache:
            cached = await self._cache.token_list_cache.get(cache_key)
            if cached:
                self.metrics.cache_hits += 1
                return cached
            self.metrics.cache_misses += 1
        
        data = await self._request("GET", f"{self.token_api_base}/all")
        tokens = [TokenInfo.from_dict(t) for t in data]
        
        if self._cache:
            await self._cache.token_list_cache.set(cache_key, tokens)
            # Also cache individual tokens
            for token in tokens:
                await self._cache.token_info_cache.set(token.address, token)
        
        return tokens
    
    async def get_token_info(self, mint: str, use_cache: bool = True) -> Optional[TokenInfo]:
        """
        Get information about a specific token.
        
        Args:
            mint: Token mint address
            use_cache: Use cached info if available
            
        Returns:
            TokenInfo or None if not found
        """
        if use_cache and self._cache:
            cached = await self._cache.token_info_cache.get(mint)
            if cached:
                self.metrics.cache_hits += 1
                return cached
            self.metrics.cache_misses += 1
        
        # Try to fetch from strict list first
        try:
            data = await self._request("GET", f"{self.token_api_base}/strict")
            for token_data in data:
                if token_data.get("address") == mint:
                    token = TokenInfo.from_dict(token_data)
                    if self._cache:
                        await self._cache.token_info_cache.set(mint, token)
                    return token
        except Exception:
            pass
        
        # Fall back to all tokens
        tokens = await self.get_token_list(use_cache=use_cache)
        for token in tokens:
            if token.address == mint:
                return token
        
        return None
    
    async def search_tokens(self, query: str, limit: int = 10) -> List[TokenInfo]:
        """
        Search tokens by name or symbol.
        
        Args:
            query: Search query (name or symbol)
            limit: Maximum results to return
            
        Returns:
            List of matching TokenInfo
        """
        query_lower = query.lower()
        tokens = await self.get_token_list()
        
        matches = []
        for token in tokens:
            if (query_lower in token.symbol.lower() or 
                query_lower in token.name.lower() or
                query_lower == token.address.lower()):
                matches.append(token)
                if len(matches) >= limit:
                    break
        
        # Sort: exact symbol match first, then by name
        def sort_key(t: TokenInfo) -> tuple:
            exact_symbol = t.symbol.lower() == query_lower
            starts_with = t.symbol.lower().startswith(query_lower)
            return (not exact_symbol, not starts_with, t.symbol)
        
        matches.sort(key=sort_key)
        return matches[:limit]
    
    async def get_token_price(
        self,
        mint: str,
        vs_currency: str = "usd",
        use_cache: bool = True
    ) -> Optional[TokenPrice]:
        """
        Get current token price.
        
        Args:
            mint: Token mint address
            vs_currency: Currency to price against (usd, sol)
            use_cache: Use cached price if available
            
        Returns:
            TokenPrice or None if not available
        """
        cache_key = f"price:{mint}:{vs_currency}"
        
        if use_cache and self._cache:
            cached = await self._cache.price_cache.get(cache_key)
            if cached:
                self.metrics.cache_hits += 1
                return cached
            self.metrics.cache_misses += 1
        
        try:
            params = {"ids": mint}
            if vs_currency.lower() == "sol":
                params["vsToken"] = SOL_MINT
            
            data = await self._request("GET", f"{self.price_api_base}/price", params=params)
            
            price_data = data.get("data", {}).get(mint)
            if not price_data:
                return None
            
            price = TokenPrice.from_dict(mint, price_data)
            
            if self._cache:
                await self._cache.price_cache.set(cache_key, price)
            
            return price
        
        except Exception as e:
            logger.warning(f"Failed to get price for {mint}: {e}")
            return None
    
    async def get_token_prices(
        self,
        mints: List[str],
        vs_currency: str = "usd"
    ) -> Dict[str, TokenPrice]:
        """
        Get prices for multiple tokens in a single request.
        
        Args:
            mints: List of token mint addresses
            vs_currency: Currency to price against
            
        Returns:
            Dict mapping mint to TokenPrice
        """
        if not mints:
            return {}
        
        try:
            params = {"ids": ",".join(mints)}
            if vs_currency.lower() == "sol":
                params["vsToken"] = SOL_MINT
            
            data = await self._request("GET", f"{self.price_api_base}/price", params=params)
            
            results = {}
            for mint, price_data in data.get("data", {}).items():
                if price_data:
                    price = TokenPrice.from_dict(mint, price_data)
                    results[mint] = price
                    
                    if self._cache:
                        cache_key = f"price:{mint}:{vs_currency}"
                        await self._cache.price_cache.set(cache_key, price)
            
            return results
        
        except Exception as e:
            logger.warning(f"Failed to get prices: {e}")
            return {}


    # ========================================================================
    # ROUTE ANALYSIS
    # ========================================================================
    
    async def analyze_route(self, quote: SwapQuote) -> RouteAnalysis:
        """
        Analyze a swap route for efficiency and risks.
        
        Args:
            quote: Quote to analyze
            
        Returns:
            RouteAnalysis with detailed breakdown
        """
        # Try to get prices for better analysis
        input_price = await self.get_token_price(quote.input_mint)
        output_price = await self.get_token_price(quote.output_mint)
        
        return RouteAnalysis.from_quote(
            quote,
            input_price.price_usd if input_price else 0,
            output_price.price_usd if output_price else 0
        )
    
    async def compare_routes(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = DEFAULT_SLIPPAGE_BPS
    ) -> List[RouteAnalysis]:
        """
        Get and analyze multiple routes for comparison.
        
        Args:
            input_mint: Input token mint
            output_mint: Output token mint
            amount: Amount to swap
            slippage_bps: Slippage tolerance
            
        Returns:
            List of RouteAnalysis sorted by best output
        """
        quotes = await self.get_quotes(
            input_mint, output_mint, amount, slippage_bps, count=5
        )
        
        analyses = []
        for quote in quotes:
            analysis = await self.analyze_route(quote)
            analyses.append(analysis)
        
        # Sort by effective output (accounting for warnings)
        analyses.sort(key=lambda a: a.quote.out_amount, reverse=True)
        return analyses
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current client metrics."""
        return self.metrics.to_dict()
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics = JupiterMetrics()
    
    async def health_check(self) -> bool:
        """
        Check if Jupiter API is accessible.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to get a simple quote
            await self.get_quote(
                SOL_MINT,
                USDC_MINT,
                1_000_000,  # 0.001 SOL
                slippage_bps=100,
                use_cache=False
            )
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def quick_quote(
    input_mint: str,
    output_mint: str,
    amount: int,
    slippage_bps: int = DEFAULT_SLIPPAGE_BPS
) -> SwapQuote:
    """
    Quick function to get a single quote without managing client lifecycle.
    
    Args:
        input_mint: Input token mint
        output_mint: Output token mint
        amount: Amount in smallest unit
        slippage_bps: Slippage tolerance
        
    Returns:
        SwapQuote
    """
    async with JupiterClient() as client:
        return await client.get_quote(input_mint, output_mint, amount, slippage_bps)


async def get_sol_to_usdc_price() -> float:
    """Get current SOL price in USDC."""
    async with JupiterClient() as client:
        quote = await client.get_quote(
            SOL_MINT,
            USDC_MINT,
            1_000_000_000,  # 1 SOL
            slippage_bps=10
        )
        # USDC has 6 decimals
        return quote.out_amount / 1_000_000


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Main client
    "JupiterClient",
    
    # Data classes
    "TokenInfo",
    "TokenPrice",
    "SwapQuote",
    "SwapTransaction",
    "SwapResult",
    "RoutePlan",
    "RouteAnalysis",
    "JupiterMetrics",
    
    # Enums
    "SwapMode",
    "DexId",
    
    # Exceptions
    "JupiterError",
    "JupiterAPIError",
    "JupiterQuoteError",
    "JupiterSwapError",
    "JupiterRateLimitError",
    "JupiterTimeoutError",
    "JupiterConnectionError",
    "CircuitBreakerOpenError",
    
    # Constants
    "SOL_MINT",
    "USDC_MINT",
    "USDT_MINT",
    "DEFAULT_SLIPPAGE_BPS",
    
    # Convenience functions
    "quick_quote",
    "get_sol_to_usdc_price",
]
