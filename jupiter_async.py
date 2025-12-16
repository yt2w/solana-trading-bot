
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

logger = logging.getLogger(__name__)

JUPITER_API_BASE = "https://quote-api.jup.ag/v6"
JUPITER_PRICE_API = "https://price.jup.ag/v6"
JUPITER_TOKEN_API = "https://token.jup.ag"

SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"

DEFAULT_SLIPPAGE_BPS = 50
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0

TOKEN_LIST_CACHE_TTL = 300
PRICE_CACHE_TTL = 30
QUOTE_CACHE_TTL = 5

class JupiterError(Exception):
    
    def __init__(self, message: str, code: Optional[str] = None, details: Optional[Dict] = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}

class JupiterAPIError(JupiterError):
    
    def __init__(self, message: str, status_code: int, response_body: Optional[str] = None):
        super().__init__(message, code=f"HTTP_{status_code}")
        self.status_code = status_code
        self.response_body = response_body

class JupiterQuoteError(JupiterError):
    pass

class JupiterSwapError(JupiterError):
    pass

class JupiterRateLimitError(JupiterError):
    
    def __init__(self, retry_after: Optional[float] = None):
        super().__init__("Rate limit exceeded", code="RATE_LIMITED")
        self.retry_after = retry_after

class JupiterTimeoutError(JupiterError):
    pass

class JupiterConnectionError(JupiterError):
    pass

class CircuitBreakerOpenError(JupiterError):
    pass

class SwapMode(str, Enum):
    EXACT_IN = "ExactIn"
    EXACT_OUT = "ExactOut"

class DexId(str, Enum):
    RAYDIUM = "Raydium"
    RAYDIUM_CLMM = "Raydium CLMM"
    ORCA = "Orca"
    ORCA_WHIRLPOOL = "Whirlpool"
    METEORA = "Meteora"
    METEORA_DLMM = "Meteora DLMM"
    PHOENIX = "Phoenix"
    LIFINITY = "Lifinity"
    OPENBOOK = "OpenBook"

@dataclass
class TokenInfo:
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
        if self.in_amount == 0:
            return 0.0
        return self.out_amount / self.in_amount
    
    @property
    def minimum_received(self) -> int:
        return self.other_amount_threshold
    
    @property
    def num_hops(self) -> int:
        return len(self.route_plan)
    
    @property
    def dexes_used(self) -> List[str]:
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
    swap_transaction: str
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
        
        if quote.price_impact_pct > 1.0:
            warnings.append(f"High price impact: {quote.price_impact_pct:.2f}%")
        if quote.price_impact_pct > 5.0:
            warnings.append("CRITICAL: Very high price impact!")
        
        if quote.num_hops > 3:
            warnings.append(f"Complex route with {quote.num_hops} hops")
        
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

T = TypeVar('T')

@dataclass
class CacheEntry(Generic[T]):
    value: T
    expires_at: float
    
    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

class LRUCache(Generic[T]):
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 60.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[T]:
        async with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            if entry.is_expired:
                del self._cache[key]
                return None
            
            self._cache.move_to_end(key)
            return entry.value
    
    async def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        ttl = ttl if ttl is not None else self.default_ttl
        async with self._lock:
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            
            self._cache[key] = CacheEntry(
                value=value,
                expires_at=time.time() + ttl
            )
    
    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()
    
    async def cleanup_expired(self) -> int:
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
        async def cleanup_loop():
            while True:
                await asyncio.sleep(interval)
                await self.cleanup_all()
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    def stop_cleanup_task(self) -> None:
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None
    
    async def cleanup_all(self) -> Dict[str, int]:
        return {
            "token_list": await self.token_list_cache.cleanup_expired(),
            "token_info": await self.token_info_cache.cleanup_expired(),
            "price": await self.price_cache.cleanup_expired(),
            "quote": await self.quote_cache.cleanup_expired()
        }
    
    async def clear_all(self) -> None:
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
        data = f"{input_mint}:{output_mint}:{amount}:{slippage_bps}"
        return hashlib.md5(data.encode()).hexdigest()

class RateLimiter:
    
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
        start_time = time.time()
        
        while True:
            async with self._lock:
                now = time.time()
                time_passed = now - self.last_update
                self.tokens = min(
                    self.burst_size,
                    self.tokens + time_passed * self.requests_per_second
                )
                self.last_update = now
                
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True
            
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False
            
            wait_time = (1.0 - self.tokens) / self.requests_per_second
            await asyncio.sleep(min(wait_time, 0.1))
    
    async def wait(self) -> None:
        await self.acquire(timeout=None)

class AdaptiveRateLimiter:
    
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
        
        self._rate_limit_hits = 0
        self._successful_requests = 0
        self._last_adjustment = time.time()
    
    async def acquire(self, timeout: Optional[float] = None) -> bool:
        return await self._limiter.acquire(timeout)
    
    async def wait(self) -> None:
        await self._limiter.wait()
    
    async def on_success(self) -> None:
        async with self._lock:
            self._successful_requests += 1
            await self._maybe_increase_rate()
    
    async def on_rate_limit(self, retry_after: Optional[float] = None) -> None:
        async with self._lock:
            self._rate_limit_hits += 1
            new_rps = max(self.min_rps, self.current_rps * 0.5)
            await self._set_rate(new_rps)
            
            if retry_after:
                await asyncio.sleep(retry_after)
    
    async def _maybe_increase_rate(self) -> None:
        now = time.time()
        if now - self._last_adjustment < 10.0:
            return
        
        if self._rate_limit_hits == 0 and self._successful_requests > 10:
            new_rps = min(self.max_rps, self.current_rps * 1.1)
            await self._set_rate(new_rps)
        
        self._rate_limit_hits = 0
        self._successful_requests = 0
        self._last_adjustment = now
    
    async def _set_rate(self, rps: float) -> None:
        self.current_rps = rps
        self._limiter = RateLimiter(rps, int(rps * 2))

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    
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
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            
            if self._state == CircuitState.OPEN:
                if self._last_failure_time is None:
                    return False
                
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info("Circuit breaker: OPEN -> HALF_OPEN")
                    return True
                return False
            
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False
    
    async def record_success(self) -> None:
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
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0

class JupiterClient:
    
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
        self.api_base = api_base.rstrip("/")
        self.price_api_base = price_api_base.rstrip("/")
        self.token_api_base = token_api_base.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.referral_account = referral_account
        self.referral_fee_bps = referral_fee_bps
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        
        self._rate_limiter = AdaptiveRateLimiter(rate_limit_rps)
        self._circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None
        self._cache = JupiterCache() if enable_cache else None
        
        self.metrics = JupiterMetrics()
        
        self._closed = False
    
    async def __aenter__(self) -> "JupiterClient":
        await self._ensure_session()
        if self._cache:
            self._cache.start_cleanup_task()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
    
    async def _ensure_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._session is None or self._session.closed:
                connector = aiohttp.TCPConnector(
                    limit=100,
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
        if self._closed:
            return
        
        self._closed = True
        
        if self._cache:
            self._cache.stop_cleanup_task()
        
        if self._session and not self._session.closed:
            await self._session.close()
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
        session = await self._ensure_session()
        
        if self._circuit_breaker and not await self._circuit_breaker.can_execute():
            self.metrics.circuit_breaker_trips += 1
            raise CircuitBreakerOpenError("Circuit breaker is open")
        
        last_error: Optional[Exception] = None
        
        for attempt in range(self.max_retries):
            await self._rate_limiter.wait()
            
            start_time = time.time()
            self.metrics.total_requests += 1
            
            try:
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
                    
                    if response.status == 429:
                        self.metrics.rate_limited_requests += 1
                        retry_after = float(response.headers.get("Retry-After", self.retry_delay * (2 ** attempt)))
                        
                        await self._rate_limiter.on_rate_limit(retry_after)
                        
                        if retry_on_rate_limit and attempt < self.max_retries - 1:
                            logger.warning(f"Rate limited, retrying after {retry_after}s")
                            continue
                        raise JupiterRateLimitError(retry_after)
                    
                    try:
                        data = await response.json()
                    except Exception:
                        data = {"raw": await response.text()}
                    
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
        if use_cache and self._cache:
            cache_key = JupiterCache.make_quote_key(input_mint, output_mint, amount, slippage_bps)
            cached = await self._cache.quote_cache.get(cache_key)
            if cached:
                self.metrics.cache_hits += 1
                return cached
            self.metrics.cache_misses += 1
        
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
        quotes = []
        
        try:
            main_quote = await self.get_quote(
                input_mint, output_mint, amount, slippage_bps, **kwargs
            )
            quotes.append(main_quote)
        except JupiterQuoteError:
            pass
        
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
        
        quotes.sort(key=lambda q: q.out_amount, reverse=True)
        return quotes[:count]
    
    def select_best_quote(self, quotes: List[SwapQuote]) -> Optional[SwapQuote]:
        if not quotes:
            return None
        
        def score(q: SwapQuote) -> float:
            output_score = q.out_amount
            impact_penalty = q.price_impact_pct * 0.01 * q.out_amount
            hop_penalty = (q.num_hops - 1) * 0.001 * q.out_amount
            return output_score - impact_penalty - hop_penalty
        
        return max(quotes, key=score)

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
                    "maxLamports": 10000000,
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
        wallet_keypair: Any,
        rpc_client: Any,
        priority_fee_lamports: Optional[int] = None,
        skip_preflight: bool = False,
        max_retries: int = 3
    ) -> SwapResult:
        self.metrics.swaps_executed += 1
        
        try:
            from solders.transaction import VersionedTransaction
            from solders.signature import Signature
            from solana.rpc.commitment import Confirmed
            
            user_pubkey = str(wallet_keypair.pubkey())
            
            swap_tx = await self.get_swap_transaction(
                quote=quote,
                user_pubkey=user_pubkey,
                compute_unit_price_micro_lamports=priority_fee_lamports
            )
            
            tx_bytes = base64.b64decode(swap_tx.swap_transaction)
            transaction = VersionedTransaction.from_bytes(tx_bytes)
            
            signed_tx = VersionedTransaction(
                transaction.message,
                [wallet_keypair]
            )
            
            for attempt in range(max_retries):
                try:
                    result = await rpc_client.send_transaction(
                        signed_tx,
                        opts={
                            "skip_preflight": skip_preflight,
                            "preflight_commitment": Confirmed,
                            "max_retries": 0
                        }
                    )
                    
                    signature = str(result.value)
                    
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

    async def get_token_list(self, use_cache: bool = True) -> List[TokenInfo]:
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
            for token in tokens:
                await self._cache.token_info_cache.set(token.address, token)
        
        return tokens
    
    async def get_token_info(self, mint: str, use_cache: bool = True) -> Optional[TokenInfo]:
        if use_cache and self._cache:
            cached = await self._cache.token_info_cache.get(mint)
            if cached:
                self.metrics.cache_hits += 1
                return cached
            self.metrics.cache_misses += 1
        
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
        
        tokens = await self.get_token_list(use_cache=use_cache)
        for token in tokens:
            if token.address == mint:
                return token
        
        return None
    
    async def search_tokens(self, query: str, limit: int = 10) -> List[TokenInfo]:
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

    async def analyze_route(self, quote: SwapQuote) -> RouteAnalysis:
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
        quotes = await self.get_quotes(
            input_mint, output_mint, amount, slippage_bps, count=5
        )
        
        analyses = []
        for quote in quotes:
            analysis = await self.analyze_route(quote)
            analyses.append(analysis)
        
        analyses.sort(key=lambda a: a.quote.out_amount, reverse=True)
        return analyses
    
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.to_dict()
    
    def reset_metrics(self) -> None:
        self.metrics = JupiterMetrics()
    
    async def health_check(self) -> bool:
        try:
            await self.get_quote(
                SOL_MINT,
                USDC_MINT,
                1_000_000,
                slippage_bps=100,
                use_cache=False
            )
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

async def quick_quote(
    input_mint: str,
    output_mint: str,
    amount: int,
    slippage_bps: int = DEFAULT_SLIPPAGE_BPS
) -> SwapQuote:
    async with JupiterClient() as client:
        return await client.get_quote(input_mint, output_mint, amount, slippage_bps)

async def get_sol_to_usdc_price() -> float:
    async with JupiterClient() as client:
        quote = await client.get_quote(
            SOL_MINT,
            USDC_MINT,
            1_000_000_000,
            slippage_bps=10
        )
        return quote.out_amount / 1_000_000

__all__ = [
    "JupiterClient",
    
    "TokenInfo",
    "TokenPrice",
    "SwapQuote",
    "SwapTransaction",
    "SwapResult",
    "RoutePlan",
    "RouteAnalysis",
    "JupiterMetrics",
    
    "SwapMode",
    "DexId",
    
    "JupiterError",
    "JupiterAPIError",
    "JupiterQuoteError",
    "JupiterSwapError",
    "JupiterRateLimitError",
    "JupiterTimeoutError",
    "JupiterConnectionError",
    "CircuitBreakerOpenError",
    
    "SOL_MINT",
    "USDC_MINT",
    "USDT_MINT",
    "DEFAULT_SLIPPAGE_BPS",
    
    "quick_quote",
    "get_sol_to_usdc_price",
]
