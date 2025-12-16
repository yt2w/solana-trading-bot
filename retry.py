"""
Retry Module - Production-grade retry logic for Solana Trading Bot.

Provides comprehensive retry mechanisms with multiple strategies,
decorators, context managers, and pre-configured policies for
different operation types.
"""

import asyncio
import functools
import logging
import random
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class RetryError(Exception):
    """Base exception for retry-related errors."""

    def __init__(
        self,
        message: str,
        last_exception: Optional[Exception] = None,
        attempts: int = 0,
        total_time: float = 0.0,
    ):
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts = attempts
        self.total_time = total_time


class RetryExhaustedError(RetryError):
    """Raised when all retry attempts are exhausted."""
    pass


class RetryBudgetExceededError(RetryError):
    """Raised when retry budget is exceeded."""
    pass


class NonRetryableError(RetryError):
    """Raised when a non-retryable exception is encountered."""
    pass


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts (0 = no retries)
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay cap in seconds
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delays
        jitter_factor: Maximum jitter as fraction of delay (0.0-1.0)
        retryable_exceptions: Tuple of exceptions to retry on
        non_retryable_exceptions: Tuple of exceptions to never retry on
        retry_on_result: Optional function to check if result should trigger retry
        timeout: Optional total timeout for all attempts combined
        retry_error_codes: Set of error codes that should trigger retry
        non_retry_error_codes: Set of error codes that should NOT retry
    """
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.5
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    non_retryable_exceptions: Tuple[Type[Exception], ...] = (
        KeyboardInterrupt,
        SystemExit,
        GeneratorExit,
    )
    retry_on_result: Optional[Callable[[Any], bool]] = None
    timeout: Optional[float] = None
    retry_error_codes: Set[int] = field(default_factory=set)
    non_retry_error_codes: Set[int] = field(default_factory=set)

    def __post_init__(self):
        """Validate configuration values."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.base_delay < 0:
            raise ValueError("base_delay must be >= 0")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if not 0 <= self.jitter_factor <= 1:
            raise ValueError("jitter_factor must be between 0 and 1")


class BackoffStrategy(ABC):
    """Abstract base class for backoff strategies."""

    @abstractmethod
    def get_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay for a given attempt number."""
        pass

    def _apply_jitter(self, delay: float, config: RetryConfig) -> float:
        """Apply jitter to delay if configured."""
        if config.jitter and config.jitter_factor > 0:
            jitter_range = delay * config.jitter_factor
            delay += random.uniform(-jitter_range, jitter_range)
        return max(0, min(delay, config.max_delay))


class ExponentialBackoff(BackoffStrategy):
    """Exponential backoff: delay = base_delay * (exponential_base ^ attempt)"""

    def get_delay(self, attempt: int, config: RetryConfig) -> float:
        delay = config.base_delay * (config.exponential_base ** (attempt - 1))
        return self._apply_jitter(min(delay, config.max_delay), config)


class LinearBackoff(BackoffStrategy):
    """Linear backoff: delay = base_delay * attempt"""

    def get_delay(self, attempt: int, config: RetryConfig) -> float:
        delay = config.base_delay * attempt
        return self._apply_jitter(min(delay, config.max_delay), config)


class ConstantBackoff(BackoffStrategy):
    """Constant backoff: delay = base_delay (always)"""

    def get_delay(self, attempt: int, config: RetryConfig) -> float:
        return self._apply_jitter(config.base_delay, config)


class FibonacciBackoff(BackoffStrategy):
    """Fibonacci sequence backoff: delay = base_delay * fibonacci(attempt)"""

    def __init__(self):
        self._cache: Dict[int, int] = {0: 0, 1: 1}

    def _fibonacci(self, n: int) -> int:
        if n not in self._cache:
            self._cache[n] = self._fibonacci(n - 1) + self._fibonacci(n - 2)
        return self._cache[n]

    def get_delay(self, attempt: int, config: RetryConfig) -> float:
        fib = self._fibonacci(attempt)
        delay = config.base_delay * fib
        return self._apply_jitter(min(delay, config.max_delay), config)


class DecorrelatedJitter(BackoffStrategy):
    """AWS-style decorrelated jitter backoff."""

    def __init__(self):
        self._previous_delay: Optional[float] = None

    def get_delay(self, attempt: int, config: RetryConfig) -> float:
        if attempt == 1 or self._previous_delay is None:
            self._previous_delay = config.base_delay
            return config.base_delay

        delay = random.uniform(config.base_delay, self._previous_delay * 3)
        delay = min(delay, config.max_delay)
        self._previous_delay = delay
        return delay

    def reset(self):
        self._previous_delay = None


BACKOFF_STRATEGIES: Dict[str, Type[BackoffStrategy]] = {
    "exponential": ExponentialBackoff,
    "linear": LinearBackoff,
    "constant": ConstantBackoff,
    "fibonacci": FibonacciBackoff,
    "decorrelated": DecorrelatedJitter,
}


@dataclass
class RetryStatistics:
    """Statistics tracking for retry operations."""

    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_retries: int = 0
    total_delay_time: float = 0.0
    exceptions_by_type: Dict[str, int] = field(default_factory=dict)
    last_exception: Optional[Exception] = None
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None

    def record_attempt(self, success: bool, retries: int, delay_time: float):
        """Record an attempt outcome."""
        self.total_attempts += 1
        self.total_retries += retries
        self.total_delay_time += delay_time

        if success:
            self.successful_attempts += 1
            self.last_success_time = time.time()
        else:
            self.failed_attempts += 1
            self.last_failure_time = time.time()

    def record_exception(self, exc: Exception):
        """Record an exception occurrence."""
        exc_type = type(exc).__name__
        self.exceptions_by_type[exc_type] = self.exceptions_by_type.get(exc_type, 0) + 1
        self.last_exception = exc

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_attempts == 0:
            return 0.0
        return (self.successful_attempts / self.total_attempts) * 100

    @property
    def average_retries(self) -> float:
        """Calculate average retries per attempt."""
        if self.total_attempts == 0:
            return 0.0
        return self.total_retries / self.total_attempts

    def reset(self):
        """Reset all statistics."""
        self.total_attempts = 0
        self.successful_attempts = 0
        self.failed_attempts = 0
        self.total_retries = 0
        self.total_delay_time = 0.0
        self.exceptions_by_type.clear()
        self.last_exception = None
        self.last_success_time = None
        self.last_failure_time = None


class RetryBudget:
    """
    Manages retry budget to prevent retry storms.
    Limits the number of retries allowed within a time window.
    """

    def __init__(
        self,
        max_retries_per_window: int = 100,
        window_seconds: float = 60.0,
        min_retries_per_second: float = 1.0,
    ):
        self.max_retries_per_window = max_retries_per_window
        self.window_seconds = window_seconds
        self.min_retries_per_second = min_retries_per_second
        self._retry_times: Deque[float] = deque()
        self._lock = asyncio.Lock()

    def _cleanup_old_entries(self):
        """Remove entries outside the time window."""
        cutoff = time.time() - self.window_seconds
        while self._retry_times and self._retry_times[0] < cutoff:
            self._retry_times.popleft()

    def can_retry(self) -> bool:
        """Check if a retry is allowed within budget."""
        self._cleanup_old_entries()
        if len(self._retry_times) < self.min_retries_per_second * self.window_seconds:
            return True
        return len(self._retry_times) < self.max_retries_per_window

    def record_retry(self):
        """Record a retry attempt."""
        self._cleanup_old_entries()
        self._retry_times.append(time.time())

    async def acquire(self) -> bool:
        """Async acquire a retry permit."""
        async with self._lock:
            if self.can_retry():
                self.record_retry()
                return True
            return False

    @property
    def current_usage(self) -> int:
        """Get current retry count in window."""
        self._cleanup_old_entries()
        return len(self._retry_times)

    @property
    def remaining_budget(self) -> int:
        """Get remaining retries in budget."""
        return max(0, self.max_retries_per_window - self.current_usage)


def should_retry(
    exception: Exception,
    config: RetryConfig,
    attempt: int,
) -> bool:
    """Determine if an exception should trigger a retry."""
    if attempt >= config.max_retries:
        return False

    if isinstance(exception, config.non_retryable_exceptions):
        return False

    error_code = getattr(exception, "code", None) or getattr(exception, "error_code", None)
    if error_code is not None:
        if error_code in config.non_retry_error_codes:
            return False
        if config.retry_error_codes and error_code not in config.retry_error_codes:
            return False

    if isinstance(exception, config.retryable_exceptions):
        return True

    return False


def calculate_delay(
    attempt: int,
    config: RetryConfig,
    strategy: Optional[BackoffStrategy] = None,
) -> float:
    """Calculate delay for a retry attempt."""
    if strategy is None:
        strategy = ExponentialBackoff()
    return strategy.get_delay(attempt, config)


def get_exception_info(exc: Exception) -> Dict[str, Any]:
    """Extract useful information from an exception."""
    info = {
        "type": type(exc).__name__,
        "message": str(exc),
        "args": exc.args,
    }
    for attr in ("code", "error_code", "status_code", "errno"):
        if hasattr(exc, attr):
            info[attr] = getattr(exc, attr)
    return info


class RetryContext(Generic[T]):
    """
    Context manager for retry blocks with callbacks.

    Usage:
        async with RetryContext(config) as ctx:
            result = await ctx.execute(my_async_function, arg1, arg2)
    """

    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        strategy: Optional[BackoffStrategy] = None,
        on_retry: Optional[Callable[[int, Exception, float], None]] = None,
        on_success: Optional[Callable[[T, int], None]] = None,
        on_failure: Optional[Callable[[Exception, int], None]] = None,
        budget: Optional[RetryBudget] = None,
        statistics: Optional[RetryStatistics] = None,
    ):
        self.config = config or RetryConfig()
        self.strategy = strategy or ExponentialBackoff()
        self.on_retry = on_retry
        self.on_success = on_success
        self.on_failure = on_failure
        self.budget = budget
        self.statistics = statistics or RetryStatistics()

        self._attempt = 0
        self._start_time: Optional[float] = None
        self._last_exception: Optional[Exception] = None
        self._total_delay = 0.0

    async def __aenter__(self) -> "RetryContext[T]":
        self._start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def __enter__(self) -> "RetryContext[T]":
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs,
    ) -> T:
        """Execute an async function with retry logic."""
        self._attempt = 0
        self._total_delay = 0.0

        while True:
            self._attempt += 1

            if self.config.timeout and self._start_time:
                elapsed = time.time() - self._start_time
                if elapsed >= self.config.timeout:
                    self._handle_failure()
                    raise RetryExhaustedError(
                        f"Timeout exceeded after {elapsed:.2f}s",
                        self._last_exception,
                        self._attempt,
                        elapsed,
                    )

            try:
                result = await func(*args, **kwargs)

                if self.config.retry_on_result and self.config.retry_on_result(result):
                    if self._attempt <= self.config.max_retries:
                        delay = self._wait_before_retry(None)
                        await asyncio.sleep(delay)
                        continue

                self._handle_success(result)
                return result

            except Exception as e:
                self._last_exception = e
                self.statistics.record_exception(e)

                if not should_retry(e, self.config, self._attempt):
                    self._handle_failure()
                    if isinstance(e, self.config.non_retryable_exceptions):
                        raise NonRetryableError(
                            f"Non-retryable exception: {e}",
                            e,
                            self._attempt,
                            self._total_delay,
                        ) from e
                    raise RetryExhaustedError(
                        f"Exhausted {self._attempt} attempts",
                        e,
                        self._attempt,
                        self._total_delay,
                    ) from e

                if self.budget and not await self.budget.acquire():
                    self._handle_failure()
                    raise RetryBudgetExceededError(
                        "Retry budget exceeded",
                        e,
                        self._attempt,
                        self._total_delay,
                    ) from e

                delay = self._wait_before_retry(e)
                await asyncio.sleep(delay)

    def execute_sync(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """Execute a sync function with retry logic."""
        self._attempt = 0
        self._total_delay = 0.0

        while True:
            self._attempt += 1

            if self.config.timeout and self._start_time:
                elapsed = time.time() - self._start_time
                if elapsed >= self.config.timeout:
                    self._handle_failure()
                    raise RetryExhaustedError(
                        f"Timeout exceeded after {elapsed:.2f}s",
                        self._last_exception,
                        self._attempt,
                        elapsed,
                    )

            try:
                result = func(*args, **kwargs)

                if self.config.retry_on_result and self.config.retry_on_result(result):
                    if self._attempt <= self.config.max_retries:
                        delay = self._wait_before_retry(None)
                        time.sleep(delay)
                        continue

                self._handle_success(result)
                return result

            except Exception as e:
                self._last_exception = e
                self.statistics.record_exception(e)

                if not should_retry(e, self.config, self._attempt):
                    self._handle_failure()
                    raise RetryExhaustedError(
                        f"Exhausted {self._attempt} attempts",
                        e,
                        self._attempt,
                        self._total_delay,
                    ) from e

                delay = self._wait_before_retry(e)
                time.sleep(delay)

    def _wait_before_retry(self, exception: Optional[Exception]) -> float:
        """Calculate and log delay before retry."""
        delay = self.strategy.get_delay(self._attempt, self.config)
        self._total_delay += delay

        logger.warning(
            "Retry attempt %d/%d after %.2fs delay. Exception: %s",
            self._attempt,
            self.config.max_retries,
            delay,
            exception,
        )

        if self.on_retry:
            self.on_retry(self._attempt, exception, delay)

        return delay

    def _handle_success(self, result: T):
        """Handle successful execution."""
        self.statistics.record_attempt(True, self._attempt - 1, self._total_delay)

        if self._attempt > 1:
            logger.info(
                "Operation succeeded after %d attempts (%.2fs total delay)",
                self._attempt,
                self._total_delay,
            )

        if self.on_success:
            self.on_success(result, self._attempt)

    def _handle_failure(self):
        """Handle final failure."""
        self.statistics.record_attempt(False, self._attempt - 1, self._total_delay)

        logger.error(
            "Operation failed after %d attempts (%.2fs total delay). Last error: %s",
            self._attempt,
            self._total_delay,
            self._last_exception,
        )

        if self.on_failure and self._last_exception:
            self.on_failure(self._last_exception, self._attempt)

    @property
    def attempts(self) -> int:
        return self._attempt

    @property
    def last_exception(self) -> Optional[Exception]:
        return self._last_exception

    @property
    def total_delay(self) -> float:
        return self._total_delay


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    strategy: Optional[BackoffStrategy] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> Callable[[F], F]:
    """
    Retry decorator for synchronous functions.

    Example:
        @retry(max_retries=3, base_delay=1.0)
        def fetch_data():
            ...
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retryable_exceptions=exceptions,
    )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ctx = RetryContext(
                config=config,
                strategy=strategy or ExponentialBackoff(),
                on_retry=on_retry,
            )
            with ctx:
                return ctx.execute_sync(func, *args, **kwargs)

        return wrapper

    return decorator


def async_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    strategy: Optional[BackoffStrategy] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> Callable[[F], F]:
    """
    Retry decorator for asynchronous functions.

    Example:
        @async_retry(max_retries=3, base_delay=1.0)
        async def fetch_data():
            ...
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retryable_exceptions=exceptions,
    )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            ctx = RetryContext(
                config=config,
                strategy=strategy or ExponentialBackoff(),
                on_retry=on_retry,
            )
            async with ctx:
                return await ctx.execute(func, *args, **kwargs)

        return wrapper

    return decorator


def retry_on_exception(
    *exceptions: Type[Exception],
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Callable[[F], F]:
    """
    Retry decorator that only retries on specific exceptions.

    Example:
        @retry_on_exception(ConnectionError, TimeoutError, max_retries=5)
        async def connect():
            ...
    """
    if not exceptions:
        exceptions = (Exception,)

    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        retryable_exceptions=exceptions,
    )

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                ctx = RetryContext(config=config)
                async with ctx:
                    return await ctx.execute(func, *args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                ctx = RetryContext(config=config)
                with ctx:
                    return ctx.execute_sync(func, *args, **kwargs)
            return sync_wrapper

    return decorator


def retry_with_fallback(
    fallback_fn: Callable[..., T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Retry decorator that calls a fallback function on final failure.

    Example:
        def get_cached_price():
            return cached_prices.get("SOL")

        @retry_with_fallback(get_cached_price, max_retries=3)
        async def get_live_price():
            ...
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        retryable_exceptions=exceptions,
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                ctx = RetryContext(config=config)
                try:
                    async with ctx:
                        return await ctx.execute(func, *args, **kwargs)
                except RetryError as e:
                    logger.warning(
                        "All retries exhausted, calling fallback. Error: %s",
                        e.last_exception,
                    )
                    if asyncio.iscoroutinefunction(fallback_fn):
                        return await fallback_fn(*args, **kwargs)
                    return fallback_fn(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                ctx = RetryContext(config=config)
                try:
                    with ctx:
                        return ctx.execute_sync(func, *args, **kwargs)
                except RetryError as e:
                    logger.warning(
                        "All retries exhausted, calling fallback. Error: %s",
                        e.last_exception,
                    )
                    return fallback_fn(*args, **kwargs)
            return sync_wrapper

    return decorator


def with_retry(
    func: Callable[..., T],
    config: Optional[RetryConfig] = None,
    strategy: Optional[BackoffStrategy] = None,
) -> Callable[..., T]:
    """
    Wrap a function with retry logic.

    Example:
        risky_operation = with_retry(my_function, config=RPC_RETRY_POLICY)
        result = await risky_operation(arg1, arg2)
    """
    config = config or RetryConfig()
    strategy = strategy or ExponentialBackoff()

    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapped(*args, **kwargs) -> T:
            ctx = RetryContext(config=config, strategy=strategy)
            async with ctx:
                return await ctx.execute(func, *args, **kwargs)
        return async_wrapped
    else:
        @functools.wraps(func)
        def sync_wrapped(*args, **kwargs) -> T:
            ctx = RetryContext(config=config, strategy=strategy)
            with ctx:
                return ctx.execute_sync(func, *args, **kwargs)
        return sync_wrapped


RPC_RETRY_POLICY = RetryConfig(
    max_retries=5,
    base_delay=0.5,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True,
    jitter_factor=0.3,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
        OSError,
        Exception,
    ),
    retry_error_codes={
        429,
        500,
        502,
        503,
        504,
        -32005,
        -32000,
    },
    non_retry_error_codes={
        -32600,
        -32601,
        -32602,
    },
)


HTTP_RETRY_POLICY = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=15.0,
    exponential_base=2.0,
    jitter=True,
    jitter_factor=0.5,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
        OSError,
    ),
    retry_error_codes={429, 500, 502, 503, 504},
)


TRANSACTION_RETRY_POLICY = RetryConfig(
    max_retries=3,
    base_delay=2.0,
    max_delay=20.0,
    exponential_base=2.0,
    jitter=True,
    jitter_factor=0.2,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
    ),
    non_retryable_exceptions=(
        KeyboardInterrupt,
        SystemExit,
    ),
)


JUPITER_RETRY_POLICY = RetryConfig(
    max_retries=4,
    base_delay=0.3,
    max_delay=10.0,
    exponential_base=2.0,
    jitter=True,
    jitter_factor=0.4,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
        OSError,
    ),
    retry_error_codes={429, 500, 502, 503, 504},
    timeout=30.0,
)


CACHE_RETRY_POLICY = RetryConfig(
    max_retries=2,
    base_delay=0.1,
    max_delay=1.0,
    exponential_base=2.0,
    jitter=True,
    jitter_factor=0.5,
    retryable_exceptions=(ConnectionError, TimeoutError),
)


class RetryBuilder:
    """
    Fluent builder for creating retry configurations.

    Example:
        config = (
            RetryBuilder()
            .with_max_retries(5)
            .with_exponential_backoff(base=1.0, max=30.0)
            .with_jitter(factor=0.3)
            .retry_on(ConnectionError, TimeoutError)
            .build()
        )
    """

    def __init__(self):
        self._max_retries = 3
        self._base_delay = 1.0
        self._max_delay = 60.0
        self._exponential_base = 2.0
        self._jitter = True
        self._jitter_factor = 0.5
        self._retryable_exceptions: List[Type[Exception]] = [Exception]
        self._non_retryable_exceptions: List[Type[Exception]] = [
            KeyboardInterrupt, SystemExit, GeneratorExit
        ]
        self._retry_error_codes: Set[int] = set()
        self._non_retry_error_codes: Set[int] = set()
        self._timeout: Optional[float] = None
        self._retry_on_result: Optional[Callable[[Any], bool]] = None

    def with_max_retries(self, n: int) -> "RetryBuilder":
        """Set maximum retry attempts."""
        self._max_retries = n
        return self

    def with_exponential_backoff(
        self,
        base: float = 1.0,
        max: float = 60.0,
        exponential_base: float = 2.0,
    ) -> "RetryBuilder":
        """Configure exponential backoff."""
        self._base_delay = base
        self._max_delay = max
        self._exponential_base = exponential_base
        return self

    def with_linear_backoff(self, base: float = 1.0, max: float = 60.0) -> "RetryBuilder":
        """Configure linear backoff."""
        self._base_delay = base
        self._max_delay = max
        self._exponential_base = 1.0
        return self

    def with_constant_delay(self, delay: float) -> "RetryBuilder":
        """Configure constant delay."""
        self._base_delay = delay
        self._max_delay = delay
        return self

    def with_jitter(self, enabled: bool = True, factor: float = 0.5) -> "RetryBuilder":
        """Configure jitter."""
        self._jitter = enabled
        self._jitter_factor = factor
        return self

    def retry_on(self, *exceptions: Type[Exception]) -> "RetryBuilder":
        """Set exceptions to retry on."""
        self._retryable_exceptions = list(exceptions)
        return self

    def never_retry_on(self, *exceptions: Type[Exception]) -> "RetryBuilder":
        """Set exceptions to never retry on."""
        self._non_retryable_exceptions = list(exceptions)
        return self

    def retry_on_codes(self, *codes: int) -> "RetryBuilder":
        """Set error codes to retry on."""
        self._retry_error_codes = set(codes)
        return self

    def never_retry_on_codes(self, *codes: int) -> "RetryBuilder":
        """Set error codes to never retry on."""
        self._non_retry_error_codes = set(codes)
        return self

    def with_timeout(self, seconds: float) -> "RetryBuilder":
        """Set total timeout for all attempts."""
        self._timeout = seconds
        return self

    def retry_if_result(self, predicate: Callable[[Any], bool]) -> "RetryBuilder":
        """Set predicate to check if result should trigger retry."""
        self._retry_on_result = predicate
        return self

    def build(self) -> RetryConfig:
        """Build the retry configuration."""
        return RetryConfig(
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            max_delay=self._max_delay,
            exponential_base=self._exponential_base,
            jitter=self._jitter,
            jitter_factor=self._jitter_factor,
            retryable_exceptions=tuple(self._retryable_exceptions),
            non_retryable_exceptions=tuple(self._non_retryable_exceptions),
            retry_error_codes=self._retry_error_codes,
            non_retry_error_codes=self._non_retry_error_codes,
            timeout=self._timeout,
            retry_on_result=self._retry_on_result,
        )


async def retry_async_operation(
    operation: Callable[..., Awaitable[T]],
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs,
) -> T:
    """Execute an async operation with retry."""
    ctx = RetryContext(config=config or RetryConfig())
    async with ctx:
        return await ctx.execute(operation, *args, **kwargs)


def retry_sync_operation(
    operation: Callable[..., T],
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs,
) -> T:
    """Execute a sync operation with retry."""
    ctx = RetryContext(config=config or RetryConfig())
    with ctx:
        return ctx.execute_sync(operation, *args, **kwargs)


__all__ = [

    "RetryError",
    "RetryExhaustedError",
    "RetryBudgetExceededError",
    "NonRetryableError",

    "RetryConfig",

    "BackoffStrategy",
    "ExponentialBackoff",
    "LinearBackoff",
    "ConstantBackoff",
    "FibonacciBackoff",
    "DecorrelatedJitter",
    "BACKOFF_STRATEGIES",

    "RetryStatistics",
    "RetryBudget",

    "RetryContext",

    "retry",
    "async_retry",
    "retry_on_exception",
    "retry_with_fallback",

    "should_retry",
    "calculate_delay",
    "get_exception_info",
    "with_retry",
    "retry_async_operation",
    "retry_sync_operation",

    "RPC_RETRY_POLICY",
    "HTTP_RETRY_POLICY",
    "TRANSACTION_RETRY_POLICY",
    "JUPITER_RETRY_POLICY",
    "CACHE_RETRY_POLICY",

    "RetryBuilder",
]
