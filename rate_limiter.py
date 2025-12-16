
import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])

class RateLimitTier(Enum):
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    UNLIMITED = "unlimited"

class OperationType(Enum):
    WALLET_CREATE = "wallet_create"
    WALLET_EXPORT = "wallet_export"
    WALLET_IMPORT = "wallet_import"
    WALLET_BALANCE = "wallet_balance"

    TRADE_BUY = "trade_buy"
    TRADE_SELL = "trade_sell"
    TRADE_SWAP = "trade_swap"

    QUERY_PRICE = "query_price"
    QUERY_PORTFOLIO = "query_portfolio"
    QUERY_HISTORY = "query_history"

    API_CALL = "api_call"
    API_BATCH = "api_batch"

    ADMIN_ACTION = "admin_action"

    GENERIC = "generic"

class ViolationType(Enum):
    SOFT = "soft"
    HARD = "hard"
    ABUSE = "abuse"

@dataclass
class RateLimitConfig:

    window_size_seconds: int = 60
    sliding_window_precision: int = 10

    default_limit: int = 60
    burst_limit: int = 10
    burst_window_seconds: int = 5

    violation_threshold: int = 5
    violation_window_seconds: int = 300
    abuse_cooldown_seconds: int = 600
    abuse_multiplier: float = 2.0
    max_abuse_cooldown_seconds: int = 3600

    persistence_enabled: bool = True
    persistence_path: str = "data/rate_limits"
    persistence_interval_seconds: int = 60

    cleanup_interval_seconds: int = 300
    entry_ttl_seconds: int = 3600

    adaptive_enabled: bool = True
    adaptive_increase_threshold: float = 0.9
    adaptive_decrease_threshold: float = 0.5
    adaptive_adjustment_factor: float = 0.1

    @classmethod
    def from_env(cls) -> 'RateLimitConfig':
        return cls(
            window_size_seconds=int(os.getenv('RATE_LIMIT_WINDOW', '60')),
            default_limit=int(os.getenv('RATE_LIMIT_DEFAULT', '60')),
            burst_limit=int(os.getenv('RATE_LIMIT_BURST', '10')),
            violation_threshold=int(os.getenv('RATE_LIMIT_VIOLATION_THRESHOLD', '5')),
            abuse_cooldown_seconds=int(os.getenv('RATE_LIMIT_ABUSE_COOLDOWN', '600')),
            persistence_enabled=os.getenv('RATE_LIMIT_PERSISTENCE', 'true').lower() == 'true',
            persistence_path=os.getenv('RATE_LIMIT_PERSISTENCE_PATH', 'data/rate_limits'),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RateLimitConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

@dataclass
class TierLimits:
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int
    concurrent_limit: int

    operation_limits: Dict[str, int] = field(default_factory=dict)

DEFAULT_TIER_LIMITS: Dict[RateLimitTier, TierLimits] = {
    RateLimitTier.FREE: TierLimits(
        requests_per_minute=10,
        requests_per_hour=100,
        requests_per_day=500,
        burst_limit=3,
        concurrent_limit=2,
        operation_limits={
            OperationType.WALLET_CREATE.value: 2,
            OperationType.WALLET_EXPORT.value: 5,
            OperationType.TRADE_BUY.value: 20,
            OperationType.TRADE_SELL.value: 20,
        }
    ),
    RateLimitTier.BASIC: TierLimits(
        requests_per_minute=30,
        requests_per_hour=500,
        requests_per_day=5000,
        burst_limit=10,
        concurrent_limit=5,
        operation_limits={
            OperationType.WALLET_CREATE.value: 10,
            OperationType.WALLET_EXPORT.value: 20,
            OperationType.TRADE_BUY.value: 100,
            OperationType.TRADE_SELL.value: 100,
        }
    ),
    RateLimitTier.PREMIUM: TierLimits(
        requests_per_minute=100,
        requests_per_hour=2000,
        requests_per_day=20000,
        burst_limit=30,
        concurrent_limit=15,
        operation_limits={
            OperationType.WALLET_CREATE.value: 50,
            OperationType.WALLET_EXPORT.value: 100,
            OperationType.TRADE_BUY.value: 500,
            OperationType.TRADE_SELL.value: 500,
        }
    ),
    RateLimitTier.ENTERPRISE: TierLimits(
        requests_per_minute=500,
        requests_per_hour=10000,
        requests_per_day=100000,
        burst_limit=100,
        concurrent_limit=50,
        operation_limits={}
    ),
    RateLimitTier.UNLIMITED: TierLimits(
        requests_per_minute=999999,
        requests_per_hour=999999,
        requests_per_day=999999,
        burst_limit=999999,
        concurrent_limit=999999,
        operation_limits={}
    ),
}

@dataclass
class RateLimitResult:
    allowed: bool
    remaining: int
    limit: int
    reset_at: float
    retry_after: Optional[float] = None
    violation_type: Optional[ViolationType] = None
    message: str = ""

    @property
    def reset_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.reset_at)

    @property
    def headers(self) -> Dict[str, str]:
        headers = {
            'X-RateLimit-Limit': str(self.limit),
            'X-RateLimit-Remaining': str(max(0, self.remaining)),
            'X-RateLimit-Reset': str(int(self.reset_at)),
        }
        if self.retry_after is not None:
            headers['Retry-After'] = str(int(self.retry_after))
        return headers

    def to_dict(self) -> Dict[str, Any]:
        return {
            'allowed': self.allowed,
            'remaining': self.remaining,
            'limit': self.limit,
            'reset_at': self.reset_at,
            'retry_after': self.retry_after,
            'violation_type': self.violation_type.value if self.violation_type else None,
            'message': self.message,
        }

@dataclass
class UsageStats:
    total_requests: int = 0
    allowed_requests: int = 0
    rejected_requests: int = 0
    violations: int = 0
    abuse_flags: int = 0
    peak_usage: float = 0.0
    avg_usage: float = 0.0
    last_request_at: Optional[float] = None

    def record_request(self, allowed: bool, utilization: float) -> None:
        self.total_requests += 1
        if allowed:
            self.allowed_requests += 1
        else:
            self.rejected_requests += 1
            self.violations += 1
        self.peak_usage = max(self.peak_usage, utilization)
        if self.total_requests > 1:
            self.avg_usage = (self.avg_usage * (self.total_requests - 1) + utilization) / self.total_requests
        else:
            self.avg_usage = utilization
        self.last_request_at = time.time()

class BaseRateLimiter(ABC):

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._lock = asyncio.Lock()
        self._stats = UsageStats()

    @abstractmethod
    async def acquire(self, key: str, cost: int = 1) -> RateLimitResult:
        pass

    @abstractmethod
    async def get_remaining(self, key: str) -> int:
        pass

    @abstractmethod
    async def reset(self, key: str) -> None:
        pass

    @abstractmethod
    async def get_state(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def load_state(self, state: Dict[str, Any]) -> None:
        pass

    @property
    def stats(self) -> UsageStats:
        return self._stats

class SlidingWindowRateLimiter(BaseRateLimiter):

    def __init__(
        self,
        limit: int,
        window_seconds: int = 60,
        precision: int = 10,
        config: Optional[RateLimitConfig] = None
    ):
        super().__init__(config)
        self.limit = limit
        self.window_seconds = window_seconds
        self.precision = precision
        self.sub_window_seconds = window_seconds / precision

        self._buckets: Dict[str, List[Tuple[float, int]]] = defaultdict(list)

    def _get_current_window(self) -> float:
        now = time.time()
        return now - (now % self.sub_window_seconds)

    def _cleanup_old_entries(self, key: str, now: float) -> None:
        cutoff = now - self.window_seconds
        self._buckets[key] = [
            (ts, count) for ts, count in self._buckets[key]
            if ts > cutoff
        ]

    def _calculate_count(self, key: str, now: float) -> float:
        self._cleanup_old_entries(key, now)

        if not self._buckets[key]:
            return 0.0

        window_start = now - self.window_seconds
        total = 0.0

        for ts, count in self._buckets[key]:
            if ts >= window_start:
                weight = 1.0
            else:
                overlap = (ts + self.sub_window_seconds) - window_start
                weight = max(0, overlap / self.sub_window_seconds)
            total += count * weight

        return total

    async def acquire(self, key: str, cost: int = 1) -> RateLimitResult:
        async with self._lock:
            now = time.time()
            current_window = self._get_current_window()

            current_count = self._calculate_count(key, now)
            remaining = max(0, self.limit - int(current_count))

            if self._buckets[key]:
                oldest_ts = min(ts for ts, _ in self._buckets[key])
                reset_at = oldest_ts + self.window_seconds
            else:
                reset_at = now + self.window_seconds

            if current_count + cost <= self.limit:
                found = False
                for i, (ts, count) in enumerate(self._buckets[key]):
                    if ts == current_window:
                        self._buckets[key][i] = (ts, count + cost)
                        found = True
                        break
                if not found:
                    self._buckets[key].append((current_window, cost))

                utilization = (current_count + cost) / self.limit
                self._stats.record_request(True, utilization)

                return RateLimitResult(
                    allowed=True,
                    remaining=remaining - cost,
                    limit=self.limit,
                    reset_at=reset_at,
                    message="Request allowed"
                )
            else:
                retry_after = reset_at - now
                utilization = current_count / self.limit
                self._stats.record_request(False, utilization)

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    limit=self.limit,
                    reset_at=reset_at,
                    retry_after=max(0, retry_after),
                    violation_type=ViolationType.HARD,
                    message=f"Rate limit exceeded. Retry after {retry_after:.1f}s"
                )

    async def get_remaining(self, key: str) -> int:
        async with self._lock:
            now = time.time()
            current_count = self._calculate_count(key, now)
            return max(0, self.limit - int(current_count))

    async def reset(self, key: str) -> None:
        async with self._lock:
            self._buckets.pop(key, None)

    async def get_state(self) -> Dict[str, Any]:
        async with self._lock:
            return {
                'type': 'sliding_window',
                'limit': self.limit,
                'window_seconds': self.window_seconds,
                'buckets': {k: list(v) for k, v in self._buckets.items()},
                'stats': asdict(self._stats),
            }

    async def load_state(self, state: Dict[str, Any]) -> None:
        async with self._lock:
            if state.get('type') != 'sliding_window':
                logger.warning("State type mismatch, skipping load")
                return

            now = time.time()
            cutoff = now - self.window_seconds

            for key, entries in state.get('buckets', {}).items():
                valid_entries = [(ts, count) for ts, count in entries if ts > cutoff]
                if valid_entries:
                    self._buckets[key] = valid_entries

            if 'stats' in state:
                for k, v in state['stats'].items():
                    if hasattr(self._stats, k):
                        setattr(self._stats, k, v)

class LeakyBucketRateLimiter(BaseRateLimiter):

    def __init__(
        self,
        capacity: int,
        leak_rate: float,
        config: Optional[RateLimitConfig] = None
    ):
        super().__init__(config)
        self.capacity = capacity
        self.leak_rate = leak_rate

        self._buckets: Dict[str, Tuple[float, float]] = {}

    def _get_current_level(self, key: str, now: float) -> float:
        if key not in self._buckets:
            return 0.0

        level, last_update = self._buckets[key]
        elapsed = now - last_update
        leaked = elapsed * self.leak_rate
        return max(0, level - leaked)

    async def acquire(self, key: str, cost: int = 1) -> RateLimitResult:
        async with self._lock:
            now = time.time()
            current_level = self._get_current_level(key, now)

            if current_level + cost <= self.capacity:
                new_level = current_level + cost
                self._buckets[key] = (new_level, now)

                remaining = int(self.capacity - new_level)
                reset_at = now + (new_level / self.leak_rate)
                utilization = new_level / self.capacity
                self._stats.record_request(True, utilization)

                return RateLimitResult(
                    allowed=True,
                    remaining=remaining,
                    limit=self.capacity,
                    reset_at=reset_at,
                    message="Request allowed"
                )
            else:
                excess = (current_level + cost) - self.capacity
                retry_after = excess / self.leak_rate
                reset_at = now + (current_level / self.leak_rate)
                utilization = current_level / self.capacity
                self._stats.record_request(False, utilization)

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    limit=self.capacity,
                    reset_at=reset_at,
                    retry_after=retry_after,
                    violation_type=ViolationType.HARD,
                    message=f"Bucket full. Retry after {retry_after:.1f}s"
                )

    async def get_remaining(self, key: str) -> int:
        async with self._lock:
            now = time.time()
            current_level = self._get_current_level(key, now)
            return int(self.capacity - current_level)

    async def reset(self, key: str) -> None:
        async with self._lock:
            self._buckets.pop(key, None)

    async def get_state(self) -> Dict[str, Any]:
        async with self._lock:
            return {
                'type': 'leaky_bucket',
                'capacity': self.capacity,
                'leak_rate': self.leak_rate,
                'buckets': dict(self._buckets),
                'stats': asdict(self._stats),
            }

    async def load_state(self, state: Dict[str, Any]) -> None:
        async with self._lock:
            if state.get('type') != 'leaky_bucket':
                logger.warning("State type mismatch, skipping load")
                return

            now = time.time()
            for key, bucket_data in state.get('buckets', {}).items():
                if isinstance(bucket_data, (list, tuple)) and len(bucket_data) == 2:
                    level, last_update = bucket_data
                    elapsed = now - last_update
                    current_level = max(0, level - (elapsed * self.leak_rate))
                    if current_level > 0:
                        self._buckets[key] = (current_level, now)

            if 'stats' in state:
                for k, v in state['stats'].items():
                    if hasattr(self._stats, k):
                        setattr(self._stats, k, v)

class AdaptiveRateLimiter(BaseRateLimiter):

    def __init__(
        self,
        base_limit: int,
        min_limit: int,
        max_limit: int,
        window_seconds: int = 60,
        config: Optional[RateLimitConfig] = None
    ):
        super().__init__(config)
        self.base_limit = base_limit
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.window_seconds = window_seconds

        self._effective_limits: Dict[str, int] = defaultdict(lambda: base_limit)

        self._response_history: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)

        self._limiter = SlidingWindowRateLimiter(
            limit=base_limit,
            window_seconds=window_seconds,
            config=config
        )

    def _calculate_success_rate(self, key: str, now: float) -> float:
        cutoff = now - self.window_seconds
        history = self._response_history[key]

        recent = [(ts, success) for ts, success in history if ts > cutoff]
        self._response_history[key] = recent

        if not recent:
            return 1.0

        successes = sum(1 for _, success in recent if success)
        return successes / len(recent)

    async def adjust_limit(self, key: str, success: bool) -> None:
        async with self._lock:
            now = time.time()

            self._response_history[key].append((now, success))

            success_rate = self._calculate_success_rate(key, now)
            current_limit = self._effective_limits[key]

            if success_rate < self.config.adaptive_decrease_threshold:
                new_limit = int(current_limit * (1 - self.config.adaptive_adjustment_factor))
                new_limit = max(self.min_limit, new_limit)
                if new_limit != current_limit:
                    logger.info(f"Reducing limit for {key}: {current_limit} -> {new_limit}")
            elif success_rate > self.config.adaptive_increase_threshold:
                new_limit = int(current_limit * (1 + self.config.adaptive_adjustment_factor))
                new_limit = min(self.max_limit, new_limit)
                if new_limit != current_limit:
                    logger.info(f"Increasing limit for {key}: {current_limit} -> {new_limit}")
            else:
                new_limit = current_limit

            self._effective_limits[key] = new_limit

    async def acquire(self, key: str, cost: int = 1) -> RateLimitResult:
        async with self._lock:
            effective_limit = self._effective_limits[key]
            self._limiter.limit = effective_limit

        return await self._limiter.acquire(key, cost)

    async def get_remaining(self, key: str) -> int:
        return await self._limiter.get_remaining(key)

    async def reset(self, key: str) -> None:
        async with self._lock:
            self._effective_limits.pop(key, None)
            self._response_history.pop(key, None)
        await self._limiter.reset(key)

    async def get_state(self) -> Dict[str, Any]:
        limiter_state = await self._limiter.get_state()
        async with self._lock:
            return {
                'type': 'adaptive',
                'base_limit': self.base_limit,
                'min_limit': self.min_limit,
                'max_limit': self.max_limit,
                'effective_limits': dict(self._effective_limits),
                'limiter_state': limiter_state,
                'stats': asdict(self._stats),
            }

    async def load_state(self, state: Dict[str, Any]) -> None:
        if state.get('type') != 'adaptive':
            logger.warning("State type mismatch, skipping load")
            return

        async with self._lock:
            for key, limit in state.get('effective_limits', {}).items():
                self._effective_limits[key] = limit

        if 'limiter_state' in state:
            await self._limiter.load_state(state['limiter_state'])

        if 'stats' in state:
            for k, v in state['stats'].items():
                if hasattr(self._stats, k):
                    setattr(self._stats, k, v)

class UserRateLimiter:

    def __init__(
        self,
        tier_limits: Optional[Dict[RateLimitTier, TierLimits]] = None,
        config: Optional[RateLimitConfig] = None
    ):
        self.config = config or RateLimitConfig()
        self.tier_limits = tier_limits or DEFAULT_TIER_LIMITS

        self._lock = asyncio.Lock()

        self._minute_limiters: Dict[str, SlidingWindowRateLimiter] = {}
        self._hour_limiters: Dict[str, SlidingWindowRateLimiter] = {}
        self._day_limiters: Dict[str, SlidingWindowRateLimiter] = {}

        self._user_tiers: Dict[str, RateLimitTier] = {}

        self._violations: Dict[str, List[float]] = defaultdict(list)
        self._abuse_cooldowns: Dict[str, float] = {}
        self._abuse_counts: Dict[str, int] = defaultdict(int)

    def _get_user_tier(self, user_id: str) -> RateLimitTier:
        return self._user_tiers.get(user_id, RateLimitTier.FREE)

    def _get_tier_limits(self, tier: RateLimitTier) -> TierLimits:
        return self.tier_limits.get(tier, self.tier_limits[RateLimitTier.FREE])

    async def set_user_tier(self, user_id: str, tier: RateLimitTier) -> None:
        async with self._lock:
            old_tier = self._user_tiers.get(user_id)
            self._user_tiers[user_id] = tier

            if old_tier != tier:
                self._minute_limiters.pop(user_id, None)
                self._hour_limiters.pop(user_id, None)
                self._day_limiters.pop(user_id, None)
                logger.info(f"User {user_id} tier changed: {old_tier} -> {tier}")

    def _get_or_create_limiters(
        self,
        user_id: str,
        limits: TierLimits
    ) -> Tuple[SlidingWindowRateLimiter, SlidingWindowRateLimiter, SlidingWindowRateLimiter]:
        if user_id not in self._minute_limiters:
            self._minute_limiters[user_id] = SlidingWindowRateLimiter(
                limit=limits.requests_per_minute,
                window_seconds=60,
                config=self.config
            )
        if user_id not in self._hour_limiters:
            self._hour_limiters[user_id] = SlidingWindowRateLimiter(
                limit=limits.requests_per_hour,
                window_seconds=3600,
                config=self.config
            )
        if user_id not in self._day_limiters:
            self._day_limiters[user_id] = SlidingWindowRateLimiter(
                limit=limits.requests_per_day,
                window_seconds=86400,
                config=self.config
            )

        return (
            self._minute_limiters[user_id],
            self._hour_limiters[user_id],
            self._day_limiters[user_id]
        )

    def _check_abuse_cooldown(self, user_id: str, now: float) -> Optional[RateLimitResult]:
        if user_id in self._abuse_cooldowns:
            cooldown_until = self._abuse_cooldowns[user_id]
            if now < cooldown_until:
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    limit=0,
                    reset_at=cooldown_until,
                    retry_after=cooldown_until - now,
                    violation_type=ViolationType.ABUSE,
                    message=f"Account in cooldown due to abuse. Retry after {cooldown_until - now:.0f}s"
                )
            else:
                del self._abuse_cooldowns[user_id]
        return None

    def _record_violation(self, user_id: str, now: float) -> bool:
        cutoff = now - self.config.violation_window_seconds
        self._violations[user_id] = [
            ts for ts in self._violations[user_id] if ts > cutoff
        ]

        self._violations[user_id].append(now)

        if len(self._violations[user_id]) >= self.config.violation_threshold:
            self._abuse_counts[user_id] += 1
            multiplier = self.config.abuse_multiplier ** (self._abuse_counts[user_id] - 1)
            cooldown = min(
                self.config.abuse_cooldown_seconds * multiplier,
                self.config.max_abuse_cooldown_seconds
            )
            self._abuse_cooldowns[user_id] = now + cooldown
            self._violations[user_id] = []
            logger.warning(
                f"Abuse detected for user {user_id}. "
                f"Cooldown: {cooldown}s (count: {self._abuse_counts[user_id]})"
            )
            return True

        return False

    async def acquire(
        self,
        user_id: str,
        cost: int = 1
    ) -> RateLimitResult:
        async with self._lock:
            now = time.time()

            abuse_result = self._check_abuse_cooldown(user_id, now)
            if abuse_result:
                return abuse_result

            tier = self._get_user_tier(user_id)
            limits = self._get_tier_limits(tier)

            minute_limiter, hour_limiter, day_limiter = self._get_or_create_limiters(
                user_id, limits
            )

        results = await asyncio.gather(
            minute_limiter.acquire(user_id, cost),
            hour_limiter.acquire(user_id, cost),
            day_limiter.acquire(user_id, cost)
        )

        for result in results:
            if not result.allowed:
                async with self._lock:
                    self._record_violation(user_id, time.time())
                return result

        return min(results, key=lambda r: r.remaining)

    async def get_user_status(self, user_id: str) -> Dict[str, Any]:
        async with self._lock:
            now = time.time()
            tier = self._get_user_tier(user_id)
            limits = self._get_tier_limits(tier)

            minute_limiter, hour_limiter, day_limiter = self._get_or_create_limiters(
                user_id, limits
            )

            minute_remaining = await minute_limiter.get_remaining(user_id)
            hour_remaining = await hour_limiter.get_remaining(user_id)
            day_remaining = await day_limiter.get_remaining(user_id)

            abuse_cooldown = self._abuse_cooldowns.get(user_id)

            return {
                'user_id': user_id,
                'tier': tier.value,
                'limits': {
                    'per_minute': limits.requests_per_minute,
                    'per_hour': limits.requests_per_hour,
                    'per_day': limits.requests_per_day,
                },
                'remaining': {
                    'per_minute': minute_remaining,
                    'per_hour': hour_remaining,
                    'per_day': day_remaining,
                },
                'violations': len(self._violations.get(user_id, [])),
                'abuse_count': self._abuse_counts.get(user_id, 0),
                'in_cooldown': abuse_cooldown is not None and now < abuse_cooldown,
                'cooldown_until': abuse_cooldown if abuse_cooldown and now < abuse_cooldown else None,
            }

    async def reset_user(self, user_id: str) -> None:
        async with self._lock:
            self._minute_limiters.pop(user_id, None)
            self._hour_limiters.pop(user_id, None)
            self._day_limiters.pop(user_id, None)
            self._violations.pop(user_id, None)
            self._abuse_cooldowns.pop(user_id, None)
            self._abuse_counts.pop(user_id, None)
            logger.info(f"Reset all limits for user {user_id}")

    async def get_state(self) -> Dict[str, Any]:
        async with self._lock:
            return {
                'type': 'user_rate_limiter',
                'user_tiers': {k: v.value for k, v in self._user_tiers.items()},
                'violations': dict(self._violations),
                'abuse_cooldowns': dict(self._abuse_cooldowns),
                'abuse_counts': dict(self._abuse_counts),
            }

    async def load_state(self, state: Dict[str, Any]) -> None:
        if state.get('type') != 'user_rate_limiter':
            logger.warning("State type mismatch, skipping load")
            return

        async with self._lock:
            now = time.time()

            for user_id, tier_value in state.get('user_tiers', {}).items():
                try:
                    self._user_tiers[user_id] = RateLimitTier(tier_value)
                except ValueError:
                    logger.warning(f"Unknown tier {tier_value} for user {user_id}")

            cutoff = now - self.config.violation_window_seconds
            for user_id, timestamps in state.get('violations', {}).items():
                valid = [ts for ts in timestamps if ts > cutoff]
                if valid:
                    self._violations[user_id] = valid

            for user_id, until in state.get('abuse_cooldowns', {}).items():
                if until > now:
                    self._abuse_cooldowns[user_id] = until

            self._abuse_counts.update(state.get('abuse_counts', {}))

class OperationRateLimiter:

    def __init__(
        self,
        operation_limits: Optional[Dict[OperationType, int]] = None,
        default_limit: int = 60,
        window_seconds: int = 60,
        config: Optional[RateLimitConfig] = None
    ):
        self.config = config or RateLimitConfig()
        self.default_limit = default_limit
        self.window_seconds = window_seconds

        self.operation_limits = operation_limits or {
            OperationType.WALLET_CREATE: 5,
            OperationType.WALLET_EXPORT: 10,
            OperationType.WALLET_IMPORT: 10,
            OperationType.WALLET_BALANCE: 60,
            OperationType.TRADE_BUY: 30,
            OperationType.TRADE_SELL: 30,
            OperationType.TRADE_SWAP: 30,
            OperationType.QUERY_PRICE: 120,
            OperationType.QUERY_PORTFOLIO: 60,
            OperationType.QUERY_HISTORY: 30,
            OperationType.API_CALL: 100,
            OperationType.API_BATCH: 10,
            OperationType.ADMIN_ACTION: 20,
            OperationType.GENERIC: 60,
        }

        self._lock = asyncio.Lock()
        self._limiters: Dict[str, SlidingWindowRateLimiter] = {}

    def _get_limiter(self, operation: OperationType) -> SlidingWindowRateLimiter:
        key = operation.value
        if key not in self._limiters:
            limit = self.operation_limits.get(operation, self.default_limit)
            self._limiters[key] = SlidingWindowRateLimiter(
                limit=limit,
                window_seconds=self.window_seconds,
                config=self.config
            )
        return self._limiters[key]

    async def acquire(
        self,
        user_id: str,
        operation: OperationType,
        cost: int = 1
    ) -> RateLimitResult:
        async with self._lock:
            limiter = self._get_limiter(operation)

        key = f"{user_id}:{operation.value}"
        result = await limiter.acquire(key, cost)

        if not result.allowed:
            result.message = f"Rate limit exceeded for {operation.value}. {result.message}"

        return result

    async def get_remaining(
        self,
        user_id: str,
        operation: OperationType
    ) -> int:
        async with self._lock:
            limiter = self._get_limiter(operation)
        key = f"{user_id}:{operation.value}"
        return await limiter.get_remaining(key)

    async def get_operation_status(
        self,
        user_id: str
    ) -> Dict[str, Dict[str, Any]]:
        status = {}
        for op in OperationType:
            remaining = await self.get_remaining(user_id, op)
            limit = self.operation_limits.get(op, self.default_limit)
            status[op.value] = {
                'limit': limit,
                'remaining': remaining,
                'window_seconds': self.window_seconds,
            }
        return status

    async def get_state(self) -> Dict[str, Any]:
        states = {}
        for key, limiter in self._limiters.items():
            states[key] = await limiter.get_state()
        return {
            'type': 'operation_rate_limiter',
            'limiter_states': states,
        }

    async def load_state(self, state: Dict[str, Any]) -> None:
        if state.get('type') != 'operation_rate_limiter':
            return

        for key, limiter_state in state.get('limiter_states', {}).items():
            try:
                op = OperationType(key)
                limiter = self._get_limiter(op)
                await limiter.load_state(limiter_state)
            except ValueError:
                logger.warning(f"Unknown operation type: {key}")

class GlobalRateLimiter:

    def __init__(
        self,
        total_limit: int = 10000,
        window_seconds: int = 60,
        config: Optional[RateLimitConfig] = None
    ):
        self.config = config or RateLimitConfig()
        self.total_limit = total_limit

        self._limiter = SlidingWindowRateLimiter(
            limit=total_limit,
            window_seconds=window_seconds,
            config=config
        )

        self._endpoint_limiters: Dict[str, SlidingWindowRateLimiter] = {}
        self._endpoint_limits: Dict[str, int] = {}

    def set_endpoint_limit(self, endpoint: str, limit: int) -> None:
        self._endpoint_limits[endpoint] = limit
        self._endpoint_limiters[endpoint] = SlidingWindowRateLimiter(
            limit=limit,
            window_seconds=60,
            config=self.config
        )

    async def acquire(
        self,
        cost: int = 1,
        endpoint: Optional[str] = None
    ) -> RateLimitResult:
        global_result = await self._limiter.acquire("global", cost)
        if not global_result.allowed:
            global_result.message = f"System rate limit exceeded. {global_result.message}"
            return global_result

        if endpoint and endpoint in self._endpoint_limiters:
            endpoint_result = await self._endpoint_limiters[endpoint].acquire(endpoint, cost)
            if not endpoint_result.allowed:
                endpoint_result.message = f"Endpoint {endpoint} rate limit exceeded. {endpoint_result.message}"
                return endpoint_result

        return global_result

    async def get_system_status(self) -> Dict[str, Any]:
        remaining = await self._limiter.get_remaining("global")

        endpoint_status = {}
        for endpoint, limiter in self._endpoint_limiters.items():
            ep_remaining = await limiter.get_remaining(endpoint)
            endpoint_status[endpoint] = {
                'limit': self._endpoint_limits[endpoint],
                'remaining': ep_remaining,
            }

        return {
            'global': {
                'limit': self.total_limit,
                'remaining': remaining,
            },
            'endpoints': endpoint_status,
            'stats': asdict(self._limiter.stats),
        }

    async def get_state(self) -> Dict[str, Any]:
        limiter_state = await self._limiter.get_state()
        endpoint_states = {}
        for endpoint, limiter in self._endpoint_limiters.items():
            endpoint_states[endpoint] = await limiter.get_state()

        return {
            'type': 'global_rate_limiter',
            'limiter_state': limiter_state,
            'endpoint_states': endpoint_states,
            'endpoint_limits': self._endpoint_limits,
        }

    async def load_state(self, state: Dict[str, Any]) -> None:
        if state.get('type') != 'global_rate_limiter':
            return

        if 'limiter_state' in state:
            await self._limiter.load_state(state['limiter_state'])

        for endpoint, limit in state.get('endpoint_limits', {}).items():
            self.set_endpoint_limit(endpoint, limit)

        for endpoint, ep_state in state.get('endpoint_states', {}).items():
            if endpoint in self._endpoint_limiters:
                await self._endpoint_limiters[endpoint].load_state(ep_state)

class CompositeRateLimiter:

    def __init__(
        self,
        user_limiter: Optional[UserRateLimiter] = None,
        operation_limiter: Optional[OperationRateLimiter] = None,
        global_limiter: Optional[GlobalRateLimiter] = None,
        config: Optional[RateLimitConfig] = None
    ):
        self.config = config or RateLimitConfig()
        self.user_limiter = user_limiter or UserRateLimiter(config=config)
        self.operation_limiter = operation_limiter or OperationRateLimiter(config=config)
        self.global_limiter = global_limiter or GlobalRateLimiter(config=config)

        self._persistence_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def acquire(
        self,
        user_id: str,
        operation: OperationType = OperationType.GENERIC,
        cost: int = 1,
        endpoint: Optional[str] = None
    ) -> RateLimitResult:
        global_result = await self.global_limiter.acquire(cost, endpoint)
        if not global_result.allowed:
            return global_result

        user_result = await self.user_limiter.acquire(user_id, cost)
        if not user_result.allowed:
            return user_result

        op_result = await self.operation_limiter.acquire(user_id, operation, cost)
        if not op_result.allowed:
            return op_result

        results = [global_result, user_result, op_result]
        return min(results, key=lambda r: r.remaining)

    async def get_comprehensive_status(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        return {
            'user': await self.user_limiter.get_user_status(user_id),
            'operations': await self.operation_limiter.get_operation_status(user_id),
            'global': await self.global_limiter.get_system_status(),
        }

    async def start(self) -> None:
        if self._running:
            return

        self._running = True

        if self.config.persistence_enabled:
            self._persistence_task = asyncio.create_task(self._persistence_loop())

        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("CompositeRateLimiter started")

    async def stop(self) -> None:
        self._running = False

        if self._persistence_task:
            self._persistence_task.cancel()
            try:
                await self._persistence_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self.config.persistence_enabled:
            await self.save_state()

        logger.info("CompositeRateLimiter stopped")

    async def _persistence_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.config.persistence_interval_seconds)
                await self.save_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in persistence loop: {e}")

    async def _cleanup_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.config.cleanup_interval_seconds)
                logger.debug("Cleanup cycle completed")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def save_state(self) -> None:
        try:
            path = Path(self.config.persistence_path)
            path.mkdir(parents=True, exist_ok=True)

            state = {
                'timestamp': time.time(),
                'user_limiter': await self.user_limiter.get_state(),
                'operation_limiter': await self.operation_limiter.get_state(),
                'global_limiter': await self.global_limiter.get_state(),
            }

            state_file = path / 'rate_limiter_state.json'
            temp_file = path / 'rate_limiter_state.json.tmp'

            with open(temp_file, 'w') as f:
                json.dump(state, f, indent=2)

            temp_file.replace(state_file)

            logger.debug(f"State saved to {state_file}")

        except Exception as e:
            logger.error(f"Error saving state: {e}")

    async def load_state(self) -> bool:
        try:
            state_file = Path(self.config.persistence_path) / 'rate_limiter_state.json'

            if not state_file.exists():
                logger.info("No persisted state found")
                return False

            with open(state_file, 'r') as f:
                state = json.load(f)

            age = time.time() - state.get('timestamp', 0)
            if age > self.config.entry_ttl_seconds:
                logger.info(f"Persisted state too old ({age:.0f}s), ignoring")
                return False

            await self.user_limiter.load_state(state.get('user_limiter', {}))
            await self.operation_limiter.load_state(state.get('operation_limiter', {}))
            await self.global_limiter.load_state(state.get('global_limiter', {}))

            logger.info(f"State loaded from {state_file}")
            return True

        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False

def rate_limit(
    limiter: Union[BaseRateLimiter, CompositeRateLimiter],
    user_id_param: str = 'user_id',
    operation: OperationType = OperationType.GENERIC,
    cost: int = 1,
    on_limited: Optional[Callable[[RateLimitResult], Any]] = None
):
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user_id = kwargs.get(user_id_param)
            if user_id is None:
                import inspect
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                if user_id_param in params:
                    idx = params.index(user_id_param)
                    if idx < len(args):
                        user_id = args[idx]

            if user_id is None:
                raise ValueError(f"Could not find {user_id_param} in function arguments")

            if isinstance(limiter, CompositeRateLimiter):
                result = await limiter.acquire(user_id, operation, cost)
            else:
                result = await limiter.acquire(str(user_id), cost)

            if not result.allowed:
                if on_limited:
                    return on_limited(result)
                raise RateLimitExceeded(result)

            return await func(*args, **kwargs)

        return wrapper
    return decorator

def rate_limit_sync(
    limiter: BaseRateLimiter,
    key_param: str = 'key',
    cost: int = 1
):
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = kwargs.get(key_param, 'default')

            async def check():
                return await limiter.acquire(str(key), cost)

            result = asyncio.run(check())

            if not result.allowed:
                raise RateLimitExceeded(result)

            return func(*args, **kwargs)

        return wrapper
    return decorator

class RateLimitExceeded(Exception):

    def __init__(self, result: RateLimitResult):
        self.result = result
        super().__init__(result.message)

    @property
    def retry_after(self) -> Optional[float]:
        return self.result.retry_after

    @property
    def headers(self) -> Dict[str, str]:
        return self.result.headers

class RateLimitMonitor:

    def __init__(
        self,
        composite_limiter: CompositeRateLimiter,
        alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        self.limiter = composite_limiter
        self.alert_callback = alert_callback or self._default_alert

        self._alert_thresholds = {
            'high_utilization': 0.9,
            'abuse_detected': True,
            'global_limit_warning': 0.8,
        }

        self._alert_cooldowns: Dict[str, float] = {}
        self._alert_cooldown_seconds = 300

    def _default_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        logger.warning(f"RATE LIMIT ALERT [{alert_type}]: {json.dumps(data)}")

    def _can_alert(self, alert_key: str) -> bool:
        now = time.time()
        if alert_key in self._alert_cooldowns:
            if now < self._alert_cooldowns[alert_key]:
                return False
        self._alert_cooldowns[alert_key] = now + self._alert_cooldown_seconds
        return True

    async def check_and_alert(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        alerts = []

        global_status = await self.limiter.global_limiter.get_system_status()
        global_utilization = 1 - (global_status['global']['remaining'] / global_status['global']['limit'])

        if global_utilization >= self._alert_thresholds['global_limit_warning']:
            alert_key = 'global_high_utilization'
            if self._can_alert(alert_key):
                alert = {
                    'type': 'global_high_utilization',
                    'utilization': global_utilization,
                    'remaining': global_status['global']['remaining'],
                    'limit': global_status['global']['limit'],
                }
                alerts.append(alert)
                self.alert_callback('global_high_utilization', alert)

        if user_id:
            user_status = await self.limiter.user_limiter.get_user_status(user_id)

            if user_status.get('in_cooldown'):
                alert_key = f'user_abuse_{user_id}'
                if self._can_alert(alert_key):
                    alert = {
                        'type': 'user_abuse',
                        'user_id': user_id,
                        'abuse_count': user_status['abuse_count'],
                        'cooldown_until': user_status['cooldown_until'],
                    }
                    alerts.append(alert)
                    self.alert_callback('user_abuse', alert)

        return alerts

    async def get_metrics(self) -> Dict[str, Any]:
        global_status = await self.limiter.global_limiter.get_system_status()

        return {
            'timestamp': time.time(),
            'global': {
                'utilization': 1 - (global_status['global']['remaining'] / global_status['global']['limit']),
                'remaining': global_status['global']['remaining'],
                'limit': global_status['global']['limit'],
            },
            'stats': global_status.get('stats', {}),
        }

def create_default_limiter(config: Optional[RateLimitConfig] = None) -> CompositeRateLimiter:
    config = config or RateLimitConfig.from_env()

    return CompositeRateLimiter(
        user_limiter=UserRateLimiter(config=config),
        operation_limiter=OperationRateLimiter(config=config),
        global_limiter=GlobalRateLimiter(config=config),
        config=config
    )

async def create_and_start_limiter(
    config: Optional[RateLimitConfig] = None,
    load_persisted: bool = True
) -> CompositeRateLimiter:
    limiter = create_default_limiter(config)

    if load_persisted:
        await limiter.load_state()

    await limiter.start()

    return limiter

if __name__ == "__main__":
    async def test():
        print("Creating rate limiter...")
        limiter = await create_and_start_limiter()

        print("\nTesting basic rate limiting:")
        for i in range(5):
            result = await limiter.acquire(
                user_id="test_user",
                operation=OperationType.TRADE_BUY
            )
            print(f"  Request {i+1}: allowed={result.allowed}, remaining={result.remaining}")

        print("\nUser status:")
        status = await limiter.get_comprehensive_status("test_user")
        print(f"  Tier: {status['user']['tier']}")
        print(f"  Remaining (minute): {status['user']['remaining']['per_minute']}")

        print("\nSetting user to premium tier:")
        await limiter.user_limiter.set_user_tier("test_user", RateLimitTier.PREMIUM)
        status = await limiter.get_comprehensive_status("test_user")
        print(f"  New tier: {status['user']['tier']}")
        print(f"  New limit (minute): {status['user']['limits']['per_minute']}")

        await limiter.stop()
        print("\nTest completed!")

    asyncio.run(test())
