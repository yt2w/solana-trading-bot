"""
Token Scanner - Rug Pull Detection & Token Safety Analysis
Production-grade token safety scanner for Solana tokens.
"""

import asyncio
import aiohttp
import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, List, Any, Callable, Set, Tuple
from collections import defaultdict


logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Token risk level classification."""
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    DANGER = "DANGER"
    SCAM = "SCAM"


class CheckStatus(Enum):
    """Status of individual safety check."""
    PASSED = "PASSED"
    WARNING = "WARNING"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"
    SKIPPED = "SKIPPED"


class AlertType(Enum):
    """Types of real-time alerts."""
    LP_REMOVAL = "LP_REMOVAL"
    LARGE_SELL = "LARGE_SELL"
    AUTHORITY_CHANGE = "AUTHORITY_CHANGE"
    HOLDER_CONCENTRATION = "HOLDER_CONCENTRATION"
    PRICE_CRASH = "PRICE_CRASH"
    SUSPICIOUS_ACTIVITY = "SUSPICIOUS_ACTIVITY"


KNOWN_SAFE_TOKENS: Set[str] = {
    "So11111111111111111111111111111111111111112",
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
    "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",
    "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj",
    "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
    "jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL",
    "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3",
    "rndrizKT3MK1iimdxRdWabcF7Zg7AR5T4nud4EkHBof",
}

KNOWN_SCAM_TOKENS: Set[str] = set()

CHECK_WEIGHTS: Dict[str, float] = {
    "mint_authority": 25.0,
    "freeze_authority": 15.0,
    "lp_locked": 20.0,
    "holder_concentration": 10.0,
    "creator_holdings": 8.0,
    "honeypot": 15.0,
    "liquidity_depth": 10.0,
    "token_age": 5.0,
    "social_presence": 2.0,
}

CRITICAL_FLAGS: Set[str] = {
    "mint_authority_active",
    "honeypot_detected",
    "zero_liquidity",
    "confirmed_scam",
}


@dataclass
class SafetyCheck:
    """Result of an individual safety check."""
    name: str
    status: CheckStatus
    value: Any
    message: str
    weight: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def score_contribution(self) -> float:
        if self.status == CheckStatus.PASSED:
            return self.weight
        elif self.status == CheckStatus.WARNING:
            return self.weight * 0.5
        return 0.0

    @property
    def is_critical_failure(self) -> bool:
        return (self.status == CheckStatus.FAILED and
                self.name in ["mint_authority", "honeypot", "confirmed_scam"])


@dataclass
class TokenSafetyReport:
    """Comprehensive token safety analysis report."""
    mint: str
    risk_score: float
    risk_level: RiskLevel
    checks: Dict[str, SafetyCheck]
    flags: List[str]
    warnings: List[str]
    recommendation: str
    confidence: float
    token_info: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    scan_duration_ms: float = 0.0
    data_sources_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mint": self.mint,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level.value,
            "checks": {name: {"status": c.status.value, "value": c.value,
                             "message": c.message, "weight": c.weight}
                      for name, c in self.checks.items()},
            "flags": self.flags,
            "warnings": self.warnings,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "token_info": self.token_info,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TokenAlert:
    """Real-time monitoring alert."""
    mint: str
    alert_type: AlertType
    severity: str
    message: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HolderInfo:
    """Token holder information."""
    address: str
    balance: float
    percentage: float
    is_creator: bool = False
    is_lp: bool = False


@dataclass
class LiquidityInfo:
    """Liquidity pool information."""
    pool_address: str
    dex: str
    base_amount: float
    quote_amount: float
    liquidity_usd: float
    is_locked: bool
    lock_end_time: Optional[datetime]
    lock_percentage: float


class SolanaRPCClient:
    """Solana RPC client for on-chain data."""

    def __init__(self, endpoint: str, rate_limit: int = 10):
        self.endpoint = endpoint
        self.rate_limit = rate_limit
        self._semaphore = asyncio.Semaphore(rate_limit)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _rpc_call(self, method: str, params: List[Any]) -> Optional[Dict]:
        """Make RPC call with rate limiting."""
        async with self._semaphore:
            await self._ensure_session()
            payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
            try:
                async with self._session.post(
                    self.endpoint, json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("result")
                    logger.warning(f"RPC call failed: {response.status}")
                    return None
            except Exception as e:
                logger.error(f"RPC error: {e}")
                return None

    async def get_account_info(self, address: str) -> Optional[Dict]:
        return await self._rpc_call("getAccountInfo", [address, {"encoding": "jsonParsed"}])

    async def get_mint_info(self, mint: str) -> Optional[Dict]:
        result = await self.get_account_info(mint)
        if result and result.get("value"):
            return result["value"].get("data", {}).get("parsed", {}).get("info", {})
        return None

    async def get_token_supply(self, mint: str) -> Optional[Dict]:
        return await self._rpc_call("getTokenSupply", [mint])

    async def get_token_largest_accounts(self, mint: str) -> Optional[List[Dict]]:
        result = await self._rpc_call("getTokenLargestAccounts", [mint])
        if result and result.get("value"):
            return result["value"]
        return None

    async def get_signatures_for_address(self, address: str, limit: int = 10) -> Optional[List[Dict]]:
        return await self._rpc_call("getSignaturesForAddress", [address, {"limit": limit}])


class BirdeyeClient:
    """Birdeye API client for token metadata and analytics."""

    BASE_URL = "https://public-api.birdeye.so"

    def __init__(self, api_key: str, rate_limit: int = 5):
        self.api_key = api_key
        self.rate_limit = rate_limit
        self._semaphore = asyncio.Semaphore(rate_limit)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"X-API-KEY": self.api_key, "x-chain": "solana"}
            )

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        async with self._semaphore:
            await self._ensure_session()
            try:
                async with self._session.get(
                    f"{self.BASE_URL}{endpoint}", params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
            except Exception as e:
                logger.error(f"Birdeye error: {e}")
                return None

    async def get_token_overview(self, mint: str) -> Optional[Dict]:
        result = await self._get("/defi/token_overview", {"address": mint})
        return result.get("data") if result else None

    async def get_token_security(self, mint: str) -> Optional[Dict]:
        result = await self._get("/defi/token_security", {"address": mint})
        return result.get("data") if result else None

    async def get_token_holders(self, mint: str, limit: int = 20) -> Optional[List]:
        result = await self._get("/defi/token_holder", {"address": mint, "offset": 0, "limit": limit})
        return result.get("data", {}).get("items") if result else None

    async def get_token_creation_info(self, mint: str) -> Optional[Dict]:
        result = await self._get("/defi/token_creation_info", {"address": mint})
        return result.get("data") if result else None


class DexScreenerClient:
    """DexScreener API client for liquidity and market data."""

    BASE_URL = "https://api.dexscreener.com/latest"

    def __init__(self, rate_limit: int = 5):
        self.rate_limit = rate_limit
        self._semaphore = asyncio.Semaphore(rate_limit)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get(self, endpoint: str) -> Optional[Dict]:
        async with self._semaphore:
            await self._ensure_session()
            try:
                async with self._session.get(
                    f"{self.BASE_URL}{endpoint}",
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
            except Exception as e:
                logger.error(f"DexScreener error: {e}")
                return None

    async def get_token_pairs(self, mint: str) -> Optional[List[Dict]]:
        result = await self._get(f"/dex/tokens/{mint}")
        return result.get("pairs") if result else None

    async def get_pair_info(self, pair_address: str) -> Optional[Dict]:
        result = await self._get(f"/dex/pairs/solana/{pair_address}")
        if result and result.get("pairs"):
            return result["pairs"][0]
        return None


class RugCheckClient:
    """RugCheck API client for rug pull analysis."""

    BASE_URL = "https://api.rugcheck.xyz/v1"

    def __init__(self, rate_limit: int = 3):
        self.rate_limit = rate_limit
        self._semaphore = asyncio.Semaphore(rate_limit)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get(self, endpoint: str) -> Optional[Dict]:
        async with self._semaphore:
            await self._ensure_session()
            try:
                async with self._session.get(
                    f"{self.BASE_URL}{endpoint}",
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
            except Exception as e:
                logger.error(f"RugCheck error: {e}")
                return None

    async def get_token_report(self, mint: str) -> Optional[Dict]:
        return await self._get(f"/tokens/{mint}/report")

    async def get_token_summary(self, mint: str) -> Optional[Dict]:
        return await self._get(f"/tokens/{mint}/report/summary")


class HeliusClient:
    """Helius API client for enhanced Solana data."""

    BASE_URL = "https://api.helius.xyz/v0"

    def __init__(self, api_key: str, rate_limit: int = 10):
        self.api_key = api_key
        self.rate_limit = rate_limit
        self._semaphore = asyncio.Semaphore(rate_limit)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        async with self._semaphore:
            await self._ensure_session()
            params = params or {}
            params["api-key"] = self.api_key
            try:
                async with self._session.get(
                    f"{self.BASE_URL}{endpoint}", params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
            except Exception as e:
                logger.error(f"Helius error: {e}")
                return None

    async def _post(self, endpoint: str, data: Dict) -> Optional[Dict]:
        async with self._semaphore:
            await self._ensure_session()
            try:
                async with self._session.post(
                    f"{self.BASE_URL}{endpoint}?api-key={self.api_key}",
                    json=data, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
            except Exception as e:
                logger.error(f"Helius error: {e}")
                return None

    async def get_token_metadata(self, mint: str) -> Optional[Dict]:
        result = await self._post("/token-metadata", {"mintAccounts": [mint]})
        if result and len(result) > 0:
            return result[0]
        return None


class TokenCache:
    """In-memory cache with TTL support."""

    def __init__(self, default_ttl: int = 60):
        self.default_ttl = default_ttl
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._lock = asyncio.Lock()

    def _make_key(self, category: str, identifier: str) -> str:
        return f"{category}:{identifier}"

    async def get(self, category: str, identifier: str) -> Optional[Any]:
        key = self._make_key(category, identifier)
        async with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if datetime.utcnow() < expiry:
                    return value
                del self._cache[key]
            return None

    async def set(self, category: str, identifier: str, value: Any, ttl: Optional[int] = None):
        key = self._make_key(category, identifier)
        ttl = ttl or self.default_ttl
        expiry = datetime.utcnow() + timedelta(seconds=ttl)
        async with self._lock:
            self._cache[key] = (value, expiry)

    async def invalidate(self, category: str, identifier: str):
        key = self._make_key(category, identifier)
        async with self._lock:
            self._cache.pop(key, None)

    async def clear(self):
        async with self._lock:
            self._cache.clear()


class TokenScanner:
    """
    Production-grade token safety scanner for Solana tokens.

    Features:
    - Multiple data source integration
    - Comprehensive safety checks
    - Risk score calculation
    - Real-time monitoring
    - Caching for performance
    """

    def __init__(
        self,
        rpc_endpoint: str,
        birdeye_api_key: Optional[str] = None,
        helius_api_key: Optional[str] = None,
        cache_ttl: int = 60,
        min_liquidity_usd: float = 10000.0,
        max_holder_concentration: float = 50.0,
        min_token_age_minutes: int = 60,
    ):
        self.min_liquidity_usd = min_liquidity_usd
        self.max_holder_concentration = max_holder_concentration
        self.min_token_age_minutes = min_token_age_minutes


        self.rpc = SolanaRPCClient(rpc_endpoint)
        self.birdeye = BirdeyeClient(birdeye_api_key) if birdeye_api_key else None
        self.helius = HeliusClient(helius_api_key) if helius_api_key else None
        self.dexscreener = DexScreenerClient()
        self.rugcheck = RugCheckClient()


        self.cache = TokenCache(default_ttl=cache_ttl)


        self.whitelist: Set[str] = set(KNOWN_SAFE_TOKENS)
        self.blacklist: Set[str] = set(KNOWN_SCAM_TOKENS)
        self.user_whitelist: Set[str] = set()
        self.user_blacklist: Set[str] = set()


        self._monitors: Dict[str, asyncio.Task] = {}
        self._alert_callbacks: Dict[str, List[Callable]] = defaultdict(list)

        logger.info("TokenScanner initialized")

    async def close(self):
        """Close all client connections."""
        await self.rpc.close()
        await self.dexscreener.close()
        await self.rugcheck.close()
        if self.birdeye:
            await self.birdeye.close()
        if self.helius:
            await self.helius.close()
        for task in self._monitors.values():
            task.cancel()
        logger.info("TokenScanner closed")


    def add_to_whitelist(self, mint: str):
        self.user_whitelist.add(mint)

    def remove_from_whitelist(self, mint: str):
        self.user_whitelist.discard(mint)

    def add_to_blacklist(self, mint: str):
        self.user_blacklist.add(mint)

    def remove_from_blacklist(self, mint: str):
        self.user_blacklist.discard(mint)

    def is_whitelisted(self, mint: str) -> bool:
        return mint in self.whitelist or mint in self.user_whitelist

    def is_blacklisted(self, mint: str) -> bool:
        return mint in self.blacklist or mint in self.user_blacklist


    async def check_mint_authority(self, mint: str) -> SafetyCheck:
        """Check if mint authority is revoked. Active = CRITICAL risk."""
        try:
            cached = await self.cache.get("mint_authority", mint)
            if cached is not None:
                return cached

            mint_info = await self.rpc.get_mint_info(mint)
            if mint_info is None:
                return SafetyCheck(
                    name="mint_authority", status=CheckStatus.UNKNOWN,
                    value=None, message="Could not fetch mint info",
                    weight=CHECK_WEIGHTS["mint_authority"]
                )

            mint_authority = mint_info.get("mintAuthority")
            if mint_authority is None:
                result = SafetyCheck(
                    name="mint_authority", status=CheckStatus.PASSED,
                    value=None, message="Mint authority is revoked",
                    weight=CHECK_WEIGHTS["mint_authority"],
                    details={"revoked": True}
                )
            else:
                result = SafetyCheck(
                    name="mint_authority", status=CheckStatus.FAILED,
                    value=mint_authority,
                    message=f"MINT AUTHORITY ACTIVE: {mint_authority[:8]}...",
                    weight=CHECK_WEIGHTS["mint_authority"],
                    details={"revoked": False, "authority": mint_authority}
                )

            await self.cache.set("mint_authority", mint, result, ttl=300)
            return result
        except Exception as e:
            logger.error(f"Error checking mint authority: {e}")
            return SafetyCheck(
                name="mint_authority", status=CheckStatus.UNKNOWN,
                value=None, message=f"Error: {str(e)}",
                weight=CHECK_WEIGHTS["mint_authority"]
            )

    async def check_freeze_authority(self, mint: str) -> SafetyCheck:
        """Check if freeze authority is revoked."""
        try:
            cached = await self.cache.get("freeze_authority", mint)
            if cached is not None:
                return cached

            mint_info = await self.rpc.get_mint_info(mint)
            if mint_info is None:
                return SafetyCheck(
                    name="freeze_authority", status=CheckStatus.UNKNOWN,
                    value=None, message="Could not fetch mint info",
                    weight=CHECK_WEIGHTS["freeze_authority"]
                )

            freeze_authority = mint_info.get("freezeAuthority")
            if freeze_authority is None:
                result = SafetyCheck(
                    name="freeze_authority", status=CheckStatus.PASSED,
                    value=None, message="Freeze authority is revoked",
                    weight=CHECK_WEIGHTS["freeze_authority"],
                    details={"revoked": True}
                )
            else:
                result = SafetyCheck(
                    name="freeze_authority", status=CheckStatus.WARNING,
                    value=freeze_authority,
                    message=f"Freeze authority active: {freeze_authority[:8]}...",
                    weight=CHECK_WEIGHTS["freeze_authority"],
                    details={"revoked": False, "authority": freeze_authority}
                )

            await self.cache.set("freeze_authority", mint, result, ttl=300)
            return result
        except Exception as e:
            logger.error(f"Error checking freeze authority: {e}")
            return SafetyCheck(
                name="freeze_authority", status=CheckStatus.UNKNOWN,
                value=None, message=f"Error: {str(e)}",
                weight=CHECK_WEIGHTS["freeze_authority"]
            )

    async def check_lp_lock(self, mint: str) -> SafetyCheck:
        """Check if liquidity pool is locked."""
        try:
            cached = await self.cache.get("lp_lock", mint)
            if cached is not None:
                return cached

            rugcheck_report = await self.rugcheck.get_token_report(mint)
            if rugcheck_report:
                lp_locked = rugcheck_report.get("lpLocked", False)
                lock_percentage = rugcheck_report.get("lpLockedPct", 0)

                if lp_locked and lock_percentage >= 80:
                    result = SafetyCheck(
                        name="lp_locked", status=CheckStatus.PASSED,
                        value=lock_percentage,
                        message=f"LP {lock_percentage:.1f}% locked",
                        weight=CHECK_WEIGHTS["lp_locked"],
                        details={"locked": True, "percentage": lock_percentage}
                    )
                elif lp_locked and lock_percentage >= 50:
                    result = SafetyCheck(
                        name="lp_locked", status=CheckStatus.WARNING,
                        value=lock_percentage,
                        message=f"LP only {lock_percentage:.1f}% locked",
                        weight=CHECK_WEIGHTS["lp_locked"],
                        details={"locked": True, "percentage": lock_percentage}
                    )
                else:
                    result = SafetyCheck(
                        name="lp_locked", status=CheckStatus.FAILED,
                        value=lock_percentage,
                        message=f"LP NOT LOCKED ({lock_percentage:.1f}%)",
                        weight=CHECK_WEIGHTS["lp_locked"],
                        details={"locked": False, "percentage": lock_percentage}
                    )
            else:
                result = SafetyCheck(
                    name="lp_locked", status=CheckStatus.UNKNOWN,
                    value=None, message="Could not verify LP lock status",
                    weight=CHECK_WEIGHTS["lp_locked"]
                )

            await self.cache.set("lp_lock", mint, result, ttl=120)
            return result
        except Exception as e:
            logger.error(f"Error checking LP lock: {e}")
            return SafetyCheck(
                name="lp_locked", status=CheckStatus.UNKNOWN,
                value=None, message=f"Error: {str(e)}",
                weight=CHECK_WEIGHTS["lp_locked"]
            )

    async def check_holder_concentration(self, mint: str) -> SafetyCheck:
        """Check top holder concentration."""
        try:
            cached = await self.cache.get("holder_concentration", mint)
            if cached is not None:
                return cached

            top_holders = await self.rpc.get_token_largest_accounts(mint)
            if not top_holders:
                return SafetyCheck(
                    name="holder_concentration", status=CheckStatus.UNKNOWN,
                    value=None, message="Could not fetch holder data",
                    weight=CHECK_WEIGHTS["holder_concentration"]
                )

            supply_info = await self.rpc.get_token_supply(mint)
            if not supply_info:
                return SafetyCheck(
                    name="holder_concentration", status=CheckStatus.UNKNOWN,
                    value=None, message="Could not fetch supply data",
                    weight=CHECK_WEIGHTS["holder_concentration"]
                )

            total_supply = float(supply_info.get("value", {}).get("uiAmount", 0))
            if total_supply == 0:
                return SafetyCheck(
                    name="holder_concentration", status=CheckStatus.UNKNOWN,
                    value=None, message="Zero supply detected",
                    weight=CHECK_WEIGHTS["holder_concentration"]
                )

            top_10_amount = sum(float(h.get("uiAmount", 0)) for h in top_holders[:10])
            concentration = (top_10_amount / total_supply) * 100

            if concentration <= self.max_holder_concentration:
                result = SafetyCheck(
                    name="holder_concentration", status=CheckStatus.PASSED,
                    value=concentration,
                    message=f"Top 10 hold {concentration:.1f}%",
                    weight=CHECK_WEIGHTS["holder_concentration"],
                    details={"top_10_percentage": concentration}
                )
            elif concentration <= 70:
                result = SafetyCheck(
                    name="holder_concentration", status=CheckStatus.WARNING,
                    value=concentration,
                    message=f"Top 10 hold {concentration:.1f}% (concentrated)",
                    weight=CHECK_WEIGHTS["holder_concentration"],
                    details={"top_10_percentage": concentration}
                )
            else:
                result = SafetyCheck(
                    name="holder_concentration", status=CheckStatus.FAILED,
                    value=concentration,
                    message=f"HIGH CONCENTRATION: Top 10 hold {concentration:.1f}%",
                    weight=CHECK_WEIGHTS["holder_concentration"],
                    details={"top_10_percentage": concentration}
                )

            await self.cache.set("holder_concentration", mint, result, ttl=60)
            return result
        except Exception as e:
            logger.error(f"Error checking holder concentration: {e}")
            return SafetyCheck(
                name="holder_concentration", status=CheckStatus.UNKNOWN,
                value=None, message=f"Error: {str(e)}",
                weight=CHECK_WEIGHTS["holder_concentration"]
            )


    async def check_creator_holdings(self, mint: str) -> SafetyCheck:
        """Check if creator still holds large percentage."""
        try:
            cached = await self.cache.get("creator_holdings", mint)
            if cached is not None:
                return cached

            creator_pct = 0.0
            creator_address = None

            if self.birdeye:
                creation_info = await self.birdeye.get_token_creation_info(mint)
                if creation_info:
                    creator_address = creation_info.get("creator")

            rugcheck_report = await self.rugcheck.get_token_report(mint)
            if rugcheck_report:
                creator_pct = rugcheck_report.get("creatorPct", 0)
                if not creator_address:
                    creator_address = rugcheck_report.get("creator")

            if creator_pct == 0 and not creator_address:
                return SafetyCheck(
                    name="creator_holdings", status=CheckStatus.UNKNOWN,
                    value=None, message="Could not determine creator holdings",
                    weight=CHECK_WEIGHTS["creator_holdings"]
                )

            if creator_pct <= 5:
                result = SafetyCheck(
                    name="creator_holdings", status=CheckStatus.PASSED,
                    value=creator_pct, message=f"Creator holds {creator_pct:.1f}%",
                    weight=CHECK_WEIGHTS["creator_holdings"],
                    details={"creator": creator_address, "percentage": creator_pct}
                )
            elif creator_pct <= 15:
                result = SafetyCheck(
                    name="creator_holdings", status=CheckStatus.WARNING,
                    value=creator_pct, message=f"Creator holds {creator_pct:.1f}%",
                    weight=CHECK_WEIGHTS["creator_holdings"],
                    details={"creator": creator_address, "percentage": creator_pct}
                )
            else:
                result = SafetyCheck(
                    name="creator_holdings", status=CheckStatus.FAILED,
                    value=creator_pct, message=f"Creator holds {creator_pct:.1f}%",
                    weight=CHECK_WEIGHTS["creator_holdings"],
                    details={"creator": creator_address, "percentage": creator_pct}
                )

            await self.cache.set("creator_holdings", mint, result, ttl=60)
            return result
        except Exception as e:
            logger.error(f"Error checking creator holdings: {e}")
            return SafetyCheck(
                name="creator_holdings", status=CheckStatus.UNKNOWN,
                value=None, message=f"Error: {str(e)}",
                weight=CHECK_WEIGHTS["creator_holdings"]
            )

    async def check_honeypot(self, mint: str) -> SafetyCheck:
        """Check if token is a honeypot (can buy but cannot sell)."""
        try:
            cached = await self.cache.get("honeypot", mint)
            if cached is not None:
                return cached

            is_honeypot = False
            honeypot_details = {}

            rugcheck_report = await self.rugcheck.get_token_report(mint)
            if rugcheck_report:
                risks = rugcheck_report.get("risks", [])
                honeypot_risks = [
                    r for r in risks
                    if "honeypot" in r.get("name", "").lower() or
                       "sell" in r.get("description", "").lower()
                ]
                is_honeypot = len(honeypot_risks) > 0
                honeypot_details["risks"] = honeypot_risks

            if self.birdeye and not is_honeypot:
                security = await self.birdeye.get_token_security(mint)
                if security:
                    is_honeypot = security.get("isHoneypot", False)
                    honeypot_details["birdeye"] = security

            if not is_honeypot:
                result = SafetyCheck(
                    name="honeypot", status=CheckStatus.PASSED,
                    value=False, message="No honeypot detected",
                    weight=CHECK_WEIGHTS["honeypot"],
                    details=honeypot_details
                )
            else:
                result = SafetyCheck(
                    name="honeypot", status=CheckStatus.FAILED,
                    value=True, message="HONEYPOT DETECTED - CANNOT SELL",
                    weight=CHECK_WEIGHTS["honeypot"],
                    details=honeypot_details
                )

            await self.cache.set("honeypot", mint, result, ttl=120)
            return result
        except Exception as e:
            logger.error(f"Error checking honeypot: {e}")
            return SafetyCheck(
                name="honeypot", status=CheckStatus.UNKNOWN,
                value=None, message=f"Error: {str(e)}",
                weight=CHECK_WEIGHTS["honeypot"]
            )

    async def check_liquidity_depth(self, mint: str) -> SafetyCheck:
        """Check liquidity depth. Low liquidity = high risk."""
        try:
            cached = await self.cache.get("liquidity_depth", mint)
            if cached is not None:
                return cached

            pairs = await self.dexscreener.get_token_pairs(mint)
            if not pairs:
                return SafetyCheck(
                    name="liquidity_depth", status=CheckStatus.FAILED,
                    value=0, message="NO LIQUIDITY POOLS FOUND",
                    weight=CHECK_WEIGHTS["liquidity_depth"]
                )

            total_liquidity = sum(
                float(p.get("liquidity", {}).get("usd", 0)) for p in pairs
            )

            if total_liquidity >= self.min_liquidity_usd:
                result = SafetyCheck(
                    name="liquidity_depth", status=CheckStatus.PASSED,
                    value=total_liquidity,
                    message=f"Liquidity: ${total_liquidity:,.0f}",
                    weight=CHECK_WEIGHTS["liquidity_depth"],
                    details={"usd": total_liquidity, "pools": len(pairs)}
                )
            elif total_liquidity >= self.min_liquidity_usd * 0.5:
                result = SafetyCheck(
                    name="liquidity_depth", status=CheckStatus.WARNING,
                    value=total_liquidity,
                    message=f"Low liquidity: ${total_liquidity:,.0f}",
                    weight=CHECK_WEIGHTS["liquidity_depth"],
                    details={"usd": total_liquidity, "pools": len(pairs)}
                )
            else:
                result = SafetyCheck(
                    name="liquidity_depth", status=CheckStatus.FAILED,
                    value=total_liquidity,
                    message=f"VERY LOW LIQUIDITY: ${total_liquidity:,.0f}",
                    weight=CHECK_WEIGHTS["liquidity_depth"],
                    details={"usd": total_liquidity, "pools": len(pairs)}
                )

            await self.cache.set("liquidity_depth", mint, result, ttl=30)
            return result
        except Exception as e:
            logger.error(f"Error checking liquidity: {e}")
            return SafetyCheck(
                name="liquidity_depth", status=CheckStatus.UNKNOWN,
                value=None, message=f"Error: {str(e)}",
                weight=CHECK_WEIGHTS["liquidity_depth"]
            )

    async def check_age(self, mint: str) -> SafetyCheck:
        """Check token age. Very new tokens = higher risk."""
        try:
            cached = await self.cache.get("token_age", mint)
            if cached is not None:
                return cached

            created_at = None

            pairs = await self.dexscreener.get_token_pairs(mint)
            if pairs:
                for pair in pairs:
                    pair_created = pair.get("pairCreatedAt")
                    if pair_created:
                        pair_time = datetime.fromtimestamp(pair_created / 1000)
                        if created_at is None or pair_time < created_at:
                            created_at = pair_time

            if self.birdeye and created_at is None:
                creation_info = await self.birdeye.get_token_creation_info(mint)
                if creation_info:
                    created_ts = creation_info.get("createdTime")
                    if created_ts:
                        created_at = datetime.fromtimestamp(created_ts)

            if created_at is None:
                return SafetyCheck(
                    name="token_age", status=CheckStatus.UNKNOWN,
                    value=None, message="Could not determine token age",
                    weight=CHECK_WEIGHTS["token_age"]
                )

            age_minutes = (datetime.utcnow() - created_at).total_seconds() / 60
            age_hours = age_minutes / 60
            age_days = age_hours / 24

            if age_minutes < self.min_token_age_minutes:
                result = SafetyCheck(
                    name="token_age", status=CheckStatus.FAILED,
                    value=age_minutes,
                    message=f"VERY NEW: {age_minutes:.0f} minutes old",
                    weight=CHECK_WEIGHTS["token_age"],
                    details={"age_minutes": age_minutes}
                )
            elif age_hours < 24:
                result = SafetyCheck(
                    name="token_age", status=CheckStatus.WARNING,
                    value=age_minutes,
                    message=f"New token: {age_hours:.1f} hours old",
                    weight=CHECK_WEIGHTS["token_age"],
                    details={"age_minutes": age_minutes}
                )
            else:
                result = SafetyCheck(
                    name="token_age", status=CheckStatus.PASSED,
                    value=age_minutes,
                    message=f"Token age: {age_days:.1f} days",
                    weight=CHECK_WEIGHTS["token_age"],
                    details={"age_minutes": age_minutes}
                )

            await self.cache.set("token_age", mint, result, ttl=60)
            return result
        except Exception as e:
            logger.error(f"Error checking token age: {e}")
            return SafetyCheck(
                name="token_age", status=CheckStatus.UNKNOWN,
                value=None, message=f"Error: {str(e)}",
                weight=CHECK_WEIGHTS["token_age"]
            )

    async def check_social_presence(self, mint: str) -> SafetyCheck:
        """Check for social media presence."""
        try:
            cached = await self.cache.get("social_presence", mint)
            if cached is not None:
                return cached

            socials = {"twitter": None, "website": None, "telegram": None, "discord": None}

            pairs = await self.dexscreener.get_token_pairs(mint)
            if pairs and len(pairs) > 0:
                info = pairs[0].get("info", {})
                if info.get("socials"):
                    for social in info["socials"]:
                        social_type = social.get("type", "").lower()
                        if social_type in socials:
                            socials[social_type] = social.get("url")
                websites = info.get("websites", [])
                if websites:
                    socials["website"] = websites[0].get("url")

            if self.birdeye:
                overview = await self.birdeye.get_token_overview(mint)
                if overview:
                    extensions = overview.get("extensions", {})
                    if not socials["twitter"]:
                        socials["twitter"] = extensions.get("twitter")
                    if not socials["website"]:
                        socials["website"] = extensions.get("website")

            valid_socials = sum(1 for v in socials.values() if v)

            if valid_socials >= 2:
                result = SafetyCheck(
                    name="social_presence", status=CheckStatus.PASSED,
                    value=valid_socials, message=f"Has {valid_socials} social links",
                    weight=CHECK_WEIGHTS["social_presence"],
                    details=socials
                )
            elif valid_socials == 1:
                result = SafetyCheck(
                    name="social_presence", status=CheckStatus.WARNING,
                    value=valid_socials, message="Limited social presence (1 link)",
                    weight=CHECK_WEIGHTS["social_presence"],
                    details=socials
                )
            else:
                result = SafetyCheck(
                    name="social_presence", status=CheckStatus.WARNING,
                    value=0, message="No social presence found",
                    weight=CHECK_WEIGHTS["social_presence"],
                    details=socials
                )

            await self.cache.set("social_presence", mint, result, ttl=300)
            return result
        except Exception as e:
            logger.error(f"Error checking social presence: {e}")
            return SafetyCheck(
                name="social_presence", status=CheckStatus.UNKNOWN,
                value=None, message=f"Error: {str(e)}",
                weight=CHECK_WEIGHTS["social_presence"]
            )


    def calculate_risk_score(
        self,
        checks: Dict[str, SafetyCheck],
        has_critical_flags: bool = False
    ) -> Tuple[float, float]:
        """Calculate overall risk score from individual checks."""
        if has_critical_flags:
            return 0.0, 1.0

        total_weight = sum(CHECK_WEIGHTS.values())
        earned_score = 0.0
        checked_weight = 0.0

        for name, check in checks.items():
            if check.status not in [CheckStatus.UNKNOWN, CheckStatus.SKIPPED]:
                earned_score += check.score_contribution
                checked_weight += check.weight

        confidence = checked_weight / total_weight if total_weight > 0 else 0.0
        score = (earned_score / checked_weight) * 100 if checked_weight > 0 else 50.0

        return score, confidence

    def determine_risk_level(self, score: float, flags: List[str]) -> RiskLevel:
        """Determine risk level from score and flags."""
        if flags:
            return RiskLevel.SCAM
        elif score >= 80:
            return RiskLevel.SAFE
        elif score >= 50:
            return RiskLevel.CAUTION
        elif score >= 20:
            return RiskLevel.DANGER
        else:
            return RiskLevel.SCAM

    def generate_recommendation(
        self,
        risk_level: RiskLevel,
        flags: List[str],
        warnings: List[str]
    ) -> str:
        """Generate human-readable recommendation."""
        if risk_level == RiskLevel.SCAM:
            return "DO NOT BUY - High probability of scam/rug pull"
        elif risk_level == RiskLevel.DANGER:
            return "HIGH RISK - Only trade with extreme caution and small amounts"
        elif risk_level == RiskLevel.CAUTION:
            if len(warnings) > 3:
                return "MODERATE RISK - Several concerns, proceed carefully"
            return "EXERCISE CAUTION - Some risk factors present"
        else:
            if warnings:
                return "RELATIVELY SAFE - Minor concerns, standard precautions apply"
            return "APPEARS SAFE - Standard trading precautions apply"


    async def scan_token(self, mint: str, skip_cache: bool = False) -> TokenSafetyReport:
        """Perform comprehensive token safety scan."""
        start_time = datetime.utcnow()
        logger.info(f"Starting token scan: {mint}")


        if self.is_blacklisted(mint):
            return TokenSafetyReport(
                mint=mint, risk_score=0.0, risk_level=RiskLevel.SCAM,
                checks={}, flags=["confirmed_scam"], warnings=[],
                recommendation="TOKEN IS BLACKLISTED - Known scam",
                confidence=1.0, token_info={"blacklisted": True},
                scan_duration_ms=0, data_sources_used=["blacklist"]
            )

        if self.is_whitelisted(mint):
            return TokenSafetyReport(
                mint=mint, risk_score=100.0, risk_level=RiskLevel.SAFE,
                checks={}, flags=[], warnings=[],
                recommendation="TOKEN IS WHITELISTED - Known safe token",
                confidence=1.0, token_info={"whitelisted": True},
                scan_duration_ms=0, data_sources_used=["whitelist"]
            )


        if skip_cache:
            for category in CHECK_WEIGHTS.keys():
                await self.cache.invalidate(category, mint)


        check_tasks = {
            "mint_authority": self.check_mint_authority(mint),
            "freeze_authority": self.check_freeze_authority(mint),
            "lp_locked": self.check_lp_lock(mint),
            "holder_concentration": self.check_holder_concentration(mint),
            "creator_holdings": self.check_creator_holdings(mint),
            "honeypot": self.check_honeypot(mint),
            "liquidity_depth": self.check_liquidity_depth(mint),
            "token_age": self.check_age(mint),
            "social_presence": self.check_social_presence(mint),
        }

        results = await asyncio.gather(*check_tasks.values(), return_exceptions=True)

        checks: Dict[str, SafetyCheck] = {}
        for name, result in zip(check_tasks.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Check {name} failed: {result}")
                checks[name] = SafetyCheck(
                    name=name, status=CheckStatus.UNKNOWN,
                    value=None, message=f"Error: {str(result)}",
                    weight=CHECK_WEIGHTS.get(name, 0)
                )
            else:
                checks[name] = result


        flags: List[str] = []
        warnings: List[str] = []

        for name, check in checks.items():
            if check.is_critical_failure:
                flags.append(f"{name}: {check.message}")
            elif check.status == CheckStatus.FAILED:
                warnings.append(f"{name}: {check.message}")
            elif check.status == CheckStatus.WARNING:
                warnings.append(f"{name}: {check.message}")


        has_critical = (
            checks.get("mint_authority", SafetyCheck("", CheckStatus.UNKNOWN, None, "", 0)).status == CheckStatus.FAILED or
            checks.get("honeypot", SafetyCheck("", CheckStatus.UNKNOWN, None, "", 0)).status == CheckStatus.FAILED
        )


        risk_score, confidence = self.calculate_risk_score(checks, has_critical)
        risk_level = self.determine_risk_level(risk_score, flags)
        recommendation = self.generate_recommendation(risk_level, flags, warnings)


        token_info = {}
        try:
            pairs = await self.dexscreener.get_token_pairs(mint)
            if pairs and len(pairs) > 0:
                main_pair = pairs[0]
                token_info = {
                    "name": main_pair.get("baseToken", {}).get("name"),
                    "symbol": main_pair.get("baseToken", {}).get("symbol"),
                    "price_usd": main_pair.get("priceUsd"),
                    "price_change_24h": main_pair.get("priceChange", {}).get("h24"),
                    "volume_24h": main_pair.get("volume", {}).get("h24"),
                    "liquidity_usd": main_pair.get("liquidity", {}).get("usd"),
                    "fdv": main_pair.get("fdv"),
                    "market_cap": main_pair.get("marketCap"),
                }
        except Exception as e:
            logger.warning(f"Could not fetch token info: {e}")

        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        data_sources = ["solana_rpc", "dexscreener", "rugcheck"]
        if self.birdeye:
            data_sources.append("birdeye")
        if self.helius:
            data_sources.append("helius")

        report = TokenSafetyReport(
            mint=mint, risk_score=risk_score, risk_level=risk_level,
            checks=checks, flags=flags, warnings=warnings,
            recommendation=recommendation, confidence=confidence,
            token_info=token_info, timestamp=end_time,
            scan_duration_ms=duration_ms, data_sources_used=data_sources
        )

        logger.info(f"Scan complete: {mint} - Score: {risk_score:.1f}, Level: {risk_level.value}")
        return report

    async def quick_scan(self, mint: str) -> Tuple[bool, str]:
        """Quick safety check - returns pass/fail and reason."""
        if self.is_blacklisted(mint):
            return False, "Token is blacklisted"
        if self.is_whitelisted(mint):
            return True, "Token is whitelisted"

        mint_check = await self.check_mint_authority(mint)
        if mint_check.status == CheckStatus.FAILED:
            return False, "Mint authority not revoked"

        honeypot_check = await self.check_honeypot(mint)
        if honeypot_check.status == CheckStatus.FAILED:
            return False, "Honeypot detected"

        liquidity_check = await self.check_liquidity_depth(mint)
        if liquidity_check.value and liquidity_check.value < 1000:
            return False, f"Very low liquidity: ${liquidity_check.value:,.0f}"

        return True, "Passed quick safety checks"


    async def monitor_token(
        self,
        mint: str,
        callback: Callable[[TokenAlert], None],
        interval_seconds: int = 30,
    ) -> str:
        """Start real-time monitoring for a token."""
        monitor_id = f"monitor_{mint}_{datetime.utcnow().timestamp()}"

        async def monitor_loop():
            last_state = {}

            while True:
                try:
                    pairs = await self.dexscreener.get_token_pairs(mint)

                    if pairs:
                        current_liquidity = sum(
                            float(p.get("liquidity", {}).get("usd", 0)) for p in pairs
                        )


                        if "liquidity" in last_state and last_state["liquidity"] > 0:
                            liq_change_pct = ((current_liquidity - last_state["liquidity"])
                                            / last_state["liquidity"] * 100)

                            if liq_change_pct < -20:
                                alert = TokenAlert(
                                    mint=mint,
                                    alert_type=AlertType.LP_REMOVAL,
                                    severity="HIGH" if liq_change_pct < -50 else "MEDIUM",
                                    message=f"Liquidity removed: {liq_change_pct:.1f}%",
                                    data={
                                        "previous": last_state["liquidity"],
                                        "current": current_liquidity,
                                        "change_pct": liq_change_pct,
                                    }
                                )
                                callback(alert)

                        last_state["liquidity"] = current_liquidity


                        if pairs[0].get("priceChange", {}).get("m5"):
                            price_change_5m = float(pairs[0]["priceChange"]["m5"])
                            if price_change_5m < -30:
                                alert = TokenAlert(
                                    mint=mint,
                                    alert_type=AlertType.PRICE_CRASH,
                                    severity="HIGH",
                                    message=f"Price crashed {price_change_5m:.1f}% in 5min",
                                    data={"price_change_5m": price_change_5m}
                                )
                                callback(alert)


                    mint_info = await self.rpc.get_mint_info(mint)
                    if mint_info:
                        current_mint_auth = mint_info.get("mintAuthority")

                        if "mint_authority" in last_state:
                            if last_state["mint_authority"] is None and current_mint_auth is not None:
                                alert = TokenAlert(
                                    mint=mint,
                                    alert_type=AlertType.AUTHORITY_CHANGE,
                                    severity="CRITICAL",
                                    message="Mint authority RE-ENABLED!",
                                    data={"new_authority": current_mint_auth}
                                )
                                callback(alert)

                        last_state["mint_authority"] = current_mint_auth

                    await asyncio.sleep(interval_seconds)

                except asyncio.CancelledError:
                    logger.info(f"Monitor {monitor_id} cancelled")
                    break
                except Exception as e:
                    logger.error(f"Monitor error: {e}")
                    await asyncio.sleep(interval_seconds)

        task = asyncio.create_task(monitor_loop())
        self._monitors[monitor_id] = task
        self._alert_callbacks[mint].append(callback)

        logger.info(f"Started monitoring {mint} with ID {monitor_id}")
        return monitor_id

    def stop_monitor(self, monitor_id: str) -> bool:
        """Stop a specific monitor."""
        if monitor_id in self._monitors:
            self._monitors[monitor_id].cancel()
            del self._monitors[monitor_id]
            logger.info(f"Stopped monitor {monitor_id}")
            return True
        return False

    def stop_all_monitors(self):
        """Stop all active monitors."""
        for monitor_id, task in list(self._monitors.items()):
            task.cancel()
        self._monitors.clear()
        self._alert_callbacks.clear()
        logger.info("Stopped all monitors")


    async def batch_scan(
        self,
        mints: List[str],
        max_concurrent: int = 5
    ) -> Dict[str, TokenSafetyReport]:
        """Scan multiple tokens with concurrency limit."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def scan_with_limit(mint: str) -> Tuple[str, TokenSafetyReport]:
            async with semaphore:
                report = await self.scan_token(mint)
                return mint, report

        tasks = [scan_with_limit(mint) for mint in mints]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        reports = {}
        for result in results:
            if isinstance(result, tuple):
                mint, report = result
                reports[mint] = report
            else:
                logger.error(f"Batch scan error: {result}")

        return reports

    async def get_token_summary(self, mint: str) -> Dict[str, Any]:
        """Get quick token summary without full scan."""
        summary = {
            "mint": mint,
            "whitelisted": self.is_whitelisted(mint),
            "blacklisted": self.is_blacklisted(mint),
        }

        pairs = await self.dexscreener.get_token_pairs(mint)
        if pairs and len(pairs) > 0:
            main_pair = pairs[0]
            summary.update({
                "name": main_pair.get("baseToken", {}).get("name"),
                "symbol": main_pair.get("baseToken", {}).get("symbol"),
                "price_usd": main_pair.get("priceUsd"),
                "liquidity_usd": main_pair.get("liquidity", {}).get("usd"),
                "volume_24h": main_pair.get("volume", {}).get("h24"),
            })

        return summary


def create_token_scanner(
    rpc_endpoint: str = "https://api.mainnet-beta.solana.com",
    birdeye_api_key: Optional[str] = None,
    helius_api_key: Optional[str] = None,
    **kwargs
) -> TokenScanner:
    """Factory function to create TokenScanner."""
    return TokenScanner(
        rpc_endpoint=rpc_endpoint,
        birdeye_api_key=birdeye_api_key,
        helius_api_key=helius_api_key,
        **kwargs
    )


async def main():
    """Example usage of TokenScanner."""
    import os

    scanner = create_token_scanner(
        rpc_endpoint=os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"),
        birdeye_api_key=os.getenv("BIRDEYE_API_KEY"),
        helius_api_key=os.getenv("HELIUS_API_KEY"),
    )

    try:

        bonk_mint = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"

        print("Scanning token...")
        report = await scanner.scan_token(bonk_mint)
        print(f"Risk Score: {report.risk_score:.1f}/100")
        print(f"Risk Level: {report.risk_level.value}")
        print(f"Recommendation: {report.recommendation}")
        print(f"Scan Duration: {report.scan_duration_ms:.0f}ms")

    finally:
        await scanner.close()


if __name__ == "__main__":
    asyncio.run(main())
