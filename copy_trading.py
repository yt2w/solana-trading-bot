"""
Copy Trading Engine - Track successful wallets and mirror their trades.

Production-grade copy trading system with real-time monitoring,
configurable parameters, and comprehensive risk controls.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Set, Callable, Tuple
from enum import Enum
import json
import aiohttp
from decimal import Decimal

logger = logging.getLogger(__name__)


class TradeType(Enum):
    """Type of trade."""
    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"


class CopyStatus(Enum):
    """Status of a copy operation."""
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


@dataclass
class PerformanceStats:
    """Performance statistics for a tracked wallet."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl_sol: Decimal = Decimal("0")
    total_volume_sol: Decimal = Decimal("0")
    average_trade_size: Decimal = Decimal("0")
    best_trade_pnl: Decimal = Decimal("0")
    worst_trade_pnl: Decimal = Decimal("0")
    win_rate: float = 0.0
    average_hold_time: timedelta = field(default_factory=lambda: timedelta(0))
    last_trade_time: Optional[datetime] = None
    tokens_traded: Set[str] = field(default_factory=set)
    
    def update_stats(self, trade_pnl: Decimal, trade_size: Decimal, is_win: bool):
        """Update stats with new trade."""
        self.total_trades += 1
        self.total_volume_sol += trade_size
        
        if is_win:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
            
        self.total_pnl_sol += trade_pnl
        self.average_trade_size = self.total_volume_sol / self.total_trades
        
        if trade_pnl > self.best_trade_pnl:
            self.best_trade_pnl = trade_pnl
        if trade_pnl < self.worst_trade_pnl:
            self.worst_trade_pnl = trade_pnl
            
        self.win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        self.last_trade_time = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl_sol": str(self.total_pnl_sol),
            "total_volume_sol": str(self.total_volume_sol),
            "average_trade_size": str(self.average_trade_size),
            "best_trade_pnl": str(self.best_trade_pnl),
            "worst_trade_pnl": str(self.worst_trade_pnl),
            "win_rate": self.win_rate,
            "tokens_traded": list(self.tokens_traded),
            "last_trade_time": self.last_trade_time.isoformat() if self.last_trade_time else None
        }


@dataclass
class TrackedWallet:
    """A wallet being tracked for copy trading."""
    address: str
    nickname: str
    user_id: str
    track_since: datetime = field(default_factory=datetime.utcnow)
    performance_stats: PerformanceStats = field(default_factory=PerformanceStats)
    copy_enabled: bool = False
    is_active: bool = True
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    total_copies_made: int = 0
    successful_copies: int = 0
    failed_copies: int = 0
    total_copy_volume: Decimal = Decimal("0")
    copy_pnl: Decimal = Decimal("0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "address": self.address,
            "nickname": self.nickname,
            "user_id": self.user_id,
            "track_since": self.track_since.isoformat(),
            "performance_stats": self.performance_stats.to_dict(),
            "copy_enabled": self.copy_enabled,
            "is_active": self.is_active,
            "tags": self.tags,
            "notes": self.notes,
            "total_copies_made": self.total_copies_made,
            "successful_copies": self.successful_copies,
            "failed_copies": self.failed_copies,
            "total_copy_volume": str(self.total_copy_volume),
            "copy_pnl": str(self.copy_pnl)
        }


@dataclass
class CopyConfig:
    """Configuration for copy trading from a specific wallet."""
    source_wallet: str
    user_id: str
    copy_percentage: float = 100.0
    max_copy_amount: Decimal = Decimal("1.0")
    min_trade_size: Decimal = Decimal("0.01")
    delay_seconds: float = 2.0
    copy_buys: bool = True
    copy_sells: bool = True
    token_whitelist: Set[str] = field(default_factory=set)
    token_blacklist: Set[str] = field(default_factory=set)
    require_safety_check: bool = True
    max_copies_per_day: int = 50
    max_exposure_per_token: Decimal = Decimal("5.0")
    slippage_bps: int = 300
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    copies_today: int = 0
    last_copy_date: Optional[datetime] = None
    
    def should_copy(self, trade_type: 'TradeType', token_mint: str, trade_size: Decimal) -> Tuple[bool, str]:
        """Determine if a trade should be copied."""
        if not self.is_active:
            return False, "Copy config is not active"
        if trade_type == TradeType.BUY and not self.copy_buys:
            return False, "Buy copying disabled"
        if trade_type == TradeType.SELL and not self.copy_sells:
            return False, "Sell copying disabled"
        if trade_size < self.min_trade_size:
            return False, f"Trade too small: {trade_size} < {self.min_trade_size}"
        if self.token_whitelist and token_mint not in self.token_whitelist:
            return False, "Token not in whitelist"
        if token_mint in self.token_blacklist:
            return False, "Token is blacklisted"
        if self._is_new_day():
            self.copies_today = 0
        if self.copies_today >= self.max_copies_per_day:
            return False, f"Daily copy limit reached: {self.max_copies_per_day}"
        return True, "OK"
    
    def _is_new_day(self) -> bool:
        """Check if it's a new day since last copy."""
        if not self.last_copy_date:
            return True
        return datetime.utcnow().date() > self.last_copy_date.date()
    
    def calculate_copy_amount(self, original_amount: Decimal) -> Decimal:
        """Calculate the amount to copy."""
        copy_amount = original_amount * Decimal(str(self.copy_percentage / 100))
        return min(copy_amount, self.max_copy_amount)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_wallet": self.source_wallet,
            "user_id": self.user_id,
            "copy_percentage": self.copy_percentage,
            "max_copy_amount": str(self.max_copy_amount),
            "min_trade_size": str(self.min_trade_size),
            "delay_seconds": self.delay_seconds,
            "copy_buys": self.copy_buys,
            "copy_sells": self.copy_sells,
            "token_whitelist": list(self.token_whitelist),
            "token_blacklist": list(self.token_blacklist),
            "require_safety_check": self.require_safety_check,
            "max_copies_per_day": self.max_copies_per_day,
            "max_exposure_per_token": str(self.max_exposure_per_token),
            "slippage_bps": self.slippage_bps,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "copies_today": self.copies_today
        }


@dataclass
class DetectedTrade:
    """A trade detected from on-chain activity."""
    signature: str
    wallet: str
    trade_type: 'TradeType'
    token_mint: str
    token_symbol: Optional[str]
    amount_sol: Decimal
    amount_token: Decimal
    price_per_token: Decimal
    timestamp: datetime
    program_id: str
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signature": self.signature,
            "wallet": self.wallet,
            "trade_type": self.trade_type.value,
            "token_mint": self.token_mint,
            "token_symbol": self.token_symbol,
            "amount_sol": str(self.amount_sol),
            "amount_token": str(self.amount_token),
            "price_per_token": str(self.price_per_token),
            "timestamp": self.timestamp.isoformat(),
            "program_id": self.program_id
        }


@dataclass
class CopyResult:
    """Result of a copy trade execution."""
    original_trade: 'DetectedTrade'
    copy_config: 'CopyConfig'
    status: 'CopyStatus'
    copy_signature: Optional[str] = None
    copy_amount: Decimal = Decimal("0")
    executed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    slippage_actual: Optional[float] = None
    execution_time_ms: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_signature": self.original_trade.signature,
            "source_wallet": self.original_trade.wallet,
            "status": self.status.value,
            "copy_signature": self.copy_signature,
            "copy_amount": str(self.copy_amount),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "error_message": self.error_message,
            "slippage_actual": self.slippage_actual,
            "execution_time_ms": self.execution_time_ms
        }


class TransactionParser:
    """Parse Solana transactions to extract trade information."""
    
    JUPITER_V6 = "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4"
    RAYDIUM_V4 = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
    ORCA_WHIRLPOOL = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc"
    PUMP_FUN = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
    WSOL_MINT = "So11111111111111111111111111111111111111112"
    
    @classmethod
    def parse_transaction(cls, tx_data: Dict[str, Any]) -> Optional['DetectedTrade']:
        """Parse a transaction to extract trade information."""
        try:
            signature = tx_data.get("signature", "")
            account_keys = tx_data.get("transaction", {}).get("message", {}).get("accountKeys", [])
            signer = account_keys[0] if account_keys else None
            if not signer:
                return None
            program_id = cls._find_dex_program(account_keys, tx_data)
            if not program_id:
                return None
            return cls._parse_swap(tx_data, signer, signature, program_id)
        except Exception as e:
            logger.debug(f"Failed to parse transaction: {e}")
            return None
    
    @classmethod
    def _find_dex_program(cls, account_keys: List[str], tx_data: Dict) -> Optional[str]:
        """Find which DEX program was used."""
        dex_programs = [cls.JUPITER_V6, cls.RAYDIUM_V4, cls.ORCA_WHIRLPOOL, cls.PUMP_FUN]
        for program in dex_programs:
            if program in account_keys:
                return program
        return None
    
    @classmethod
    def _parse_swap(cls, tx_data: Dict, signer: str, signature: str, program_id: str) -> Optional['DetectedTrade']:
        """Parse swap transaction from any DEX."""
        try:
            pre_balances = tx_data.get("meta", {}).get("preTokenBalances", [])
            post_balances = tx_data.get("meta", {}).get("postTokenBalances", [])
            
            pre_sol = tx_data.get("meta", {}).get("preBalances", [0])[0]
            post_sol = tx_data.get("meta", {}).get("postBalances", [0])[0]
            sol_change = Decimal(str((post_sol - pre_sol) / 1e9))
            
            token_mint = None
            token_change = Decimal("0")
            
            for post in post_balances:
                if post.get("owner") == signer:
                    mint = post.get("mint")
                    if mint and mint != cls.WSOL_MINT:
                        post_amount = Decimal(post.get("uiTokenAmount", {}).get("uiAmountString", "0"))
                        pre_amount = Decimal("0")
                        for pre in pre_balances:
                            if pre.get("owner") == signer and pre.get("mint") == mint:
                                pre_amount = Decimal(pre.get("uiTokenAmount", {}).get("uiAmountString", "0"))
                                break
                        change = post_amount - pre_amount
                        if abs(change) > abs(token_change):
                            token_change = change
                            token_mint = mint
            
            if not token_mint:
                return None
            
            if token_change > 0 and sol_change < 0:
                trade_type = TradeType.BUY
                amount_sol = abs(sol_change)
            elif token_change < 0 and sol_change > 0:
                trade_type = TradeType.SELL
                amount_sol = sol_change
            else:
                trade_type = TradeType.UNKNOWN
                amount_sol = abs(sol_change)
            
            price = amount_sol / abs(token_change) if token_change != 0 else Decimal("0")
            
            return DetectedTrade(
                signature=signature,
                wallet=signer,
                trade_type=trade_type,
                token_mint=token_mint,
                token_symbol=None,
                amount_sol=amount_sol,
                amount_token=abs(token_change),
                price_per_token=price,
                timestamp=datetime.utcnow(),
                program_id=program_id,
                raw_data=tx_data
            )
        except Exception as e:
            logger.debug(f"Failed to parse swap: {e}")
            return None


class WalletMonitor:
    """Real-time wallet monitoring via WebSocket."""
    
    def __init__(
        self,
        helius_api_key: str,
        on_trade_detected: Optional[Callable[['DetectedTrade'], None]] = None
    ):
        self.helius_api_key = helius_api_key
        self.on_trade_detected = on_trade_detected
        self.websocket_url = f"wss://mainnet.helius-rpc.com/?api-key={helius_api_key}"
        self._monitored_wallets: Set[str] = set()
        self._ws = None
        self._running = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._reconnect_delay = 5
        
    async def start(self):
        """Start monitoring."""
        self._running = True
        asyncio.create_task(self._connect())
        
    async def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
            
    async def add_wallet(self, address: str):
        """Add a wallet to monitor."""
        if address not in self._monitored_wallets:
            self._monitored_wallets.add(address)
            if self._ws:
                await self._subscribe_to_wallet(address)
                
    async def remove_wallet(self, address: str):
        """Remove a wallet from monitoring."""
        self._monitored_wallets.discard(address)
                
    async def _connect(self):
        """Connect to WebSocket with reconnection logic."""
        try:
            import websockets
        except ImportError:
            logger.warning("websockets not installed, using polling mode")
            return
            
        while self._running and self._reconnect_attempts < self._max_reconnect_attempts:
            try:
                logger.info("Connecting to Helius WebSocket...")
                self._ws = await websockets.connect(self.websocket_url)
                self._reconnect_attempts = 0
                
                for wallet in self._monitored_wallets:
                    await self._subscribe_to_wallet(wallet)
                
                await self._listen()
                
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self._reconnect_attempts += 1
                if self._running:
                    delay = self._reconnect_delay * (2 ** min(self._reconnect_attempts, 5))
                    logger.info(f"Reconnecting in {delay}s (attempt {self._reconnect_attempts})")
                    await asyncio.sleep(delay)
                    
    async def _subscribe_to_wallet(self, address: str):
        """Subscribe to wallet transactions."""
        if not self._ws:
            return
        subscribe_msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "transactionSubscribe",
            "params": [
                {"accountInclude": [address]},
                {"commitment": "confirmed", "encoding": "jsonParsed",
                 "transactionDetails": "full", "maxSupportedTransactionVersion": 0}
            ]
        }
        try:
            await self._ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to wallet: {address[:8]}...")
        except Exception as e:
            logger.error(f"Failed to subscribe to wallet {address}: {e}")
            
    async def _listen(self):
        """Listen for incoming messages."""
        if not self._ws:
            return
        async for message in self._ws:
            try:
                data = json.loads(message)
                if "result" in data:
                    logger.debug(f"Subscription confirmed: {data.get('result')}")
                    continue
                if "params" in data:
                    tx_data = data["params"].get("result", {}).get("transaction", {})
                    if tx_data:
                        trade = TransactionParser.parse_transaction(tx_data)
                        if trade and self.on_trade_detected:
                            asyncio.create_task(self._handle_trade(trade))
            except json.JSONDecodeError:
                logger.debug("Received non-JSON message")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
    async def _handle_trade(self, trade: 'DetectedTrade'):
        """Handle a detected trade."""
        try:
            if self.on_trade_detected:
                if asyncio.iscoroutinefunction(self.on_trade_detected):
                    await self.on_trade_detected(trade)
                else:
                    self.on_trade_detected(trade)
        except Exception as e:
            logger.error(f"Error in trade handler: {e}")


class CopyTradingEngine:
    """Main copy trading engine."""
    
    def __init__(
        self,
        helius_api_key: str,
        jupiter_api_url: str = "https://quote-api.jup.ag/v6",
        notification_callback: Optional[Callable] = None,
        safety_checker: Optional[Any] = None
    ):
        self.helius_api_key = helius_api_key
        self.jupiter_api_url = jupiter_api_url
        self.notification_callback = notification_callback
        self.safety_checker = safety_checker
        
        self._tracked_wallets: Dict[str, TrackedWallet] = {}
        self._copy_configs: Dict[str, CopyConfig] = {}
        self._copy_history: List[CopyResult] = []
        self._pending_copies: Dict[str, DetectedTrade] = {}
        self._scam_blacklist: Set[str] = set()
        self._monitor: Optional[WalletMonitor] = None
        self._copy_semaphore = asyncio.Semaphore(5)
        
        self._stats = {
            "total_copies": 0,
            "successful_copies": 0,
            "failed_copies": 0,
            "total_volume": Decimal("0"),
            "total_pnl": Decimal("0")
        }
        
    async def start(self):
        """Start the copy trading engine."""
        logger.info("Starting Copy Trading Engine...")
        self._monitor = WalletMonitor(
            helius_api_key=self.helius_api_key,
            on_trade_detected=self._on_trade_detected
        )
        for address in self._tracked_wallets.keys():
            await self._monitor.add_wallet(address)
        await self._monitor.start()
        
    async def stop(self):
        """Stop the copy trading engine."""
        logger.info("Stopping Copy Trading Engine...")
        if self._monitor:
            await self._monitor.stop()
            
    # ========== Wallet Tracking ==========
    
    async def track_wallet(
        self,
        address: str,
        nickname: str,
        user_id: str,
        tags: Optional[List[str]] = None,
        notes: str = ""
    ) -> TrackedWallet:
        """Start tracking a wallet."""
        if address in self._tracked_wallets:
            existing = self._tracked_wallets[address]
            logger.info(f"Already tracking wallet {address} as '{existing.nickname}'")
            return existing
        
        if address in self._scam_blacklist:
            raise ValueError(f"Wallet {address} is on the scam blacklist")
        
        wallet = TrackedWallet(
            address=address,
            nickname=nickname,
            user_id=user_id,
            tags=tags or [],
            notes=notes
        )
        
        self._tracked_wallets[address] = wallet
        
        if self._monitor:
            await self._monitor.add_wallet(address)
        
        logger.info(f"Now tracking wallet {address} as '{nickname}'")
        
        await self._notify({
            "type": "wallet_tracked",
            "wallet": wallet.to_dict()
        })
        
        return wallet
        
    async def untrack_wallet(self, address: str) -> bool:
        """Stop tracking a wallet."""
        if address not in self._tracked_wallets:
            return False
        
        wallet = self._tracked_wallets.pop(address)
        if address in self._copy_configs:
            del self._copy_configs[address]
        if self._monitor:
            await self._monitor.remove_wallet(address)
        
        logger.info(f"Stopped tracking wallet {address} ('{wallet.nickname}')")
        return True
        
    def get_tracked_wallets(self, user_id: Optional[str] = None) -> List[TrackedWallet]:
        """Get all tracked wallets, optionally filtered by user."""
        wallets = list(self._tracked_wallets.values())
        if user_id:
            wallets = [w for w in wallets if w.user_id == user_id]
        return wallets
        
    def get_wallet_performance(self, address: str) -> Optional[PerformanceStats]:
        """Get performance stats for a tracked wallet."""
        wallet = self._tracked_wallets.get(address)
        if wallet:
            return wallet.performance_stats
        return None

        
    # ========== Copy Configuration ==========
    
    async def set_copy_config(self, config: CopyConfig) -> CopyConfig:
        """Set copy configuration for a wallet."""
        if config.source_wallet not in self._tracked_wallets:
            raise ValueError(f"Wallet {config.source_wallet} is not being tracked")
        
        self._copy_configs[config.source_wallet] = config
        self._tracked_wallets[config.source_wallet].copy_enabled = True
        logger.info(f"Copy config set for wallet {config.source_wallet}")
        return config
        
    def get_copy_config(self, source_wallet: str) -> Optional[CopyConfig]:
        """Get copy configuration for a wallet."""
        return self._copy_configs.get(source_wallet)
        
    async def disable_copying(self, source_wallet: str) -> bool:
        """Disable copying for a wallet."""
        if source_wallet in self._copy_configs:
            self._copy_configs[source_wallet].is_active = False
        if source_wallet in self._tracked_wallets:
            self._tracked_wallets[source_wallet].copy_enabled = False
        return True
        
    async def enable_copying(self, source_wallet: str) -> bool:
        """Enable copying for a wallet."""
        if source_wallet not in self._copy_configs:
            return False
        self._copy_configs[source_wallet].is_active = True
        if source_wallet in self._tracked_wallets:
            self._tracked_wallets[source_wallet].copy_enabled = True
        return True
        
    # ========== Trade Detection & Copying ==========
    
    async def _on_trade_detected(self, trade: DetectedTrade):
        """Handle a detected trade from monitored wallet."""
        logger.info(f"Trade detected: {trade.trade_type.value} {trade.amount_sol} SOL "
                   f"from wallet {trade.wallet[:8]}...")
        
        if trade.wallet in self._tracked_wallets:
            wallet = self._tracked_wallets[trade.wallet]
            wallet.performance_stats.tokens_traded.add(trade.token_mint)
            wallet.performance_stats.last_trade_time = trade.timestamp
        
        await self._notify({
            "type": "trade_detected",
            "trade": trade.to_dict(),
            "wallet_nickname": self._tracked_wallets.get(trade.wallet, TrackedWallet("", "", "")).nickname
        })
        
        config = self._copy_configs.get(trade.wallet)
        if not config:
            return
        
        should_copy, reason = config.should_copy(trade.trade_type, trade.token_mint, trade.amount_sol)
        
        if not should_copy:
            logger.debug(f"Not copying trade: {reason}")
            await self._notify({
                "type": "copy_skipped",
                "trade": trade.to_dict(),
                "reason": reason
            })
            return
        
        asyncio.create_task(self._execute_delayed_copy(trade, config))
        
    async def _execute_delayed_copy(self, trade: DetectedTrade, config: CopyConfig):
        """Execute a copy trade after configured delay."""
        if config.delay_seconds > 0:
            logger.debug(f"Waiting {config.delay_seconds}s before copying...")
            await asyncio.sleep(config.delay_seconds)
        
        result = await self.execute_copy(trade, config)
        self._copy_history.append(result)
        
        self._stats["total_copies"] += 1
        if result.status == CopyStatus.SUCCESS:
            self._stats["successful_copies"] += 1
            self._stats["total_volume"] += result.copy_amount
        else:
            self._stats["failed_copies"] += 1
        
        if trade.wallet in self._tracked_wallets:
            wallet = self._tracked_wallets[trade.wallet]
            wallet.total_copies_made += 1
            if result.status == CopyStatus.SUCCESS:
                wallet.successful_copies += 1
                wallet.total_copy_volume += result.copy_amount
            else:
                wallet.failed_copies += 1

                
    async def execute_copy(self, trade: DetectedTrade, config: CopyConfig) -> CopyResult:
        """Execute a copy trade."""
        start_time = datetime.utcnow()
        
        try:
            async with self._copy_semaphore:
                copy_amount = config.calculate_copy_amount(trade.amount_sol)
                
                if config.require_safety_check and self.safety_checker:
                    try:
                        is_safe = await self._check_token_safety(trade.token_mint)
                        if not is_safe:
                            return CopyResult(
                                original_trade=trade,
                                copy_config=config,
                                status=CopyStatus.BLOCKED,
                                error_message="Token failed safety check"
                            )
                    except Exception as e:
                        logger.warning(f"Safety check failed: {e}")
                
                quote = await self._get_jupiter_quote(trade, copy_amount, config)
                if not quote:
                    return CopyResult(
                        original_trade=trade,
                        copy_config=config,
                        status=CopyStatus.FAILED,
                        error_message="Failed to get Jupiter quote"
                    )
                
                tx_signature = await self._execute_jupiter_swap(quote, config)
                
                end_time = datetime.utcnow()
                execution_time = int((end_time - start_time).total_seconds() * 1000)
                
                if tx_signature:
                    config.copies_today += 1
                    config.last_copy_date = datetime.utcnow()
                    
                    result = CopyResult(
                        original_trade=trade,
                        copy_config=config,
                        status=CopyStatus.SUCCESS,
                        copy_signature=tx_signature,
                        copy_amount=copy_amount,
                        executed_at=end_time,
                        execution_time_ms=execution_time
                    )
                    
                    await self._notify({
                        "type": "copy_executed",
                        "result": result.to_dict()
                    })
                    
                    return result
                else:
                    return CopyResult(
                        original_trade=trade,
                        copy_config=config,
                        status=CopyStatus.FAILED,
                        error_message="Swap execution failed",
                        execution_time_ms=execution_time
                    )
                    
        except Exception as e:
            logger.error(f"Error executing copy: {e}")
            return CopyResult(
                original_trade=trade,
                copy_config=config,
                status=CopyStatus.FAILED,
                error_message=str(e)
            )
            
    async def _check_token_safety(self, token_mint: str) -> bool:
        """Check if a token is safe to trade."""
        if not self.safety_checker:
            return True
        try:
            result = await self.safety_checker.scan_token(token_mint)
            return result.get("overall_safe", False)
        except Exception as e:
            logger.warning(f"Safety check error: {e}")
            return True
            
    async def _get_jupiter_quote(
        self,
        trade: DetectedTrade,
        amount: Decimal,
        config: CopyConfig
    ) -> Optional[Dict]:
        """Get a quote from Jupiter."""
        try:
            amount_lamports = int(amount * Decimal("1e9"))
            
            if trade.trade_type == TradeType.BUY:
                input_mint = "So11111111111111111111111111111111111111112"
                output_mint = trade.token_mint
            else:
                input_mint = trade.token_mint
                output_mint = "So11111111111111111111111111111111111111112"
            
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": amount_lamports,
                "slippageBps": config.slippage_bps
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.jupiter_api_url}/quote",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Jupiter quote error: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error getting Jupiter quote: {e}")
            return None
            
    async def _execute_jupiter_swap(self, quote: Dict, config: CopyConfig) -> Optional[str]:
        """Execute a swap via Jupiter."""
        logger.info("Would execute Jupiter swap (requires wallet signing)")
        return None

        
    # ========== Leaderboard ==========
    
    def get_leaderboard(
        self,
        period: str = "all",
        count: int = 10,
        sort_by: str = "pnl"
    ) -> List[Dict[str, Any]]:
        """Get top performing wallets."""
        wallets = list(self._tracked_wallets.values())
        
        if period != "all":
            now = datetime.utcnow()
            if period == "24h":
                cutoff = now - timedelta(hours=24)
            elif period == "7d":
                cutoff = now - timedelta(days=7)
            elif period == "30d":
                cutoff = now - timedelta(days=30)
            else:
                cutoff = datetime.min
                
            wallets = [w for w in wallets 
                      if w.performance_stats.last_trade_time and 
                      w.performance_stats.last_trade_time >= cutoff]
        
        if sort_by == "pnl":
            wallets.sort(key=lambda w: w.performance_stats.total_pnl_sol, reverse=True)
        elif sort_by == "win_rate":
            wallets.sort(key=lambda w: w.performance_stats.win_rate, reverse=True)
        elif sort_by == "volume":
            wallets.sort(key=lambda w: w.performance_stats.total_volume_sol, reverse=True)
        elif sort_by == "trades":
            wallets.sort(key=lambda w: w.performance_stats.total_trades, reverse=True)
        
        result = []
        for rank, wallet in enumerate(wallets[:count], 1):
            result.append({
                "rank": rank,
                "address": wallet.address,
                "nickname": wallet.nickname,
                "pnl_sol": str(wallet.performance_stats.total_pnl_sol),
                "win_rate": wallet.performance_stats.win_rate,
                "total_trades": wallet.performance_stats.total_trades,
                "volume_sol": str(wallet.performance_stats.total_volume_sol),
                "best_trade": str(wallet.performance_stats.best_trade_pnl),
                "copy_enabled": wallet.copy_enabled
            })
        
        return result
        
    def get_wallet_rank(self, address: str, sort_by: str = "pnl") -> Optional[Dict[str, Any]]:
        """Get rank for a specific wallet."""
        wallets = list(self._tracked_wallets.values())
        
        if sort_by == "pnl":
            wallets.sort(key=lambda w: w.performance_stats.total_pnl_sol, reverse=True)
        elif sort_by == "win_rate":
            wallets.sort(key=lambda w: w.performance_stats.win_rate, reverse=True)
        
        for rank, wallet in enumerate(wallets, 1):
            if wallet.address == address:
                return {
                    "rank": rank,
                    "total_wallets": len(wallets),
                    "percentile": ((len(wallets) - rank) / len(wallets)) * 100 if wallets else 0,
                    "wallet": wallet.to_dict()
                }
        
        return None
        
    # ========== Performance Analysis ==========
    
    def get_copy_performance(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get copy trading performance analysis."""
        history = self._copy_history
        
        if user_id:
            history = [h for h in history if h.copy_config.user_id == user_id]
        
        if not history:
            return {
                "total_copies": 0,
                "success_rate": 0,
                "total_volume": "0",
                "average_execution_time_ms": 0
            }
        
        successful = [h for h in history if h.status == CopyStatus.SUCCESS]
        failed = [h for h in history if h.status == CopyStatus.FAILED]
        
        total_volume = sum(h.copy_amount for h in successful)
        avg_execution_time = (
            sum(h.execution_time_ms or 0 for h in history) / len(history)
            if history else 0
        )
        
        return {
            "total_copies": len(history),
            "successful_copies": len(successful),
            "failed_copies": len(failed),
            "skipped_copies": len([h for h in history if h.status == CopyStatus.SKIPPED]),
            "blocked_copies": len([h for h in history if h.status == CopyStatus.BLOCKED]),
            "success_rate": (len(successful) / len(history) * 100) if history else 0,
            "total_volume": str(total_volume),
            "average_execution_time_ms": avg_execution_time,
            "copies_by_wallet": self._get_copies_by_wallet(history)
        }
        
    def _get_copies_by_wallet(self, history: List[CopyResult]) -> Dict[str, int]:
        """Get copy count by source wallet."""
        by_wallet: Dict[str, int] = {}
        for result in history:
            wallet = result.original_trade.wallet
            by_wallet[wallet] = by_wallet.get(wallet, 0) + 1
        return by_wallet
        
    def get_slippage_analysis(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze slippage impact on copy trades."""
        history = [h for h in self._copy_history if h.slippage_actual is not None]
        
        if user_id:
            history = [h for h in history if h.copy_config.user_id == user_id]
        
        if not history:
            return {"average_slippage": 0, "max_slippage": 0, "min_slippage": 0}
        
        slippages = [h.slippage_actual for h in history if h.slippage_actual is not None]
        
        return {
            "average_slippage": sum(slippages) / len(slippages),
            "max_slippage": max(slippages),
            "min_slippage": min(slippages),
            "trades_analyzed": len(history)
        }

        
    # ========== Safety & Blacklist ==========
    
    def add_to_blacklist(self, address: str):
        """Add a wallet to the scam blacklist."""
        self._scam_blacklist.add(address)
        logger.info(f"Added {address} to scam blacklist")
        
    def remove_from_blacklist(self, address: str):
        """Remove a wallet from the scam blacklist."""
        self._scam_blacklist.discard(address)
        
    def is_blacklisted(self, address: str) -> bool:
        """Check if a wallet is blacklisted."""
        return address in self._scam_blacklist
        
    def get_blacklist(self) -> List[str]:
        """Get all blacklisted wallets."""
        return list(self._scam_blacklist)
        
    # ========== Notifications ==========
    
    async def _notify(self, notification: Dict[str, Any]):
        """Send a notification."""
        if self.notification_callback:
            try:
                if asyncio.iscoroutinefunction(self.notification_callback):
                    await self.notification_callback(notification)
                else:
                    self.notification_callback(notification)
            except Exception as e:
                logger.error(f"Notification error: {e}")
                
    # ========== Statistics ==========
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall copy trading statistics."""
        return {
            "tracked_wallets": len(self._tracked_wallets),
            "active_copy_configs": len([c for c in self._copy_configs.values() if c.is_active]),
            "total_copies": self._stats["total_copies"],
            "successful_copies": self._stats["successful_copies"],
            "failed_copies": self._stats["failed_copies"],
            "success_rate": (
                self._stats["successful_copies"] / self._stats["total_copies"] * 100
                if self._stats["total_copies"] > 0 else 0
            ),
            "total_volume_sol": str(self._stats["total_volume"]),
            "blacklisted_wallets": len(self._scam_blacklist)
        }
        
    # ========== Persistence ==========
    
    def export_data(self) -> Dict[str, Any]:
        """Export all data for persistence."""
        return {
            "tracked_wallets": {
                addr: wallet.to_dict() 
                for addr, wallet in self._tracked_wallets.items()
            },
            "copy_configs": {
                addr: config.to_dict()
                for addr, config in self._copy_configs.items()
            },
            "blacklist": list(self._scam_blacklist),
            "stats": {
                "total_copies": self._stats["total_copies"],
                "successful_copies": self._stats["successful_copies"],
                "failed_copies": self._stats["failed_copies"],
                "total_volume": str(self._stats["total_volume"]),
                "total_pnl": str(self._stats["total_pnl"])
            }
        }
        
    def import_data(self, data: Dict[str, Any]):
        """Import data from persistence."""
        self._scam_blacklist = set(data.get("blacklist", []))
        if "stats" in data:
            stats = data["stats"]
            self._stats["total_copies"] = stats.get("total_copies", 0)
            self._stats["successful_copies"] = stats.get("successful_copies", 0)
            self._stats["failed_copies"] = stats.get("failed_copies", 0)
            self._stats["total_volume"] = Decimal(stats.get("total_volume", "0"))
            self._stats["total_pnl"] = Decimal(stats.get("total_pnl", "0"))
        logger.info("Data imported successfully")


def create_copy_trading_engine(
    helius_api_key: str,
    notification_callback: Optional[Callable] = None,
    safety_checker: Optional[Any] = None
) -> CopyTradingEngine:
    """Create and configure a copy trading engine."""
    return CopyTradingEngine(
        helius_api_key=helius_api_key,
        notification_callback=notification_callback,
        safety_checker=safety_checker
    )


if __name__ == "__main__":
    async def example():
        """Example usage of copy trading engine."""
        engine = CopyTradingEngine(helius_api_key="your-api-key")
        
        wallet = await engine.track_wallet(
            address="SomeWalletAddressHere123456789",
            nickname="Top Trader",
            user_id="user123",
            tags=["whale", "consistent"]
        )
        
        config = CopyConfig(
            source_wallet=wallet.address,
            user_id="user123",
            copy_percentage=50.0,
            max_copy_amount=Decimal("0.5"),
            min_trade_size=Decimal("0.1"),
            delay_seconds=3.0,
            require_safety_check=True
        )
        
        await engine.set_copy_config(config)
        await engine.start()
        
        leaderboard = engine.get_leaderboard(period="7d", count=10)
        print("Top 10 Traders (7 days):", leaderboard)
        
        stats = engine.get_stats()
        print("Copy Trading Stats:", stats)

    # asyncio.run(example())
