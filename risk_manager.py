"""
Risk Manager - Comprehensive Risk Management System
Production-grade risk management for Solana trading bot
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from enum import Enum, auto
from typing import Optional, Dict, List, Any, Callable, Set, Tuple
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class StopType(Enum):
    """Types of stop-loss orders"""
    FIXED = "fixed"
    TRAILING = "trailing"
    TIME_BASED = "time_based"
    BREAK_EVEN = "break_even"


class AlertType(Enum):
    """Risk alert types"""
    HIGH_EXPOSURE = auto()
    LOSING_STREAK = auto()
    TOKEN_SAFETY_CHANGE = auto()
    SLIPPAGE_WARNING = auto()
    DAILY_LOSS_WARNING = auto()
    CIRCUIT_BREAKER_TRIGGERED = auto()
    STOP_LOSS_TRIGGERED = auto()
    TAKE_PROFIT_TRIGGERED = auto()
    POSITION_LIMIT_WARNING = auto()
    UNUSUAL_ACTIVITY = auto()


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class TradeAction(Enum):
    """Trade action types"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class RiskSettings:
    """Per-user risk configuration"""

    max_position_sol: Decimal = Decimal("1.0")
    max_position_pct: Decimal = Decimal("10.0")
    max_total_exposure: Decimal = Decimal("10.0")
    max_positions: int = 10


    max_daily_loss_pct: Decimal = Decimal("10.0")
    max_daily_loss_sol: Decimal = Decimal("2.0")
    max_daily_trades: int = 50


    max_slippage_pct: Decimal = Decimal("5.0")
    default_slippage_pct: Decimal = Decimal("1.0")


    default_stop_loss_pct: Decimal = Decimal("15.0")
    default_trailing_stop_pct: Decimal = Decimal("10.0")
    auto_stop_loss: bool = True


    default_take_profit_pct: Decimal = Decimal("50.0")
    auto_take_profit: bool = False


    min_token_safety_score: int = 30
    require_liquidity_check: bool = True
    min_liquidity_sol: Decimal = Decimal("5.0")


    max_consecutive_losses: int = 5
    cooldown_minutes: int = 30


    is_premium: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "max_position_sol": str(self.max_position_sol),
            "max_position_pct": str(self.max_position_pct),
            "max_total_exposure": str(self.max_total_exposure),
            "max_positions": self.max_positions,
            "max_daily_loss_pct": str(self.max_daily_loss_pct),
            "max_daily_loss_sol": str(self.max_daily_loss_sol),
            "max_daily_trades": self.max_daily_trades,
            "max_slippage_pct": str(self.max_slippage_pct),
            "default_slippage_pct": str(self.default_slippage_pct),
            "default_stop_loss_pct": str(self.default_stop_loss_pct),
            "default_trailing_stop_pct": str(self.default_trailing_stop_pct),
            "auto_stop_loss": self.auto_stop_loss,
            "default_take_profit_pct": str(self.default_take_profit_pct),
            "auto_take_profit": self.auto_take_profit,
            "min_token_safety_score": self.min_token_safety_score,
            "require_liquidity_check": self.require_liquidity_check,
            "min_liquidity_sol": str(self.min_liquidity_sol),
            "max_consecutive_losses": self.max_consecutive_losses,
            "cooldown_minutes": self.cooldown_minutes,
            "is_premium": self.is_premium
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskSettings":
        """Create from dictionary"""
        return cls(
            max_position_sol=Decimal(str(data.get("max_position_sol", "1.0"))),
            max_position_pct=Decimal(str(data.get("max_position_pct", "10.0"))),
            max_total_exposure=Decimal(str(data.get("max_total_exposure", "10.0"))),
            max_positions=int(data.get("max_positions", 10)),
            max_daily_loss_pct=Decimal(str(data.get("max_daily_loss_pct", "10.0"))),
            max_daily_loss_sol=Decimal(str(data.get("max_daily_loss_sol", "2.0"))),
            max_daily_trades=int(data.get("max_daily_trades", 50)),
            max_slippage_pct=Decimal(str(data.get("max_slippage_pct", "5.0"))),
            default_slippage_pct=Decimal(str(data.get("default_slippage_pct", "1.0"))),
            default_stop_loss_pct=Decimal(str(data.get("default_stop_loss_pct", "15.0"))),
            default_trailing_stop_pct=Decimal(str(data.get("default_trailing_stop_pct", "10.0"))),
            auto_stop_loss=bool(data.get("auto_stop_loss", True)),
            default_take_profit_pct=Decimal(str(data.get("default_take_profit_pct", "50.0"))),
            auto_take_profit=bool(data.get("auto_take_profit", False)),
            min_token_safety_score=int(data.get("min_token_safety_score", 30)),
            require_liquidity_check=bool(data.get("require_liquidity_check", True)),
            min_liquidity_sol=Decimal(str(data.get("min_liquidity_sol", "5.0"))),
            max_consecutive_losses=int(data.get("max_consecutive_losses", 5)),
            cooldown_minutes=int(data.get("cooldown_minutes", 30)),
            is_premium=bool(data.get("is_premium", False))
        )

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate settings are within acceptable bounds"""
        errors = []
        if self.max_position_sol <= 0 or self.max_position_sol > Decimal("100"):
            errors.append("max_position_sol must be between 0 and 100 SOL")
        if self.max_position_pct <= 0 or self.max_position_pct > Decimal("100"):
            errors.append("max_position_pct must be between 0 and 100%")
        if self.max_total_exposure <= 0 or self.max_total_exposure > Decimal("1000"):
            errors.append("max_total_exposure must be between 0 and 1000 SOL")
        if self.max_positions <= 0 or self.max_positions > 100:
            errors.append("max_positions must be between 1 and 100")
        if self.max_daily_loss_pct <= 0 or self.max_daily_loss_pct > Decimal("100"):
            errors.append("max_daily_loss_pct must be between 0 and 100%")
        if self.max_slippage_pct <= 0 or self.max_slippage_pct > Decimal("50"):
            errors.append("max_slippage_pct must be between 0 and 50%")
        if self.default_slippage_pct > self.max_slippage_pct:
            errors.append("default_slippage_pct cannot exceed max_slippage_pct")
        if self.default_stop_loss_pct <= 0 or self.default_stop_loss_pct > Decimal("90"):
            errors.append("default_stop_loss_pct must be between 0 and 90%")
        if self.min_token_safety_score < 0 or self.min_token_safety_score > 100:
            errors.append("min_token_safety_score must be between 0 and 100")
        return len(errors) == 0, errors


@dataclass
class TakeProfitLevel:
    """Single take-profit level"""
    price_pct: Decimal
    sell_pct: Decimal
    triggered: bool = False
    triggered_at: Optional[datetime] = None


@dataclass
class StopLoss:
    """Stop-loss configuration"""
    stop_type: StopType
    trigger_pct: Decimal
    trailing_high: Optional[Decimal] = None
    time_limit: Optional[datetime] = None
    triggered: bool = False
    triggered_at: Optional[datetime] = None

    def update_trailing(self, current_price: Decimal, entry_price: Decimal) -> bool:
        """Update trailing stop high, returns True if updated"""
        if self.stop_type != StopType.TRAILING:
            return False

        if self.trailing_high is None or current_price > self.trailing_high:
            self.trailing_high = current_price
            return True
        return False

    def get_trigger_price(self, entry_price: Decimal) -> Decimal:
        """Calculate the price at which stop triggers"""
        if self.stop_type == StopType.TRAILING and self.trailing_high:
            return self.trailing_high * (1 - self.trigger_pct / 100)
        elif self.stop_type == StopType.BREAK_EVEN:
            return entry_price
        else:
            return entry_price * (1 - self.trigger_pct / 100)


@dataclass
class Position:
    """Trading position"""
    position_id: str
    user_id: int
    token_address: str
    token_symbol: str
    entry_price: Decimal
    amount: Decimal
    entry_value_sol: Decimal
    entry_time: datetime


    current_price: Optional[Decimal] = None
    current_value_sol: Optional[Decimal] = None


    stop_loss: Optional[StopLoss] = None
    take_profit_levels: List[TakeProfitLevel] = field(default_factory=list)


    is_open: bool = True
    closed_at: Optional[datetime] = None
    close_price: Optional[Decimal] = None
    close_reason: Optional[str] = None


    realized_pnl_sol: Decimal = Decimal("0")

    @property
    def unrealized_pnl_sol(self) -> Decimal:
        """Calculate unrealized P&L"""
        if self.current_value_sol is None:
            return Decimal("0")
        return self.current_value_sol - self.entry_value_sol

    @property
    def unrealized_pnl_pct(self) -> Decimal:
        """Calculate unrealized P&L percentage"""
        if self.entry_value_sol == 0:
            return Decimal("0")
        return (self.unrealized_pnl_sol / self.entry_value_sol) * 100

    @property
    def total_pnl_sol(self) -> Decimal:
        """Total P&L (realized + unrealized)"""
        return self.realized_pnl_sol + self.unrealized_pnl_sol

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "position_id": self.position_id,
            "user_id": self.user_id,
            "token_address": self.token_address,
            "token_symbol": self.token_symbol,
            "entry_price": str(self.entry_price),
            "amount": str(self.amount),
            "entry_value_sol": str(self.entry_value_sol),
            "entry_time": self.entry_time.isoformat(),
            "current_price": str(self.current_price) if self.current_price else None,
            "current_value_sol": str(self.current_value_sol) if self.current_value_sol else None,
            "is_open": self.is_open,
            "unrealized_pnl_sol": str(self.unrealized_pnl_sol),
            "unrealized_pnl_pct": str(self.unrealized_pnl_pct),
            "realized_pnl_sol": str(self.realized_pnl_sol),
            "total_pnl_sol": str(self.total_pnl_sol)
        }


@dataclass
class TradeValidationResult:
    """Result of trade validation"""
    approved: bool
    reason: str
    risk_level: RiskLevel = RiskLevel.LOW
    warnings: List[str] = field(default_factory=list)
    adjusted_params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "reason": self.reason,
            "risk_level": self.risk_level.value,
            "warnings": self.warnings,
            "adjusted_params": self.adjusted_params
        }


@dataclass
class RiskAlert:
    """Risk alert notification"""
    alert_id: str
    alert_type: AlertType
    user_id: int
    severity: RiskLevel
    message: str
    data: Dict[str, Any]
    timestamp: datetime
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.name,
            "user_id": self.user_id,
            "severity": self.severity.value,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged
        }


@dataclass
class DailyStats:
    """Daily trading statistics"""
    date: str
    user_id: int
    trades_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl_sol: Decimal = Decimal("0")
    starting_balance: Decimal = Decimal("0")
    current_balance: Decimal = Decimal("0")
    max_drawdown_pct: Decimal = Decimal("0")
    consecutive_losses: int = 0
    trading_halted: bool = False
    halt_reason: Optional[str] = None

    @property
    def win_rate(self) -> Decimal:
        """Calculate win rate"""
        total = self.winning_trades + self.losing_trades
        if total == 0:
            return Decimal("0")
        return Decimal(self.winning_trades) / Decimal(total) * 100

    @property
    def daily_loss_pct(self) -> Decimal:
        """Calculate daily loss percentage"""
        if self.starting_balance == 0:
            return Decimal("0")
        return (self.total_pnl_sol / self.starting_balance) * 100


class CircuitBreaker:
    """
    Circuit breaker for automatic trading halt

    States:
    - CLOSED: Normal operation, trading allowed
    - OPEN: Trading halted due to trigger condition
    - HALF_OPEN: Testing phase, limited trading allowed
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_max_trades: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = timedelta(minutes=recovery_timeout)
        self.half_open_max_trades = half_open_max_trades

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.opened_at: Optional[datetime] = None
        self.half_open_trades = 0
        self.trigger_reason: Optional[str] = None
        self._lock = asyncio.Lock()

    async def record_success(self):
        """Record a successful trade"""
        async with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.half_open_trades += 1
                if self.half_open_trades >= self.half_open_max_trades:
                    self._close()
            else:
                self.failure_count = 0

    async def record_failure(self, reason: str = "Trade failed"):
        """Record a failed trade"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now(timezone.utc)

            if self.state == CircuitBreakerState.HALF_OPEN:
                self._open(reason)
            elif self.failure_count >= self.failure_threshold:
                self._open(reason)

    async def force_open(self, reason: str):
        """Force the circuit breaker open"""
        async with self._lock:
            self._open(reason)

    async def force_close(self):
        """Force the circuit breaker closed (manual override)"""
        async with self._lock:
            self._close()

    async def can_execute(self) -> Tuple[bool, str]:
        """Check if trading is allowed"""
        async with self._lock:
            if self.state == CircuitBreakerState.CLOSED:
                return True, "Trading allowed"

            if self.state == CircuitBreakerState.OPEN:
                if self.opened_at and datetime.now(timezone.utc) - self.opened_at >= self.recovery_timeout:
                    self._half_open()
                    return True, "Circuit breaker in testing mode"

                remaining = self.recovery_timeout - (datetime.now(timezone.utc) - self.opened_at) if self.opened_at else self.recovery_timeout
                return False, f"Trading halted: {self.trigger_reason}. Resumes in {int(remaining.total_seconds() / 60)} minutes"

            if self.state == CircuitBreakerState.HALF_OPEN:
                return True, "Circuit breaker in testing mode - limited trading"

            return False, "Unknown circuit breaker state"

    def _open(self, reason: str):
        """Open the circuit breaker (halt trading)"""
        self.state = CircuitBreakerState.OPEN
        self.opened_at = datetime.now(timezone.utc)
        self.trigger_reason = reason
        self.half_open_trades = 0
        logger.warning(f"Circuit breaker OPENED: {reason}")

    def _half_open(self):
        """Transition to half-open state"""
        self.state = CircuitBreakerState.HALF_OPEN
        self.half_open_trades = 0
        logger.info("Circuit breaker entering HALF-OPEN state")

    def _close(self):
        """Close the circuit breaker (resume trading)"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.opened_at = None
        self.trigger_reason = None
        self.half_open_trades = 0
        logger.info("Circuit breaker CLOSED - trading resumed")

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "trigger_reason": self.trigger_reason,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "recovery_timeout_minutes": self.recovery_timeout.total_seconds() / 60
        }


class PositionManager:
    """Manages trading positions with CRUD operations"""

    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.user_positions: Dict[int, Set[str]] = defaultdict(set)
        self.token_positions: Dict[str, Set[str]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def create_position(
        self,
        user_id: int,
        token_address: str,
        token_symbol: str,
        entry_price: Decimal,
        amount: Decimal,
        entry_value_sol: Decimal,
        stop_loss: Optional[StopLoss] = None,
        take_profit_levels: Optional[List[TakeProfitLevel]] = None
    ) -> Position:
        """Create a new position"""
        async with self._lock:
            position_id = str(uuid.uuid4())[:8]

            position = Position(
                position_id=position_id,
                user_id=user_id,
                token_address=token_address,
                token_symbol=token_symbol,
                entry_price=entry_price,
                amount=amount,
                entry_value_sol=entry_value_sol,
                entry_time=datetime.now(timezone.utc),
                current_price=entry_price,
                current_value_sol=entry_value_sol,
                stop_loss=stop_loss,
                take_profit_levels=take_profit_levels or []
            )

            self.positions[position_id] = position
            self.user_positions[user_id].add(position_id)
            self.token_positions[token_address].add(position_id)

            logger.info(f"Created position {position_id} for user {user_id}: {amount} {token_symbol}")
            return position

    async def get_position(self, position_id: str) -> Optional[Position]:
        """Get a position by ID"""
        return self.positions.get(position_id)

    async def get_user_positions(self, user_id: int, open_only: bool = True) -> List[Position]:
        """Get all positions for a user"""
        position_ids = self.user_positions.get(user_id, set())
        positions = [self.positions[pid] for pid in position_ids if pid in self.positions]

        if open_only:
            positions = [p for p in positions if p.is_open]

        return positions

    async def get_token_positions(self, token_address: str, user_id: Optional[int] = None) -> List[Position]:
        """Get all positions for a token"""
        position_ids = self.token_positions.get(token_address, set())
        positions = [self.positions[pid] for pid in position_ids if pid in self.positions]

        if user_id is not None:
            positions = [p for p in positions if p.user_id == user_id]

        return [p for p in positions if p.is_open]

    async def update_position_price(
        self,
        position_id: str,
        current_price: Decimal,
        current_value_sol: Decimal
    ) -> Optional[Position]:
        """Update position with current price"""
        async with self._lock:
            position = self.positions.get(position_id)
            if not position or not position.is_open:
                return None

            position.current_price = current_price
            position.current_value_sol = current_value_sol

            if position.stop_loss and position.stop_loss.stop_type == StopType.TRAILING:
                position.stop_loss.update_trailing(current_price, position.entry_price)

            return position

    async def close_position(
        self,
        position_id: str,
        close_price: Decimal,
        close_reason: str,
        realized_pnl: Optional[Decimal] = None
    ) -> Optional[Position]:
        """Close a position"""
        async with self._lock:
            position = self.positions.get(position_id)
            if not position:
                return None

            position.is_open = False
            position.closed_at = datetime.now(timezone.utc)
            position.close_price = close_price
            position.close_reason = close_reason

            if realized_pnl is not None:
                position.realized_pnl_sol = realized_pnl
            else:
                close_value = position.amount * close_price
                position.realized_pnl_sol = close_value - position.entry_value_sol

            logger.info(f"Closed position {position_id}: {close_reason}, PnL: {position.realized_pnl_sol} SOL")
            return position

    async def partial_close(
        self,
        position_id: str,
        close_amount: Decimal,
        close_price: Decimal,
        reason: str
    ) -> Tuple[Optional[Position], Decimal]:
        """Partially close a position, returns position and realized PnL"""
        async with self._lock:
            position = self.positions.get(position_id)
            if not position or not position.is_open:
                return None, Decimal("0")

            if close_amount >= position.amount:
                return await self.close_position(position_id, close_price, reason), position.total_pnl_sol

            close_pct = close_amount / position.amount
            partial_entry_value = position.entry_value_sol * close_pct
            partial_close_value = close_amount * close_price
            partial_pnl = partial_close_value - partial_entry_value

            position.amount -= close_amount
            position.entry_value_sol -= partial_entry_value
            position.realized_pnl_sol += partial_pnl

            logger.info(f"Partial close of position {position_id}: {close_amount} tokens, PnL: {partial_pnl} SOL")
            return position, partial_pnl

    async def set_stop_loss(
        self,
        position_id: str,
        stop_type: StopType,
        trigger_pct: Decimal,
        time_limit: Optional[datetime] = None
    ) -> bool:
        """Set stop-loss for a position"""
        async with self._lock:
            position = self.positions.get(position_id)
            if not position or not position.is_open:
                return False

            position.stop_loss = StopLoss(
                stop_type=stop_type,
                trigger_pct=trigger_pct,
                trailing_high=position.current_price if stop_type == StopType.TRAILING else None,
                time_limit=time_limit
            )

            logger.info(f"Set {stop_type.value} stop-loss for position {position_id} at {trigger_pct}%")
            return True

    async def set_take_profit(
        self,
        position_id: str,
        levels: List[Tuple[Decimal, Decimal]]
    ) -> bool:
        """Set take-profit levels for a position"""
        async with self._lock:
            position = self.positions.get(position_id)
            if not position or not position.is_open:
                return False

            position.take_profit_levels = [
                TakeProfitLevel(price_pct=price_pct, sell_pct=sell_pct)
                for price_pct, sell_pct in levels
            ]

            logger.info(f"Set {len(levels)} take-profit levels for position {position_id}")
            return True

    async def get_open_positions_count(self, user_id: int) -> int:
        """Get count of open positions for a user"""
        positions = await self.get_user_positions(user_id, open_only=True)
        return len(positions)

    async def get_total_exposure(self, user_id: int) -> Decimal:
        """Get total exposure (invested amount) for a user"""
        positions = await self.get_user_positions(user_id, open_only=True)
        return sum(p.current_value_sol or p.entry_value_sol for p in positions)

    async def get_exposure_by_token(self, user_id: int) -> Dict[str, Decimal]:
        """Get exposure broken down by token"""
        positions = await self.get_user_positions(user_id, open_only=True)
        exposure = defaultdict(Decimal)

        for p in positions:
            exposure[p.token_address] += p.current_value_sol or p.entry_value_sol

        return dict(exposure)


class RiskManager:
    """
    Comprehensive risk management system

    Features:
    - Per-user risk settings
    - Position and exposure limits
    - Trade validation
    - Daily loss limits
    - Stop-loss and take-profit management
    - Circuit breaker functionality
    - Risk alerts
    """

    def __init__(self, token_scanner=None):
        self.token_scanner = token_scanner
        self.position_manager = PositionManager()


        self.user_settings: Dict[int, RiskSettings] = {}
        self.user_circuit_breakers: Dict[int, CircuitBreaker] = {}
        self.user_daily_stats: Dict[int, DailyStats] = {}
        self.user_balances: Dict[int, Decimal] = {}


        self.global_settings = RiskSettings()


        self.alerts: List[RiskAlert] = []
        self.alert_callbacks: List[Callable[[RiskAlert], None]] = []


        self._settings_lock = asyncio.Lock()
        self._stats_lock = asyncio.Lock()


        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_monitoring = False


    async def get_user_settings(self, user_id: int) -> RiskSettings:
        """Get risk settings for a user"""
        async with self._settings_lock:
            if user_id not in self.user_settings:
                self.user_settings[user_id] = RiskSettings()
            return self.user_settings[user_id]

    async def update_user_settings(
        self,
        user_id: int,
        updates: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Update user risk settings"""
        async with self._settings_lock:
            current = await self.get_user_settings(user_id)
            new_data = current.to_dict()

            allowed_fields = set(new_data.keys())
            for key in updates:
                if key not in allowed_fields:
                    return False, [f"Unknown setting: {key}"]

            new_data.update(updates)
            new_settings = RiskSettings.from_dict(new_data)


            if not new_settings.is_premium:
                if new_settings.max_position_sol > Decimal("5"):
                    new_settings.max_position_sol = Decimal("5")
                if new_settings.max_total_exposure > Decimal("20"):
                    new_settings.max_total_exposure = Decimal("20")
                if new_settings.max_positions > 15:
                    new_settings.max_positions = 15

            is_valid, errors = new_settings.validate()
            if not is_valid:
                return False, errors

            self.user_settings[user_id] = new_settings
            logger.info(f"Updated risk settings for user {user_id}")
            return True, []

    async def reset_user_settings(self, user_id: int):
        """Reset user settings to defaults"""
        async with self._settings_lock:
            is_premium = self.user_settings.get(user_id, RiskSettings()).is_premium
            self.user_settings[user_id] = RiskSettings(is_premium=is_premium)


    async def get_circuit_breaker(self, user_id: int) -> CircuitBreaker:
        """Get or create circuit breaker for user"""
        if user_id not in self.user_circuit_breakers:
            settings = await self.get_user_settings(user_id)
            self.user_circuit_breakers[user_id] = CircuitBreaker(
                failure_threshold=settings.max_consecutive_losses,
                recovery_timeout=settings.cooldown_minutes
            )
        return self.user_circuit_breakers[user_id]

    async def trigger_circuit_breaker(self, user_id: int, reason: str):
        """Manually trigger circuit breaker"""
        cb = await self.get_circuit_breaker(user_id)
        await cb.force_open(reason)

        await self._create_alert(
            AlertType.CIRCUIT_BREAKER_TRIGGERED,
            user_id,
            RiskLevel.CRITICAL,
            f"Trading halted: {reason}",
            {"reason": reason}
        )

    async def reset_circuit_breaker(self, user_id: int):
        """Manually reset circuit breaker"""
        cb = await self.get_circuit_breaker(user_id)
        await cb.force_close()


    async def get_daily_stats(self, user_id: int) -> DailyStats:
        """Get or create daily stats for user"""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        async with self._stats_lock:
            if user_id not in self.user_daily_stats or self.user_daily_stats[user_id].date != today:
                balance = self.user_balances.get(user_id, Decimal("10"))
                self.user_daily_stats[user_id] = DailyStats(
                    date=today,
                    user_id=user_id,
                    starting_balance=balance,
                    current_balance=balance
                )
            return self.user_daily_stats[user_id]

    async def record_trade_result(
        self,
        user_id: int,
        pnl_sol: Decimal,
        is_win: bool
    ):
        """Record a trade result for daily stats"""
        stats = await self.get_daily_stats(user_id)
        settings = await self.get_user_settings(user_id)

        async with self._stats_lock:
            stats.trades_count += 1
            stats.total_pnl_sol += pnl_sol

            if is_win:
                stats.winning_trades += 1
                stats.consecutive_losses = 0
            else:
                stats.losing_trades += 1
                stats.consecutive_losses += 1

            stats.current_balance += pnl_sol
            if user_id in self.user_balances:
                self.user_balances[user_id] += pnl_sol


            if stats.total_pnl_sol < 0:
                loss_pct = abs(stats.daily_loss_pct)
                loss_sol = abs(stats.total_pnl_sol)

                if loss_pct >= settings.max_daily_loss_pct or loss_sol >= settings.max_daily_loss_sol:
                    stats.trading_halted = True
                    stats.halt_reason = f"Daily loss limit reached: {loss_sol:.4f} SOL ({loss_pct:.2f}%)"
                    await self.trigger_circuit_breaker(user_id, stats.halt_reason)
                elif loss_pct >= settings.max_daily_loss_pct * Decimal("0.8"):
                    await self._create_alert(
                        AlertType.DAILY_LOSS_WARNING,
                        user_id,
                        RiskLevel.HIGH,
                        f"Approaching daily loss limit: {loss_sol:.4f} SOL ({loss_pct:.2f}%)",
                        {"loss_sol": str(loss_sol), "loss_pct": str(loss_pct)}
                    )


            if stats.consecutive_losses >= settings.max_consecutive_losses:
                cb = await self.get_circuit_breaker(user_id)
                await cb.record_failure(f"{stats.consecutive_losses} consecutive losses")


    async def validate_trade(
        self,
        user_id: int,
        trade_params: Dict[str, Any]
    ) -> TradeValidationResult:
        """
        Comprehensive pre-trade validation

        trade_params should include:
        - action: "buy" or "sell"
        - token_address: str
        - amount_sol: Decimal (for buy) or token_amount (for sell)
        - slippage_pct: Optional[Decimal]
        - token_safety_score: Optional[int]
        - token_liquidity: Optional[Decimal]
        """
        settings = await self.get_user_settings(user_id)
        warnings = []


        cb = await self.get_circuit_breaker(user_id)
        can_trade, cb_reason = await cb.can_execute()
        if not can_trade:
            return TradeValidationResult(
                approved=False,
                reason=cb_reason,
                risk_level=RiskLevel.CRITICAL
            )


        stats = await self.get_daily_stats(user_id)
        if stats.trading_halted:
            return TradeValidationResult(
                approved=False,
                reason=f"Daily trading halted: {stats.halt_reason}",
                risk_level=RiskLevel.CRITICAL
            )


        if stats.trades_count >= settings.max_daily_trades:
            return TradeValidationResult(
                approved=False,
                reason=f"Daily trade limit reached ({settings.max_daily_trades} trades)",
                risk_level=RiskLevel.HIGH
            )

        action = trade_params.get("action", "buy")

        if action == "buy":
            return await self._validate_buy(user_id, trade_params, settings, warnings)
        else:
            return await self._validate_sell(user_id, trade_params, settings, warnings)

    async def _validate_buy(
        self,
        user_id: int,
        params: Dict[str, Any],
        settings: RiskSettings,
        warnings: List[str]
    ) -> TradeValidationResult:
        """Validate a buy trade"""
        amount_sol = Decimal(str(params.get("amount_sol", "0")))
        token_address = params.get("token_address", "")
        slippage_pct = Decimal(str(params.get("slippage_pct", settings.default_slippage_pct)))
        token_safety_score = params.get("token_safety_score")
        token_liquidity = params.get("token_liquidity")

        adjusted_params = {}


        if amount_sol > settings.max_position_sol:
            return TradeValidationResult(
                approved=False,
                reason=f"Position size {amount_sol} SOL exceeds limit of {settings.max_position_sol} SOL",
                risk_level=RiskLevel.HIGH
            )


        balance = self.user_balances.get(user_id, Decimal("10"))
        position_pct = (amount_sol / balance) * 100 if balance > 0 else Decimal("100")

        if position_pct > settings.max_position_pct:
            return TradeValidationResult(
                approved=False,
                reason=f"Position is {position_pct:.2f}% of portfolio, exceeds limit of {settings.max_position_pct}%",
                risk_level=RiskLevel.HIGH
            )


        if amount_sol > balance:
            return TradeValidationResult(
                approved=False,
                reason=f"Insufficient balance: {balance} SOL available, {amount_sol} SOL required",
                risk_level=RiskLevel.HIGH
            )


        current_exposure = await self.position_manager.get_total_exposure(user_id)
        new_exposure = current_exposure + amount_sol

        if new_exposure > settings.max_total_exposure:
            return TradeValidationResult(
                approved=False,
                reason=f"Total exposure would be {new_exposure} SOL, exceeds limit of {settings.max_total_exposure} SOL",
                risk_level=RiskLevel.HIGH
            )


        open_positions = await self.position_manager.get_open_positions_count(user_id)
        if open_positions >= settings.max_positions:
            return TradeValidationResult(
                approved=False,
                reason=f"Maximum positions ({settings.max_positions}) reached",
                risk_level=RiskLevel.MEDIUM
            )


        existing_positions = await self.position_manager.get_token_positions(token_address, user_id)
        if existing_positions:
            warnings.append(f"Already have {len(existing_positions)} position(s) in this token")


        if slippage_pct > settings.max_slippage_pct:
            adjusted_params["slippage_pct"] = str(settings.max_slippage_pct)
            warnings.append(f"Slippage adjusted from {slippage_pct}% to {settings.max_slippage_pct}%")
            slippage_pct = settings.max_slippage_pct


        if token_safety_score is not None:
            if token_safety_score < settings.min_token_safety_score:
                return TradeValidationResult(
                    approved=False,
                    reason=f"Token safety score ({token_safety_score}) below minimum ({settings.min_token_safety_score})",
                    risk_level=RiskLevel.HIGH,
                    warnings=warnings
                )
            elif token_safety_score < 50:
                warnings.append(f"Low token safety score: {token_safety_score}")


        if settings.require_liquidity_check and token_liquidity is not None:
            if Decimal(str(token_liquidity)) < settings.min_liquidity_sol:
                return TradeValidationResult(
                    approved=False,
                    reason=f"Token liquidity ({token_liquidity} SOL) below minimum ({settings.min_liquidity_sol} SOL)",
                    risk_level=RiskLevel.HIGH,
                    warnings=warnings
                )


        risk_level = RiskLevel.LOW
        if warnings:
            risk_level = RiskLevel.MEDIUM
        if position_pct > settings.max_position_pct * Decimal("0.8"):
            risk_level = RiskLevel.MEDIUM
        if new_exposure > settings.max_total_exposure * Decimal("0.8"):
            risk_level = RiskLevel.HIGH

        return TradeValidationResult(
            approved=True,
            reason="Trade approved",
            risk_level=risk_level,
            warnings=warnings,
            adjusted_params=adjusted_params if adjusted_params else None
        )

    async def _validate_sell(
        self,
        user_id: int,
        params: Dict[str, Any],
        settings: RiskSettings,
        warnings: List[str]
    ) -> TradeValidationResult:
        """Validate a sell trade"""
        position_id = params.get("position_id")
        token_address = params.get("token_address")
        sell_pct = Decimal(str(params.get("sell_pct", "100")))

        if position_id:
            position = await self.position_manager.get_position(position_id)
            if not position or not position.is_open:
                return TradeValidationResult(
                    approved=False,
                    reason="Position not found or already closed",
                    risk_level=RiskLevel.LOW
                )
        elif token_address:
            positions = await self.position_manager.get_token_positions(token_address, user_id)
            if not positions:
                return TradeValidationResult(
                    approved=False,
                    reason="No open position for this token",
                    risk_level=RiskLevel.LOW
                )
        else:
            return TradeValidationResult(
                approved=False,
                reason="Must specify position_id or token_address",
                risk_level=RiskLevel.LOW
            )

        if sell_pct <= 0 or sell_pct > 100:
            return TradeValidationResult(
                approved=False,
                reason="Sell percentage must be between 0 and 100",
                risk_level=RiskLevel.LOW
            )

        return TradeValidationResult(
            approved=True,
            reason="Sell approved",
            risk_level=RiskLevel.LOW,
            warnings=warnings
        )


    async def set_stop_loss(
        self,
        position_id: str,
        price_pct: Decimal,
        stop_type: StopType = StopType.FIXED
    ) -> bool:
        """Set stop-loss for a position"""
        return await self.position_manager.set_stop_loss(
            position_id,
            stop_type,
            price_pct
        )

    async def set_trailing_stop(
        self,
        position_id: str,
        trail_pct: Decimal
    ) -> bool:
        """Set trailing stop for a position"""
        return await self.position_manager.set_stop_loss(
            position_id,
            StopType.TRAILING,
            trail_pct
        )

    async def set_break_even_stop(self, position_id: str) -> bool:
        """Set stop-loss at break-even (entry price)"""
        return await self.position_manager.set_stop_loss(
            position_id,
            StopType.BREAK_EVEN,
            Decimal("0")
        )

    async def set_time_based_stop(
        self,
        position_id: str,
        hours: int,
        fallback_pct: Decimal = Decimal("10")
    ) -> bool:
        """Set time-based stop - exits after X hours if not in profit"""
        time_limit = datetime.now(timezone.utc) + timedelta(hours=hours)
        return await self.position_manager.set_stop_loss(
            position_id,
            StopType.TIME_BASED,
            fallback_pct,
            time_limit=time_limit
        )


    async def set_take_profit(
        self,
        position_id: str,
        price_pct: Decimal
    ) -> bool:
        """Set single take-profit level (100% exit)"""
        return await self.position_manager.set_take_profit(
            position_id,
            [(price_pct, Decimal("100"))]
        )

    async def set_scaled_take_profit(
        self,
        position_id: str,
        levels: List[Tuple[Decimal, Decimal]]
    ) -> bool:
        """
        Set multiple take-profit levels

        levels: [(price_pct, sell_pct), ...]
        Example: [(50, 25), (100, 50), (200, 25)]
                 = Sell 25% at +50%, 50% at +100%, 25% at +200%
        """
        total_sell_pct = sum(level[1] for level in levels)
        if total_sell_pct != Decimal("100"):
            logger.warning(f"Take-profit levels sum to {total_sell_pct}%, not 100%")

        return await self.position_manager.set_take_profit(position_id, levels)


    async def get_current_exposure(self, user_id: int) -> Dict[str, Any]:
        """Get current exposure summary for a user"""
        positions = await self.position_manager.get_user_positions(user_id, open_only=True)
        settings = await self.get_user_settings(user_id)

        total_exposure = sum(p.current_value_sol or p.entry_value_sol for p in positions)
        total_unrealized_pnl = sum(p.unrealized_pnl_sol for p in positions)

        return {
            "total_exposure_sol": str(total_exposure),
            "exposure_limit_sol": str(settings.max_total_exposure),
            "exposure_pct": str((total_exposure / settings.max_total_exposure * 100) if settings.max_total_exposure > 0 else Decimal("0")),
            "open_positions": len(positions),
            "max_positions": settings.max_positions,
            "total_unrealized_pnl": str(total_unrealized_pnl),
            "positions": [p.to_dict() for p in positions]
        }

    async def get_exposure_by_token(self, user_id: int) -> Dict[str, Dict[str, Any]]:
        """Get exposure broken down by token"""
        positions = await self.position_manager.get_user_positions(user_id, open_only=True)

        token_exposure = {}
        for p in positions:
            if p.token_address not in token_exposure:
                token_exposure[p.token_address] = {
                    "symbol": p.token_symbol,
                    "total_value_sol": Decimal("0"),
                    "total_pnl_sol": Decimal("0"),
                    "position_count": 0
                }

            token_exposure[p.token_address]["total_value_sol"] += p.current_value_sol or p.entry_value_sol
            token_exposure[p.token_address]["total_pnl_sol"] += p.unrealized_pnl_sol
            token_exposure[p.token_address]["position_count"] += 1

        for token in token_exposure:
            token_exposure[token]["total_value_sol"] = str(token_exposure[token]["total_value_sol"])
            token_exposure[token]["total_pnl_sol"] = str(token_exposure[token]["total_pnl_sol"])

        return token_exposure

    async def get_pnl(
        self,
        user_id: int,
        period: str = "daily"
    ) -> Dict[str, Any]:
        """
        Get profit/loss for a period

        period: "daily", "weekly", "monthly", "all"
        """
        positions = await self.position_manager.get_user_positions(user_id, open_only=False)

        now = datetime.now(timezone.utc)

        if period == "daily":
            cutoff = now - timedelta(days=1)
        elif period == "weekly":
            cutoff = now - timedelta(weeks=1)
        elif period == "monthly":
            cutoff = now - timedelta(days=30)
        else:
            cutoff = datetime.min.replace(tzinfo=timezone.utc)

        period_positions = [
            p for p in positions
            if p.entry_time >= cutoff
        ]

        realized_pnl = sum(p.realized_pnl_sol for p in period_positions if not p.is_open)
        unrealized_pnl = sum(p.unrealized_pnl_sol for p in period_positions if p.is_open)

        winning = len([p for p in period_positions if not p.is_open and p.realized_pnl_sol > 0])
        losing = len([p for p in period_positions if not p.is_open and p.realized_pnl_sol < 0])

        return {
            "period": period,
            "realized_pnl_sol": str(realized_pnl),
            "unrealized_pnl_sol": str(unrealized_pnl),
            "total_pnl_sol": str(realized_pnl + unrealized_pnl),
            "trades_count": len([p for p in period_positions if not p.is_open]),
            "winning_trades": winning,
            "losing_trades": losing,
            "win_rate": str(Decimal(winning) / Decimal(winning + losing) * 100 if (winning + losing) > 0 else Decimal("0")),
            "open_positions": len([p for p in period_positions if p.is_open])
        }


    async def _create_alert(
        self,
        alert_type: AlertType,
        user_id: int,
        severity: RiskLevel,
        message: str,
        data: Dict[str, Any]
    ):
        """Create and dispatch a risk alert"""
        alert = RiskAlert(
            alert_id=str(uuid.uuid4())[:8],
            alert_type=alert_type,
            user_id=user_id,
            severity=severity,
            message=message,
            data=data,
            timestamp=datetime.now(timezone.utc)
        )

        self.alerts.append(alert)

        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]

        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

        logger.warning(f"Risk alert [{severity.value}]: {message}")

    def register_alert_callback(self, callback: Callable[[RiskAlert], None]):
        """Register a callback for risk alerts"""
        self.alert_callbacks.append(callback)

    async def get_alerts(
        self,
        user_id: Optional[int] = None,
        severity: Optional[RiskLevel] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent alerts, optionally filtered"""
        alerts = self.alerts

        if user_id is not None:
            alerts = [a for a in alerts if a.user_id == user_id]

        if severity is not None:
            alerts = [a for a in alerts if a.severity == severity]

        return [a.to_dict() for a in reversed(alerts[-limit:])]

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False


    async def on_position_opened(
        self,
        user_id: int,
        token_address: str,
        token_symbol: str,
        entry_price: Decimal,
        amount: Decimal,
        entry_value_sol: Decimal
    ) -> Position:
        """Handle new position opened"""
        settings = await self.get_user_settings(user_id)

        stop_loss = None
        if settings.auto_stop_loss:
            stop_loss = StopLoss(
                stop_type=StopType.FIXED,
                trigger_pct=settings.default_stop_loss_pct
            )

        take_profit_levels = []
        if settings.auto_take_profit:
            take_profit_levels = [
                TakeProfitLevel(
                    price_pct=settings.default_take_profit_pct,
                    sell_pct=Decimal("100")
                )
            ]

        position = await self.position_manager.create_position(
            user_id=user_id,
            token_address=token_address,
            token_symbol=token_symbol,
            entry_price=entry_price,
            amount=amount,
            entry_value_sol=entry_value_sol,
            stop_loss=stop_loss,
            take_profit_levels=take_profit_levels
        )

        if user_id in self.user_balances:
            self.user_balances[user_id] -= entry_value_sol

        exposure = await self.position_manager.get_total_exposure(user_id)
        if exposure > settings.max_total_exposure * Decimal("0.8"):
            await self._create_alert(
                AlertType.HIGH_EXPOSURE,
                user_id,
                RiskLevel.MEDIUM,
                f"High exposure warning: {exposure} SOL ({exposure / settings.max_total_exposure * 100:.1f}% of limit)",
                {"exposure_sol": str(exposure), "limit_sol": str(settings.max_total_exposure)}
            )

        return position

    async def on_position_closed(
        self,
        position_id: str,
        close_price: Decimal,
        close_reason: str
    ) -> Optional[Position]:
        """Handle position closed"""
        position = await self.position_manager.close_position(
            position_id,
            close_price,
            close_reason
        )

        if position:
            is_win = position.realized_pnl_sol > 0
            await self.record_trade_result(
                position.user_id,
                position.realized_pnl_sol,
                is_win
            )

            if position.user_id in self.user_balances:
                final_value = position.amount * close_price
                self.user_balances[position.user_id] += final_value

            cb = await self.get_circuit_breaker(position.user_id)
            if is_win:
                await cb.record_success()
            else:
                await cb.record_failure(f"Trade loss: {position.realized_pnl_sol} SOL")

        return position

    async def check_stop_loss(self, position: Position, current_price: Decimal) -> bool:
        """Check if stop-loss should trigger"""
        if not position.stop_loss or position.stop_loss.triggered:
            return False

        stop = position.stop_loss
        trigger_price = stop.get_trigger_price(position.entry_price)

        if stop.stop_type == StopType.TIME_BASED and stop.time_limit:
            if datetime.now(timezone.utc) >= stop.time_limit:
                if current_price <= position.entry_price:
                    return True

        if current_price <= trigger_price:
            return True

        return False

    async def check_take_profit(self, position: Position, current_price: Decimal) -> Optional[TakeProfitLevel]:
        """Check if any take-profit level should trigger"""
        for level in position.take_profit_levels:
            if level.triggered:
                continue

            trigger_price = position.entry_price * (1 + level.price_pct / 100)
            if current_price >= trigger_price:
                return level

        return None


    async def start_monitoring(self, price_callback: Callable[[str], Decimal]):
        """Start position monitoring task"""
        self._stop_monitoring = False
        self._monitoring_task = asyncio.create_task(
            self._monitor_positions(price_callback)
        )
        logger.info("Position monitoring started")

    async def stop_monitoring(self):
        """Stop position monitoring"""
        self._stop_monitoring = True
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Position monitoring stopped")

    async def _monitor_positions(self, price_callback: Callable[[str], Decimal]):
        """Background task to monitor positions"""
        while not self._stop_monitoring:
            try:
                all_positions = []
                for user_id in list(self.position_manager.user_positions.keys()):
                    positions = await self.position_manager.get_user_positions(user_id, open_only=True)
                    all_positions.extend(positions)

                for position in all_positions:
                    try:
                        current_price = await price_callback(position.token_address)
                        if current_price is None:
                            continue

                        current_value = position.amount * current_price
                        await self.position_manager.update_position_price(
                            position.position_id,
                            current_price,
                            current_value
                        )

                        if await self.check_stop_loss(position, current_price):
                            position.stop_loss.triggered = True
                            position.stop_loss.triggered_at = datetime.now(timezone.utc)

                            await self._create_alert(
                                AlertType.STOP_LOSS_TRIGGERED,
                                position.user_id,
                                RiskLevel.HIGH,
                                f"Stop-loss triggered for {position.token_symbol}",
                                {"position_id": position.position_id, "trigger_price": str(current_price)}
                            )

                            logger.warning(f"Stop-loss triggered for position {position.position_id}")

                        tp_level = await self.check_take_profit(position, current_price)
                        if tp_level:
                            tp_level.triggered = True
                            tp_level.triggered_at = datetime.now(timezone.utc)

                            await self._create_alert(
                                AlertType.TAKE_PROFIT_TRIGGERED,
                                position.user_id,
                                RiskLevel.LOW,
                                f"Take-profit triggered for {position.token_symbol} at +{tp_level.price_pct}%",
                                {"position_id": position.position_id, "level_pct": str(tp_level.price_pct)}
                            )

                            logger.info(f"Take-profit triggered for position {position.position_id}")

                    except Exception as e:
                        logger.error(f"Error monitoring position {position.position_id}: {e}")

                await asyncio.sleep(5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in position monitoring: {e}")
                await asyncio.sleep(10)


    async def set_user_balance(self, user_id: int, balance: Decimal):
        """Set user balance (called by trading system)"""
        self.user_balances[user_id] = balance

        stats = await self.get_daily_stats(user_id)
        if stats.starting_balance == Decimal("0"):
            stats.starting_balance = balance
        stats.current_balance = balance

    async def get_user_balance(self, user_id: int) -> Decimal:
        """Get cached user balance"""
        return self.user_balances.get(user_id, Decimal("0"))


    async def get_risk_status(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive risk status for a user"""
        settings = await self.get_user_settings(user_id)
        stats = await self.get_daily_stats(user_id)
        cb = await self.get_circuit_breaker(user_id)
        exposure = await self.get_current_exposure(user_id)

        return {
            "user_id": user_id,
            "settings": settings.to_dict(),
            "daily_stats": {
                "date": stats.date,
                "trades_count": stats.trades_count,
                "pnl_sol": str(stats.total_pnl_sol),
                "pnl_pct": str(stats.daily_loss_pct),
                "win_rate": str(stats.win_rate),
                "consecutive_losses": stats.consecutive_losses,
                "trading_halted": stats.trading_halted,
                "halt_reason": stats.halt_reason
            },
            "circuit_breaker": cb.get_status(),
            "exposure": exposure,
            "balance": str(self.user_balances.get(user_id, Decimal("0")))
        }


def create_risk_manager(token_scanner=None) -> RiskManager:
    """Create a new RiskManager instance"""
    return RiskManager(token_scanner=token_scanner)


async def test_risk_manager():
    """Test the risk manager"""
    print("Testing Risk Manager...")

    rm = create_risk_manager()
    user_id = 12345

    await rm.set_user_balance(user_id, Decimal("10.0"))

    settings = await rm.get_user_settings(user_id)
    print(f"Default settings: max_position={settings.max_position_sol} SOL")

    success, errors = await rm.update_user_settings(user_id, {
        "max_position_sol": "2.0",
        "default_stop_loss_pct": "20.0"
    })
    print(f"Settings update: success={success}, errors={errors}")

    result = await rm.validate_trade(user_id, {
        "action": "buy",
        "token_address": "token123",
        "amount_sol": "1.0",
        "token_safety_score": 75
    })
    print(f"Trade validation: approved={result.approved}, reason={result.reason}")

    position = await rm.on_position_opened(
        user_id=user_id,
        token_address="token123",
        token_symbol="TEST",
        entry_price=Decimal("0.001"),
        amount=Decimal("1000"),
        entry_value_sol=Decimal("1.0")
    )
    print(f"Position created: {position.position_id}")

    await rm.set_trailing_stop(position.position_id, Decimal("10"))
    print("Trailing stop set at 10%")

    await rm.set_scaled_take_profit(position.position_id, [
        (Decimal("50"), Decimal("33")),
        (Decimal("100"), Decimal("33")),
        (Decimal("200"), Decimal("34"))
    ])
    print("Scaled take-profit set")

    exposure = await rm.get_current_exposure(user_id)
    print(f"Current exposure: {exposure['total_exposure_sol']} SOL")

    status = await rm.get_risk_status(user_id)
    print(f"Risk status: CB={status['circuit_breaker']['state']}")

    cb = await rm.get_circuit_breaker(user_id)
    for i in range(5):
        await cb.record_failure("Test failure")

    can_trade, reason = await cb.can_execute()
    print(f"After 5 failures: can_trade={can_trade}, reason={reason}")

    print("\nRisk Manager tests completed!")


if __name__ == "__main__":
    asyncio.run(test_risk_manager())
