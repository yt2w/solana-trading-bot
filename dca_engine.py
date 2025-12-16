"""
DCA (Dollar Cost Averaging) Engine
Production-grade automated DCA execution system for Solana tokens.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, List, Any, Callable, Awaitable
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import json
from pathlib import Path
import aiosqlite
from collections import defaultdict

logger = logging.getLogger(__name__)


class DCAFrequency(Enum):
    """DCA execution frequency options."""
    HOURLY = "hourly"
    EVERY_4_HOURS = "every_4_hours"
    EVERY_8_HOURS = "every_8_hours"
    EVERY_12_HOURS = "every_12_hours"
    DAILY = "daily"
    TWICE_WEEKLY = "twice_weekly"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class DCAStatus(Enum):
    """DCA plan status."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PENDING = "pending"


class DayOfWeek(Enum):
    """Days of the week for scheduling."""
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


@dataclass
class DCAConfig:
    """Configuration for a DCA plan."""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: int = 0  # Telegram user ID
    token_mint: str = ""  # Target token mint address
    token_symbol: str = ""  # Token symbol for display
    amount_per_buy: Decimal = Decimal("0.1")  # SOL per purchase
    frequency: DCAFrequency = DCAFrequency.DAILY
    custom_interval_hours: Optional[int] = None  # For custom frequency
    day_of_week: Optional[DayOfWeek] = None  # For weekly
    days_of_week: List[int] = field(default_factory=list)  # For twice_weekly
    time_of_day: str = "12:00"  # HH:MM format
    total_budget: Optional[Decimal] = None  # Max total spend in SOL
    max_executions: Optional[int] = None  # Max number of buys
    end_date: Optional[datetime] = None  # Optional end date
    slippage_bps: int = 300  # 3% default slippage
    smart_timing: bool = True  # Avoid high-fee periods
    max_priority_fee_lamports: int = 100000  # Max priority fee (0.0001 SOL)
    retry_on_high_fees: bool = True  # Retry if fees too high
    safety_check_enabled: bool = True  # Check token safety before buy
    status: DCAStatus = DCAStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['frequency'] = self.frequency.value
        data['status'] = self.status.value
        data['day_of_week'] = self.day_of_week.value if self.day_of_week else None
        data['amount_per_buy'] = str(self.amount_per_buy)
        data['total_budget'] = str(self.total_budget) if self.total_budget else None
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['end_date'] = self.end_date.isoformat() if self.end_date else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DCAConfig':
        """Create from dictionary."""
        data = data.copy()
        data['frequency'] = DCAFrequency(data['frequency'])
        data['status'] = DCAStatus(data['status'])
        data['day_of_week'] = DayOfWeek(data['day_of_week']) if data.get('day_of_week') is not None else None
        data['amount_per_buy'] = Decimal(data['amount_per_buy'])
        data['total_budget'] = Decimal(data['total_budget']) if data.get('total_budget') else None
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        data['end_date'] = datetime.fromisoformat(data['end_date']) if data.get('end_date') else None
        if 'days_of_week' not in data:
            data['days_of_week'] = []
        return cls(**data)


@dataclass
class DCAStats:
    """Statistics for a DCA plan."""
    plan_id: str = ""
    total_invested_sol: Decimal = Decimal("0")
    total_invested_usd: Decimal = Decimal("0")
    total_tokens_acquired: Decimal = Decimal("0")
    average_price_sol: Decimal = Decimal("0")
    average_price_usd: Decimal = Decimal("0")
    best_price_sol: Optional[Decimal] = None
    worst_price_sol: Optional[Decimal] = None
    current_price_sol: Optional[Decimal] = None
    execution_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0  # Skipped due to high fees, etc.
    total_fees_paid_sol: Decimal = Decimal("0")
    total_priority_fees_sol: Decimal = Decimal("0")
    first_execution: Optional[datetime] = None
    last_execution: Optional[datetime] = None
    next_execution: Optional[datetime] = None
    unrealized_pnl_sol: Decimal = Decimal("0")
    unrealized_pnl_percent: Decimal = Decimal("0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'plan_id': self.plan_id,
            'total_invested_sol': str(self.total_invested_sol),
            'total_invested_usd': str(self.total_invested_usd),
            'total_tokens_acquired': str(self.total_tokens_acquired),
            'average_price_sol': str(self.average_price_sol),
            'average_price_usd': str(self.average_price_usd),
            'best_price_sol': str(self.best_price_sol) if self.best_price_sol else None,
            'worst_price_sol': str(self.worst_price_sol) if self.worst_price_sol else None,
            'current_price_sol': str(self.current_price_sol) if self.current_price_sol else None,
            'execution_count': self.execution_count,
            'failed_count': self.failed_count,
            'skipped_count': self.skipped_count,
            'total_fees_paid_sol': str(self.total_fees_paid_sol),
            'total_priority_fees_sol': str(self.total_priority_fees_sol),
            'first_execution': self.first_execution.isoformat() if self.first_execution else None,
            'last_execution': self.last_execution.isoformat() if self.last_execution else None,
            'next_execution': self.next_execution.isoformat() if self.next_execution else None,
            'unrealized_pnl_sol': str(self.unrealized_pnl_sol),
            'unrealized_pnl_percent': str(self.unrealized_pnl_percent),
        }


@dataclass
class DCAExecution:
    """Record of a single DCA execution."""
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    plan_id: str = ""
    user_id: int = 0
    token_mint: str = ""
    amount_sol: Decimal = Decimal("0")
    tokens_received: Decimal = Decimal("0")
    price_per_token_sol: Decimal = Decimal("0")
    price_per_token_usd: Decimal = Decimal("0")
    sol_price_usd: Decimal = Decimal("0")
    tx_signature: Optional[str] = None
    priority_fee_lamports: int = 0
    network_fee_lamports: int = 5000
    status: str = "pending"  # pending, success, failed, skipped
    error_message: Optional[str] = None
    retry_count: int = 0
    executed_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'execution_id': self.execution_id,
            'plan_id': self.plan_id,
            'user_id': self.user_id,
            'token_mint': self.token_mint,
            'amount_sol': str(self.amount_sol),
            'tokens_received': str(self.tokens_received),
            'price_per_token_sol': str(self.price_per_token_sol),
            'price_per_token_usd': str(self.price_per_token_usd),
            'sol_price_usd': str(self.sol_price_usd),
            'tx_signature': self.tx_signature,
            'priority_fee_lamports': self.priority_fee_lamports,
            'network_fee_lamports': self.network_fee_lamports,
            'status': self.status,
            'error_message': self.error_message,
            'retry_count': self.retry_count,
            'executed_at': self.executed_at.isoformat(),
        }


class DCAScheduler:
    """Handles scheduling of DCA executions."""
    
    def __init__(self):
        self._tasks: Dict[str, asyncio.Task] = {}
        self._next_executions: Dict[str, datetime] = {}
    
    def calculate_next_execution(
        self,
        config: DCAConfig,
        from_time: Optional[datetime] = None
    ) -> Optional[datetime]:
        """Calculate the next execution time for a plan."""
        now = from_time or datetime.utcnow()
        
        # Parse time of day
        try:
            hour, minute = map(int, config.time_of_day.split(':'))
        except:
            hour, minute = 12, 0
        
        if config.frequency == DCAFrequency.HOURLY:
            # Next hour at the specified minute
            next_exec = now.replace(minute=minute, second=0, microsecond=0)
            if next_exec <= now:
                next_exec += timedelta(hours=1)
                
        elif config.frequency == DCAFrequency.EVERY_4_HOURS:
            next_exec = now.replace(minute=minute, second=0, microsecond=0)
            # Round to next 4-hour interval (0, 4, 8, 12, 16, 20)
            target_hour = ((now.hour // 4) + 1) * 4
            if target_hour >= 24:
                next_exec = next_exec.replace(hour=0) + timedelta(days=1)
            else:
                next_exec = next_exec.replace(hour=target_hour)
                
        elif config.frequency == DCAFrequency.EVERY_8_HOURS:
            next_exec = now.replace(minute=minute, second=0, microsecond=0)
            target_hour = ((now.hour // 8) + 1) * 8
            if target_hour >= 24:
                next_exec = next_exec.replace(hour=0) + timedelta(days=1)
            else:
                next_exec = next_exec.replace(hour=target_hour)
                
        elif config.frequency == DCAFrequency.EVERY_12_HOURS:
            next_exec = now.replace(minute=minute, second=0, microsecond=0)
            target_hour = ((now.hour // 12) + 1) * 12
            if target_hour >= 24:
                next_exec = next_exec.replace(hour=0) + timedelta(days=1)
            else:
                next_exec = next_exec.replace(hour=target_hour)
                
        elif config.frequency == DCAFrequency.DAILY:
            next_exec = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_exec <= now:
                next_exec += timedelta(days=1)
                
        elif config.frequency == DCAFrequency.TWICE_WEEKLY:
            # Default to Monday and Thursday if not specified
            days = config.days_of_week if config.days_of_week else [0, 3]
            next_exec = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # Find next valid day
            for i in range(8):
                check_date = next_exec + timedelta(days=i)
                if check_date.weekday() in days and check_date > now:
                    next_exec = check_date
                    break
                    
        elif config.frequency == DCAFrequency.WEEKLY:
            target_day = config.day_of_week.value if config.day_of_week else 0
            next_exec = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            days_ahead = target_day - now.weekday()
            if days_ahead <= 0 or (days_ahead == 0 and next_exec <= now):
                days_ahead += 7
            next_exec += timedelta(days=days_ahead)
            
        elif config.frequency == DCAFrequency.BIWEEKLY:
            target_day = config.day_of_week.value if config.day_of_week else 0
            next_exec = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            days_ahead = target_day - now.weekday()
            if days_ahead <= 0 or (days_ahead == 0 and next_exec <= now):
                days_ahead += 14
            else:
                days_ahead += 7  # Add extra week for biweekly
            next_exec += timedelta(days=days_ahead)
            
        elif config.frequency == DCAFrequency.MONTHLY:
            next_exec = now.replace(
                day=1, hour=hour, minute=minute, second=0, microsecond=0
            )
            if next_exec <= now:
                # Move to next month
                if next_exec.month == 12:
                    next_exec = next_exec.replace(year=next_exec.year + 1, month=1)
                else:
                    next_exec = next_exec.replace(month=next_exec.month + 1)
                    
        elif config.frequency == DCAFrequency.CUSTOM:
            interval_hours = config.custom_interval_hours or 24
            next_exec = now + timedelta(hours=interval_hours)
            
        else:
            next_exec = now + timedelta(days=1)
        
        # Check against end date
        if config.end_date and next_exec > config.end_date:
            return None
            
        return next_exec
    
    def get_next_execution(self, plan_id: str) -> Optional[datetime]:
        """Get scheduled next execution time."""
        return self._next_executions.get(plan_id)
    
    def set_next_execution(self, plan_id: str, next_time: datetime):
        """Set next execution time."""
        self._next_executions[plan_id] = next_time
    
    def clear_schedule(self, plan_id: str):
        """Clear scheduled execution."""
        self._next_executions.pop(plan_id, None)
        if plan_id in self._tasks:
            self._tasks[plan_id].cancel()
            del self._tasks[plan_id]


class DCAEngine:
    """
    Main DCA (Dollar Cost Averaging) Engine.
    Manages multiple concurrent DCA plans with background scheduling.
    """
    
    def __init__(
        self,
        db_path: str = "data/dca_engine.db",
        jupiter_client: Any = None,
        wallet_manager: Any = None,
        token_analyzer: Any = None,
        notification_callback: Optional[Callable[[int, str, Dict], Awaitable[None]]] = None
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.jupiter = jupiter_client
        self.wallet_manager = wallet_manager
        self.token_analyzer = token_analyzer
        self.notify = notification_callback
        
        self.scheduler = DCAScheduler()
        self._plans: Dict[str, DCAConfig] = {}
        self._stats: Dict[str, DCAStats] = {}
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._execution_lock = asyncio.Lock()
        self._plan_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # Configuration
        self.max_retries = 3
        self.retry_delay_seconds = 30
        self.high_fee_threshold_lamports = 50000  # 0.00005 SOL
        self.high_fee_delay_minutes = 5
        self.max_concurrent_executions = 5
        
        logger.info("DCA Engine initialized")
    
    async def initialize(self):
        """Initialize the DCA engine and database."""
        await self._init_database()
        await self._load_plans()
        logger.info(f"DCA Engine initialized with {len(self._plans)} plans")
    
    async def _init_database(self):
        """Initialize database tables."""
        async with aiosqlite.connect(self.db_path) as db:
            # DCA Plans table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS dca_plans (
                    plan_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # DCA Statistics table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS dca_stats (
                    plan_id TEXT PRIMARY KEY,
                    stats_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (plan_id) REFERENCES dca_plans(plan_id)
                )
            """)
            
            # DCA Executions table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS dca_executions (
                    execution_id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    execution_json TEXT NOT NULL,
                    executed_at TEXT NOT NULL,
                    FOREIGN KEY (plan_id) REFERENCES dca_plans(plan_id)
                )
            """)
            
            # Indexes
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_plans_user ON dca_plans(user_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_executions_plan ON dca_executions(plan_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_executions_user ON dca_executions(user_id)"
            )
            
            await db.commit()
            logger.info("DCA database initialized")
    
    async def _load_plans(self):
        """Load all plans from database."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT plan_id, config_json FROM dca_plans"
            ) as cursor:
                async for row in cursor:
                    try:
                        config = DCAConfig.from_dict(json.loads(row[1]))
                        self._plans[config.plan_id] = config
                        
                        # Load stats
                        async with db.execute(
                            "SELECT stats_json FROM dca_stats WHERE plan_id = ?",
                            (config.plan_id,)
                        ) as stats_cursor:
                            stats_row = await stats_cursor.fetchone()
                            if stats_row:
                                stats_data = json.loads(stats_row[0])
                                self._stats[config.plan_id] = self._parse_stats(stats_data)
                            else:
                                self._stats[config.plan_id] = DCAStats(plan_id=config.plan_id)
                                
                    except Exception as e:
                        logger.error(f"Failed to load plan {row[0]}: {e}")
    
    def _parse_stats(self, data: Dict) -> DCAStats:
        """Parse stats from dictionary."""
        return DCAStats(
            plan_id=data.get('plan_id', ''),
            total_invested_sol=Decimal(data.get('total_invested_sol', '0')),
            total_invested_usd=Decimal(data.get('total_invested_usd', '0')),
            total_tokens_acquired=Decimal(data.get('total_tokens_acquired', '0')),
            average_price_sol=Decimal(data.get('average_price_sol', '0')),
            average_price_usd=Decimal(data.get('average_price_usd', '0')),
            best_price_sol=Decimal(data['best_price_sol']) if data.get('best_price_sol') else None,
            worst_price_sol=Decimal(data['worst_price_sol']) if data.get('worst_price_sol') else None,
            current_price_sol=Decimal(data['current_price_sol']) if data.get('current_price_sol') else None,
            execution_count=data.get('execution_count', 0),
            failed_count=data.get('failed_count', 0),
            skipped_count=data.get('skipped_count', 0),
            total_fees_paid_sol=Decimal(data.get('total_fees_paid_sol', '0')),
            total_priority_fees_sol=Decimal(data.get('total_priority_fees_sol', '0')),
            first_execution=datetime.fromisoformat(data['first_execution']) if data.get('first_execution') else None,
            last_execution=datetime.fromisoformat(data['last_execution']) if data.get('last_execution') else None,
            next_execution=datetime.fromisoformat(data['next_execution']) if data.get('next_execution') else None,
            unrealized_pnl_sol=Decimal(data.get('unrealized_pnl_sol', '0')),
            unrealized_pnl_percent=Decimal(data.get('unrealized_pnl_percent', '0')),
        )
    
    async def _save_plan(self, config: DCAConfig):
        """Save plan to database."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO dca_plans (plan_id, user_id, config_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                config.plan_id,
                config.user_id,
                json.dumps(config.to_dict()),
                config.created_at.isoformat(),
                datetime.utcnow().isoformat()
            ))
            await db.commit()
    
    async def _save_stats(self, stats: DCAStats):
        """Save stats to database."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO dca_stats (plan_id, stats_json, updated_at)
                VALUES (?, ?, ?)
            """, (
                stats.plan_id,
                json.dumps(stats.to_dict()),
                datetime.utcnow().isoformat()
            ))
            await db.commit()
    
    async def _save_execution(self, execution: DCAExecution):
        """Save execution to database."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO dca_executions 
                (execution_id, plan_id, user_id, execution_json, executed_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                execution.execution_id,
                execution.plan_id,
                execution.user_id,
                json.dumps(execution.to_dict()),
                execution.executed_at.isoformat()
            ))
            await db.commit()

    
    # =========================================================================
    # Plan Management
    # =========================================================================
    
    async def create_plan(
        self,
        user_id: int,
        token_mint: str,
        token_symbol: str,
        amount_per_buy: Decimal,
        frequency: DCAFrequency = DCAFrequency.DAILY,
        time_of_day: str = "12:00",
        total_budget: Optional[Decimal] = None,
        max_executions: Optional[int] = None,
        end_date: Optional[datetime] = None,
        day_of_week: Optional[DayOfWeek] = None,
        slippage_bps: int = 300,
        smart_timing: bool = True,
        auto_start: bool = True
    ) -> DCAConfig:
        """Create a new DCA plan."""
        config = DCAConfig(
            user_id=user_id,
            token_mint=token_mint,
            token_symbol=token_symbol,
            amount_per_buy=amount_per_buy,
            frequency=frequency,
            time_of_day=time_of_day,
            total_budget=total_budget,
            max_executions=max_executions,
            end_date=end_date,
            day_of_week=day_of_week,
            slippage_bps=slippage_bps,
            smart_timing=smart_timing,
            status=DCAStatus.ACTIVE if auto_start else DCAStatus.PENDING
        )
        
        # Calculate first execution
        next_exec = self.scheduler.calculate_next_execution(config)
        
        # Initialize stats
        stats = DCAStats(plan_id=config.plan_id, next_execution=next_exec)
        
        # Store in memory and database
        self._plans[config.plan_id] = config
        self._stats[config.plan_id] = stats
        
        await self._save_plan(config)
        await self._save_stats(stats)
        
        # Schedule if active
        if config.status == DCAStatus.ACTIVE and next_exec:
            self.scheduler.set_next_execution(config.plan_id, next_exec)
        
        logger.info(
            f"Created DCA plan {config.plan_id} for user {user_id}: "
            f"{amount_per_buy} SOL -> {token_symbol} ({frequency.value})"
        )
        
        # Notify user
        if self.notify:
            await self.notify(user_id, "dca_created", {
                'plan_id': config.plan_id,
                'token': token_symbol,
                'amount': str(amount_per_buy),
                'frequency': frequency.value,
                'next_execution': next_exec.isoformat() if next_exec else None
            })
        
        return config
    
    async def pause_plan(self, plan_id: str) -> bool:
        """Pause a DCA plan."""
        if plan_id not in self._plans:
            logger.warning(f"Plan {plan_id} not found")
            return False
        
        config = self._plans[plan_id]
        if config.status != DCAStatus.ACTIVE:
            logger.warning(f"Plan {plan_id} is not active (status: {config.status})")
            return False
        
        config.status = DCAStatus.PAUSED
        config.updated_at = datetime.utcnow()
        
        self.scheduler.clear_schedule(plan_id)
        await self._save_plan(config)
        
        logger.info(f"Paused DCA plan {plan_id}")
        
        if self.notify:
            await self.notify(config.user_id, "dca_paused", {
                'plan_id': plan_id,
                'token': config.token_symbol
            })
        
        return True
    
    async def resume_plan(self, plan_id: str) -> bool:
        """Resume a paused DCA plan."""
        if plan_id not in self._plans:
            logger.warning(f"Plan {plan_id} not found")
            return False
        
        config = self._plans[plan_id]
        if config.status != DCAStatus.PAUSED:
            logger.warning(f"Plan {plan_id} is not paused (status: {config.status})")
            return False
        
        config.status = DCAStatus.ACTIVE
        config.updated_at = datetime.utcnow()
        
        # Recalculate next execution
        next_exec = self.scheduler.calculate_next_execution(config)
        if next_exec:
            self.scheduler.set_next_execution(plan_id, next_exec)
            self._stats[plan_id].next_execution = next_exec
            await self._save_stats(self._stats[plan_id])
        
        await self._save_plan(config)
        
        logger.info(f"Resumed DCA plan {plan_id}, next execution: {next_exec}")
        
        if self.notify:
            await self.notify(config.user_id, "dca_resumed", {
                'plan_id': plan_id,
                'token': config.token_symbol,
                'next_execution': next_exec.isoformat() if next_exec else None
            })
        
        return True
    
    async def cancel_plan(self, plan_id: str) -> bool:
        """Cancel a DCA plan."""
        if plan_id not in self._plans:
            logger.warning(f"Plan {plan_id} not found")
            return False
        
        config = self._plans[plan_id]
        config.status = DCAStatus.CANCELLED
        config.updated_at = datetime.utcnow()
        
        self.scheduler.clear_schedule(plan_id)
        await self._save_plan(config)
        
        logger.info(f"Cancelled DCA plan {plan_id}")
        
        if self.notify:
            await self.notify(config.user_id, "dca_cancelled", {
                'plan_id': plan_id,
                'token': config.token_symbol,
                'stats': self._stats.get(plan_id, DCAStats()).to_dict()
            })
        
        return True
    
    async def modify_plan(
        self,
        plan_id: str,
        amount_per_buy: Optional[Decimal] = None,
        frequency: Optional[DCAFrequency] = None,
        time_of_day: Optional[str] = None,
        total_budget: Optional[Decimal] = None,
        slippage_bps: Optional[int] = None,
        smart_timing: Optional[bool] = None
    ) -> Optional[DCAConfig]:
        """Modify an existing DCA plan."""
        if plan_id not in self._plans:
            logger.warning(f"Plan {plan_id} not found")
            return None
        
        config = self._plans[plan_id]
        
        # Apply changes
        if amount_per_buy is not None:
            config.amount_per_buy = amount_per_buy
        if frequency is not None:
            config.frequency = frequency
        if time_of_day is not None:
            config.time_of_day = time_of_day
        if total_budget is not None:
            config.total_budget = total_budget
        if slippage_bps is not None:
            config.slippage_bps = slippage_bps
        if smart_timing is not None:
            config.smart_timing = smart_timing
        
        config.updated_at = datetime.utcnow()
        
        # Recalculate next execution if schedule changed
        if frequency is not None or time_of_day is not None:
            next_exec = self.scheduler.calculate_next_execution(config)
            if next_exec and config.status == DCAStatus.ACTIVE:
                self.scheduler.set_next_execution(plan_id, next_exec)
                self._stats[plan_id].next_execution = next_exec
                await self._save_stats(self._stats[plan_id])
        
        await self._save_plan(config)
        
        logger.info(f"Modified DCA plan {plan_id}")
        
        return config

    
    async def list_plans(
        self,
        user_id: int,
        status_filter: Optional[DCAStatus] = None,
        include_stats: bool = True
    ) -> List[Dict[str, Any]]:
        """List all DCA plans for a user."""
        plans = []
        
        for plan_id, config in self._plans.items():
            if config.user_id != user_id:
                continue
            if status_filter and config.status != status_filter:
                continue
            
            plan_data = config.to_dict()
            
            if include_stats and plan_id in self._stats:
                plan_data['stats'] = self._stats[plan_id].to_dict()
            
            plans.append(plan_data)
        
        return plans
    
    async def get_plan_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a DCA plan."""
        if plan_id not in self._plans:
            return None
        
        config = self._plans[plan_id]
        stats = self._stats.get(plan_id, DCAStats(plan_id=plan_id))
        
        # Calculate remaining budget
        remaining_budget = None
        if config.total_budget:
            remaining_budget = config.total_budget - stats.total_invested_sol
        
        # Calculate remaining executions
        remaining_executions = None
        if config.max_executions:
            remaining_executions = config.max_executions - stats.execution_count
        
        # Get recent executions
        recent_executions = await self._get_recent_executions(plan_id, limit=5)
        
        return {
            'config': config.to_dict(),
            'stats': stats.to_dict(),
            'remaining_budget_sol': str(remaining_budget) if remaining_budget else None,
            'remaining_executions': remaining_executions,
            'recent_executions': recent_executions,
            'is_active': config.status == DCAStatus.ACTIVE,
            'next_execution': self.scheduler.get_next_execution(plan_id)
        }
    
    async def _get_recent_executions(
        self,
        plan_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent executions for a plan."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT execution_json FROM dca_executions
                WHERE plan_id = ?
                ORDER BY executed_at DESC
                LIMIT ?
            """, (plan_id, limit)) as cursor:
                rows = await cursor.fetchall()
                return [json.loads(row[0]) for row in rows]
    
    # =========================================================================
    # Execution Engine
    # =========================================================================
    
    async def start(self):
        """Start the DCA engine scheduler."""
        if self._running:
            logger.warning("DCA Engine already running")
            return
        
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("DCA Engine started")
    
    async def stop(self):
        """Stop the DCA engine gracefully."""
        self._running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Clear all scheduled tasks
        for plan_id in list(self.scheduler._tasks.keys()):
            self.scheduler.clear_schedule(plan_id)
        
        logger.info("DCA Engine stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        logger.info("DCA scheduler loop started")
        
        while self._running:
            try:
                now = datetime.utcnow()
                plans_to_execute = []
                
                # Check all active plans
                for plan_id, config in self._plans.items():
                    if config.status != DCAStatus.ACTIVE:
                        continue
                    
                    next_exec = self.scheduler.get_next_execution(plan_id)
                    
                    # Initialize schedule if not set
                    if not next_exec:
                        next_exec = self.scheduler.calculate_next_execution(config)
                        if next_exec:
                            self.scheduler.set_next_execution(plan_id, next_exec)
                            self._stats[plan_id].next_execution = next_exec
                        continue
                    
                    # Check if it's time to execute
                    if next_exec <= now:
                        plans_to_execute.append(plan_id)
                
                # Execute due plans (limited concurrency)
                if plans_to_execute:
                    semaphore = asyncio.Semaphore(self.max_concurrent_executions)
                    tasks = []
                    
                    for plan_id in plans_to_execute:
                        task = asyncio.create_task(
                            self._execute_with_semaphore(semaphore, plan_id)
                        )
                        tasks.append(task)
                    
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # Sleep before next check
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(30)
    
    async def _execute_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        plan_id: str
    ):
        """Execute plan with semaphore for concurrency control."""
        async with semaphore:
            await self._execute_plan(plan_id)

    
    async def _execute_plan(self, plan_id: str):
        """Execute a single DCA plan."""
        async with self._plan_locks[plan_id]:
            if plan_id not in self._plans:
                return
            
            config = self._plans[plan_id]
            stats = self._stats.get(plan_id, DCAStats(plan_id=plan_id))
            
            logger.info(f"Executing DCA plan {plan_id} for {config.token_symbol}")
            
            execution = DCAExecution(
                plan_id=plan_id,
                user_id=config.user_id,
                token_mint=config.token_mint,
                amount_sol=config.amount_per_buy
            )
            
            try:
                # Pre-execution checks
                check_result = await self._pre_execution_checks(config, stats)
                if not check_result['passed']:
                    execution.status = 'skipped'
                    execution.error_message = check_result['reason']
                    stats.skipped_count += 1
                    
                    if check_result.get('pause_plan'):
                        await self.pause_plan(plan_id)
                    elif check_result.get('complete_plan'):
                        config.status = DCAStatus.COMPLETED
                        await self._save_plan(config)
                    
                    await self._save_execution(execution)
                    await self._save_stats(stats)
                    
                    # Schedule next execution if not completed
                    if config.status == DCAStatus.ACTIVE:
                        await self._schedule_next_execution(config, stats)
                    
                    return
                
                # Smart timing check
                if config.smart_timing:
                    fee_check = await self._check_network_fees()
                    if fee_check['fees_high']:
                        if config.retry_on_high_fees:
                            logger.info(f"High fees detected, delaying plan {plan_id}")
                            # Delay by configured amount
                            delayed_time = datetime.utcnow() + timedelta(
                                minutes=self.high_fee_delay_minutes
                            )
                            self.scheduler.set_next_execution(plan_id, delayed_time)
                            stats.next_execution = delayed_time
                            
                            execution.status = 'skipped'
                            execution.error_message = 'High network fees, retrying later'
                            stats.skipped_count += 1
                            
                            await self._save_execution(execution)
                            await self._save_stats(stats)
                            return
                
                # Execute the swap
                swap_result = await self._execute_swap(config, execution)
                
                if swap_result['success']:
                    # Update execution record
                    execution.status = 'success'
                    execution.tx_signature = swap_result.get('signature')
                    execution.tokens_received = Decimal(str(swap_result.get('tokens_received', 0)))
                    execution.price_per_token_sol = Decimal(str(swap_result.get('price_per_token', 0)))
                    execution.priority_fee_lamports = swap_result.get('priority_fee', 0)
                    
                    # Update stats
                    stats.execution_count += 1
                    stats.total_invested_sol += config.amount_per_buy
                    stats.total_tokens_acquired += execution.tokens_received
                    stats.total_fees_paid_sol += Decimal(str(execution.network_fee_lamports)) / Decimal("1e9")
                    stats.total_priority_fees_sol += Decimal(str(execution.priority_fee_lamports)) / Decimal("1e9")
                    
                    # Update price tracking
                    if execution.tokens_received > 0:
                        price = config.amount_per_buy / execution.tokens_received
                        if stats.best_price_sol is None or price < stats.best_price_sol:
                            stats.best_price_sol = price
                        if stats.worst_price_sol is None or price > stats.worst_price_sol:
                            stats.worst_price_sol = price
                        
                        # Recalculate average
                        if stats.total_tokens_acquired > 0:
                            stats.average_price_sol = stats.total_invested_sol / stats.total_tokens_acquired
                    
                    if not stats.first_execution:
                        stats.first_execution = execution.executed_at
                    stats.last_execution = execution.executed_at
                    
                    logger.info(
                        f"DCA execution successful: {plan_id} - "
                        f"{config.amount_per_buy} SOL -> {execution.tokens_received} {config.token_symbol}"
                    )
                    
                    # Notify user
                    if self.notify:
                        await self.notify(config.user_id, "dca_executed", {
                            'plan_id': plan_id,
                            'token': config.token_symbol,
                            'amount_sol': str(config.amount_per_buy),
                            'tokens_received': str(execution.tokens_received),
                            'tx_signature': execution.tx_signature,
                            'total_invested': str(stats.total_invested_sol),
                            'total_tokens': str(stats.total_tokens_acquired)
                        })
                else:
                    # Execution failed
                    execution.status = 'failed'
                    execution.error_message = swap_result.get('error', 'Unknown error')
                    execution.retry_count = swap_result.get('retry_count', 0)
                    stats.failed_count += 1
                    
                    logger.error(f"DCA execution failed: {plan_id} - {execution.error_message}")
                    
                    # Check for repeated failures
                    if stats.failed_count >= 3 and stats.failed_count > stats.execution_count:
                        logger.warning(f"Plan {plan_id} has too many failures, pausing")
                        await self.pause_plan(plan_id)
                        
                        if self.notify:
                            await self.notify(config.user_id, "dca_auto_paused", {
                                'plan_id': plan_id,
                                'token': config.token_symbol,
                                'reason': 'Too many consecutive failures',
                                'failed_count': stats.failed_count
                            })
                    else:
                        if self.notify:
                            await self.notify(config.user_id, "dca_failed", {
                                'plan_id': plan_id,
                                'token': config.token_symbol,
                                'error': execution.error_message
                            })
                
                # Save records
                await self._save_execution(execution)
                await self._save_stats(stats)
                
                # Check if plan is completed
                if self._is_plan_completed(config, stats):
                    config.status = DCAStatus.COMPLETED
                    await self._save_plan(config)
                    self.scheduler.clear_schedule(plan_id)
                    
                    if self.notify:
                        await self.notify(config.user_id, "dca_completed", {
                            'plan_id': plan_id,
                            'token': config.token_symbol,
                            'stats': stats.to_dict()
                        })
                else:
                    # Schedule next execution
                    await self._schedule_next_execution(config, stats)
                
            except Exception as e:
                logger.exception(f"Error executing DCA plan {plan_id}: {e}")
                execution.status = 'failed'
                execution.error_message = str(e)
                stats.failed_count += 1
                
                await self._save_execution(execution)
                await self._save_stats(stats)
                
                # Schedule retry
                await self._schedule_next_execution(config, stats)

    
    async def _pre_execution_checks(
        self,
        config: DCAConfig,
        stats: DCAStats
    ) -> Dict[str, Any]:
        """Perform pre-execution checks."""
        result = {'passed': True, 'reason': None}
        
        # Check plan status
        if config.status != DCAStatus.ACTIVE:
            result['passed'] = False
            result['reason'] = f'Plan is not active: {config.status.value}'
            return result
        
        # Check end date
        if config.end_date and datetime.utcnow() > config.end_date:
            result['passed'] = False
            result['reason'] = 'Plan end date reached'
            result['complete_plan'] = True
            return result
        
        # Check max executions
        if config.max_executions and stats.execution_count >= config.max_executions:
            result['passed'] = False
            result['reason'] = 'Maximum executions reached'
            result['complete_plan'] = True
            return result
        
        # Check budget
        if config.total_budget:
            remaining = config.total_budget - stats.total_invested_sol
            if remaining < config.amount_per_buy:
                result['passed'] = False
                result['reason'] = f'Insufficient budget: {remaining} SOL remaining'
                result['complete_plan'] = True
                return result
        
        # Check wallet balance
        if self.wallet_manager:
            try:
                balance = await self.wallet_manager.get_balance(config.user_id)
                if balance < float(config.amount_per_buy) + 0.01:  # Buffer for fees
                    result['passed'] = False
                    result['reason'] = f'Insufficient balance: {balance:.4f} SOL'
                    result['pause_plan'] = True
                    
                    if self.notify:
                        await self.notify(config.user_id, "dca_low_balance", {
                            'plan_id': config.plan_id,
                            'token': config.token_symbol,
                            'balance': balance,
                            'required': float(config.amount_per_buy) + 0.01
                        })
                    return result
            except Exception as e:
                logger.warning(f"Could not check balance: {e}")
        
        # Token safety check
        if config.safety_check_enabled and self.token_analyzer:
            try:
                safety = await self.token_analyzer.analyze(config.token_mint)
                if safety.get('risk_level') == 'critical':
                    result['passed'] = False
                    result['reason'] = f'Token failed safety check: {safety.get("risk_reason")}'
                    result['pause_plan'] = True
                    return result
            except Exception as e:
                logger.warning(f"Could not check token safety: {e}")
        
        return result
    
    async def _check_network_fees(self) -> Dict[str, Any]:
        """Check current network fees."""
        # Placeholder - would integrate with priority fee API
        return {
            'fees_high': False,
            'current_priority_fee': 5000,
            'recommended_fee': 10000
        }
    
    async def _execute_swap(
        self,
        config: DCAConfig,
        execution: DCAExecution
    ) -> Dict[str, Any]:
        """Execute the actual swap transaction."""
        # Placeholder for Jupiter swap integration
        # In production, this would call Jupiter API
        
        if not self.jupiter:
            logger.warning("Jupiter client not configured, simulating execution")
            return {
                'success': True,
                'signature': f'sim_{execution.execution_id[:8]}',
                'tokens_received': float(config.amount_per_buy) * 1000000,  # Simulated
                'price_per_token': float(config.amount_per_buy) / 1000000,
                'priority_fee': 5000
            }
        
        try:
            # Get quote from Jupiter
            quote = await self.jupiter.get_quote(
                input_mint="So11111111111111111111111111111111111111112",  # SOL
                output_mint=config.token_mint,
                amount=int(float(config.amount_per_buy) * 1e9),  # lamports
                slippage_bps=config.slippage_bps
            )
            
            if not quote:
                return {'success': False, 'error': 'Failed to get quote'}
            
            # Execute swap
            result = await self.jupiter.execute_swap(
                user_id=config.user_id,
                quote=quote,
                priority_fee=min(
                    config.max_priority_fee_lamports,
                    quote.get('recommended_priority_fee', 10000)
                )
            )
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _is_plan_completed(self, config: DCAConfig, stats: DCAStats) -> bool:
        """Check if a DCA plan is completed."""
        # Check max executions
        if config.max_executions and stats.execution_count >= config.max_executions:
            return True
        
        # Check budget exhausted
        if config.total_budget:
            remaining = config.total_budget - stats.total_invested_sol
            if remaining < config.amount_per_buy:
                return True
        
        # Check end date
        if config.end_date and datetime.utcnow() > config.end_date:
            return True
        
        return False
    
    async def _schedule_next_execution(self, config: DCAConfig, stats: DCAStats):
        """Schedule the next execution for a plan."""
        if config.status != DCAStatus.ACTIVE:
            return
        
        next_exec = self.scheduler.calculate_next_execution(config)
        if next_exec:
            self.scheduler.set_next_execution(config.plan_id, next_exec)
            stats.next_execution = next_exec
            await self._save_stats(stats)
            logger.debug(f"Scheduled next execution for {config.plan_id}: {next_exec}")

    
    # =========================================================================
    # Analytics & Reporting
    # =========================================================================
    
    async def get_performance_summary(
        self,
        user_id: int,
        plan_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get performance summary for user's DCA plans."""
        if plan_id:
            plans = [self._plans.get(plan_id)] if plan_id in self._plans else []
        else:
            plans = [p for p in self._plans.values() if p.user_id == user_id]
        
        total_invested = Decimal("0")
        total_tokens = {}
        total_fees = Decimal("0")
        
        for plan in plans:
            if not plan:
                continue
            stats = self._stats.get(plan.plan_id)
            if stats:
                total_invested += stats.total_invested_sol
                total_fees += stats.total_fees_paid_sol
                
                if plan.token_symbol not in total_tokens:
                    total_tokens[plan.token_symbol] = {
                        'amount': Decimal("0"),
                        'invested': Decimal("0"),
                        'avg_price': Decimal("0")
                    }
                total_tokens[plan.token_symbol]['amount'] += stats.total_tokens_acquired
                total_tokens[plan.token_symbol]['invested'] += stats.total_invested_sol
        
        # Calculate averages
        for symbol, data in total_tokens.items():
            if data['amount'] > 0:
                data['avg_price'] = data['invested'] / data['amount']
        
        return {
            'total_invested_sol': str(total_invested),
            'total_fees_sol': str(total_fees),
            'tokens_acquired': {
                k: {
                    'amount': str(v['amount']),
                    'invested': str(v['invested']),
                    'avg_price': str(v['avg_price'])
                }
                for k, v in total_tokens.items()
            },
            'active_plans': len([p for p in plans if p and p.status == DCAStatus.ACTIVE]),
            'total_plans': len(plans)
        }
    
    async def compare_to_lump_sum(
        self,
        plan_id: str,
        current_price: Decimal
    ) -> Dict[str, Any]:
        """Compare DCA performance to lump sum investment."""
        if plan_id not in self._plans or plan_id not in self._stats:
            return {'error': 'Plan not found'}
        
        stats = self._stats[plan_id]
        
        if stats.total_invested_sol == 0 or stats.total_tokens_acquired == 0:
            return {'error': 'No executions yet'}
        
        # DCA results
        dca_avg_price = stats.average_price_sol
        dca_tokens = stats.total_tokens_acquired
        dca_value = dca_tokens * current_price
        
        # Hypothetical lump sum (if all invested at first execution price)
        first_price = stats.best_price_sol or dca_avg_price  # Approximation
        lump_sum_tokens = stats.total_invested_sol / first_price if first_price else 0
        lump_sum_value = lump_sum_tokens * current_price
        
        # Calculate advantage
        dca_advantage = dca_value - lump_sum_value
        dca_advantage_percent = (dca_advantage / lump_sum_value * 100) if lump_sum_value else 0
        
        return {
            'dca_average_price': str(dca_avg_price),
            'dca_tokens_acquired': str(dca_tokens),
            'dca_current_value': str(dca_value),
            'lump_sum_tokens': str(lump_sum_tokens),
            'lump_sum_value': str(lump_sum_value),
            'dca_advantage_sol': str(dca_advantage),
            'dca_advantage_percent': f"{dca_advantage_percent:.2f}%",
            'dca_outperformed': dca_advantage > 0
        }
    
    async def get_execution_history(
        self,
        plan_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get execution history for a plan."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT execution_json FROM dca_executions
                WHERE plan_id = ?
                ORDER BY executed_at DESC
                LIMIT ? OFFSET ?
            """, (plan_id, limit, offset)) as cursor:
                rows = await cursor.fetchall()
                return [json.loads(row[0]) for row in rows]
    
    # =========================================================================
    # Manual Execution
    # =========================================================================
    
    async def execute_now(self, plan_id: str) -> Dict[str, Any]:
        """Manually trigger immediate execution of a plan."""
        if plan_id not in self._plans:
            return {'success': False, 'error': 'Plan not found'}
        
        config = self._plans[plan_id]
        
        if config.status not in [DCAStatus.ACTIVE, DCAStatus.PAUSED]:
            return {'success': False, 'error': f'Plan cannot be executed: {config.status.value}'}
        
        # Temporarily activate if paused
        was_paused = config.status == DCAStatus.PAUSED
        if was_paused:
            config.status = DCAStatus.ACTIVE
        
        try:
            await self._execute_plan(plan_id)
            
            # Get latest execution
            executions = await self._get_recent_executions(plan_id, limit=1)
            latest = executions[0] if executions else None
            
            return {
                'success': True,
                'execution': latest
            }
        finally:
            # Restore paused status if needed
            if was_paused:
                config.status = DCAStatus.PAUSED
                await self._save_plan(config)


# Helper function to create engine instance
async def create_dca_engine(
    db_path: str = "data/dca_engine.db",
    jupiter_client: Any = None,
    wallet_manager: Any = None,
    token_analyzer: Any = None,
    notification_callback: Any = None
) -> DCAEngine:
    """Factory function to create and initialize DCA engine."""
    engine = DCAEngine(
        db_path=db_path,
        jupiter_client=jupiter_client,
        wallet_manager=wallet_manager,
        token_analyzer=token_analyzer,
        notification_callback=notification_callback
    )
    await engine.initialize()
    return engine
