"""
Analytics Engine - P&L Tracking and Portfolio Analytics
Production-grade analytics system for Solana trading bot
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from decimal import Decimal
from enum import Enum
import json
import csv
import io
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


                                     

class CostBasisMethod(Enum):
    """Cost basis calculation methods"""
    FIFO = "fifo"                               
    LIFO = "lifo"                              
    AVERAGE = "average"                        
    HIFO = "hifo"                                                    


class TimePeriod(Enum):
    """Time periods for analytics"""
    HOUR = "1h"
    DAY = "24h"
    WEEK = "7d"
    MONTH = "30d"
    QUARTER = "90d"
    YEAR = "365d"
    ALL_TIME = "all"
    
    def to_timedelta(self) -> Optional[timedelta]:
        """Convert to timedelta"""
        mapping = {
            TimePeriod.HOUR: timedelta(hours=1),
            TimePeriod.DAY: timedelta(days=1),
            TimePeriod.WEEK: timedelta(days=7),
            TimePeriod.MONTH: timedelta(days=30),
            TimePeriod.QUARTER: timedelta(days=90),
            TimePeriod.YEAR: timedelta(days=365),
            TimePeriod.ALL_TIME: None
        }
        return mapping.get(self)


class AggregationLevel(Enum):
    """Data aggregation levels"""
    RAW = "raw"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


                                            

@dataclass
class TokenHolding:
    """Individual token holding"""
    token_address: str
    token_symbol: str
    balance: Decimal
    avg_cost_basis: Decimal
    current_price: Decimal
    value_sol: Decimal
    value_usd: Decimal
    unrealized_pnl_sol: Decimal
    unrealized_pnl_percent: Decimal
    first_buy_time: datetime
    last_trade_time: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_address": self.token_address,
            "token_symbol": self.token_symbol,
            "balance": str(self.balance),
            "avg_cost_basis": str(self.avg_cost_basis),
            "current_price": str(self.current_price),
            "value_sol": str(self.value_sol),
            "value_usd": str(self.value_usd),
            "unrealized_pnl_sol": str(self.unrealized_pnl_sol),
            "unrealized_pnl_percent": str(self.unrealized_pnl_percent),
            "first_buy_time": self.first_buy_time.isoformat(),
            "last_trade_time": self.last_trade_time.isoformat()
        }


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio snapshot"""
    timestamp: datetime
    total_value_sol: Decimal
    total_value_usd: Decimal
    sol_balance: Decimal
    token_holdings: List[TokenHolding]
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    total_pnl: Decimal
    num_positions: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_value_sol": str(self.total_value_sol),
            "total_value_usd": str(self.total_value_usd),
            "sol_balance": str(self.sol_balance),
            "token_holdings": [h.to_dict() for h in self.token_holdings],
            "unrealized_pnl": str(self.unrealized_pnl),
            "realized_pnl": str(self.realized_pnl),
            "total_pnl": str(self.total_pnl),
            "num_positions": self.num_positions
        }


@dataclass
class TradeStats:
    """Trading statistics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate: Decimal
    average_win: Decimal
    average_loss: Decimal
    profit_factor: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    average_hold_time: timedelta
    shortest_hold: timedelta
    longest_hold: timedelta
    best_token: Optional[str]
    best_token_pnl: Decimal
    worst_token: Optional[str]
    worst_token_pnl: Decimal
    total_volume_sol: Decimal
    average_trade_size: Decimal
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "breakeven_trades": self.breakeven_trades,
            "win_rate": str(self.win_rate),
            "average_win": str(self.average_win),
            "average_loss": str(self.average_loss),
            "profit_factor": str(self.profit_factor),
            "largest_win": str(self.largest_win),
            "largest_loss": str(self.largest_loss),
            "average_hold_time_seconds": self.average_hold_time.total_seconds(),
            "best_token": self.best_token,
            "best_token_pnl": str(self.best_token_pnl),
            "worst_token": self.worst_token,
            "worst_token_pnl": str(self.worst_token_pnl),
            "total_volume_sol": str(self.total_volume_sol),
            "average_trade_size": str(self.average_trade_size)
        }


@dataclass
class FeeAnalysis:
    """Fee breakdown analysis"""
    platform_fees_sol: Decimal
    platform_fees_usd: Decimal
    network_fees_sol: Decimal
    network_fees_usd: Decimal
    priority_fees_sol: Decimal
    priority_fees_usd: Decimal
    slippage_cost_sol: Decimal
    slippage_cost_usd: Decimal
    total_fees_sol: Decimal
    total_fees_usd: Decimal
    fees_as_percent_of_volume: Decimal
    average_fee_per_trade: Decimal
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: str(v) if isinstance(v, Decimal) else v for k, v in self.__dict__.items()}


@dataclass
class PerformanceMetrics:
    """Advanced performance metrics"""
    roi_percent: Decimal
    roi_sol: Decimal
    cagr_percent: Optional[Decimal]
    sharpe_ratio: Optional[Decimal]
    sortino_ratio: Optional[Decimal]
    max_drawdown_percent: Decimal
    max_drawdown_sol: Decimal
    current_drawdown_percent: Decimal
    recovery_factor: Optional[Decimal]
    calmar_ratio: Optional[Decimal]
    risk_of_ruin: Optional[Decimal]
    avg_daily_return: Decimal
    volatility: Optional[Decimal]
    best_day: Decimal
    worst_day: Decimal
    profitable_days: int
    unprofitable_days: int
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for k, v in self.__dict__.items():
            if v is None:
                result[k] = None
            elif isinstance(v, Decimal):
                result[k] = str(v)
            else:
                result[k] = v
        return result


@dataclass
class PnLSummary:
    """P&L summary for a period"""
    period: str
    start_time: datetime
    end_time: datetime
    starting_balance: Decimal
    ending_balance: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_pnl: Decimal
    pnl_percent: Decimal
    deposits: Decimal
    withdrawals: Decimal
    fees_paid: Decimal
    net_pnl: Decimal
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "starting_balance": str(self.starting_balance),
            "ending_balance": str(self.ending_balance),
            "realized_pnl": str(self.realized_pnl),
            "unrealized_pnl": str(self.unrealized_pnl),
            "total_pnl": str(self.total_pnl),
            "pnl_percent": str(self.pnl_percent),
            "deposits": str(self.deposits),
            "withdrawals": str(self.withdrawals),
            "fees_paid": str(self.fees_paid),
            "net_pnl": str(self.net_pnl)
        }


@dataclass
class TokenPnL:
    """P&L breakdown for a specific token"""
    token_address: str
    token_symbol: str
    total_bought: Decimal
    total_sold: Decimal
    total_buy_value: Decimal
    total_sell_value: Decimal
    remaining_balance: Decimal
    remaining_value: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_pnl: Decimal
    roi_percent: Decimal
    num_buys: int
    num_sells: int
    avg_buy_price: Decimal
    avg_sell_price: Decimal
    first_trade: datetime
    last_trade: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Decimal):
                result[k] = str(v)
            elif isinstance(v, datetime):
                result[k] = v.isoformat()
            else:
                result[k] = v
        return result


@dataclass
class ChartDataPoint:
    """Single data point for charts"""
    timestamp: datetime
    value: Decimal
    label: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": str(self.value),
            "label": self.label
        }


@dataclass
class Alert:
    """Analytics alert"""
    alert_type: str
    title: str
    message: str
    severity: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_type": self.alert_type,
            "title": self.title,
            "message": self.message,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data
        }



                                                  

class CostBasisTracker:
    """Tracks cost basis using various methods (FIFO, LIFO, Average, HIFO)"""
    
    def __init__(self, method: CostBasisMethod = CostBasisMethod.FIFO):
        self.method = method
                                                            
        self.lots: List[Tuple[Decimal, Decimal, datetime]] = []
        self.total_cost = Decimal("0")
        self.total_quantity = Decimal("0")
        
    def add_purchase(self, quantity: Decimal, cost_per_unit: Decimal, timestamp: datetime):
        """Add a purchase lot"""
        self.lots.append((quantity, cost_per_unit, timestamp))
        self.total_cost += quantity * cost_per_unit
        self.total_quantity += quantity
        
                              
        if self.method == CostBasisMethod.LIFO:
            self.lots.sort(key=lambda x: x[2], reverse=True)
        elif self.method == CostBasisMethod.HIFO:
            self.lots.sort(key=lambda x: x[1], reverse=True)
        else:                   
            self.lots.sort(key=lambda x: x[2])
    
    def calculate_sale_cost_basis(self, quantity: Decimal) -> Tuple[Decimal, Decimal]:
        """
        Calculate cost basis for a sale.
        Returns (cost_basis, average_cost_per_unit)
        """
        if self.method == CostBasisMethod.AVERAGE:
            avg_cost = self.total_cost / self.total_quantity if self.total_quantity > 0 else Decimal("0")
            cost_basis = quantity * avg_cost
                           
            self.total_quantity -= quantity
            self.total_cost -= cost_basis
            return cost_basis, avg_cost
        
                                             
        remaining = quantity
        cost_basis = Decimal("0")
        new_lots = []
        
        for lot_qty, lot_cost, lot_time in self.lots:
            if remaining <= 0:
                new_lots.append((lot_qty, lot_cost, lot_time))
                continue
                
            if lot_qty <= remaining:
                                
                cost_basis += lot_qty * lot_cost
                remaining -= lot_qty
                self.total_quantity -= lot_qty
                self.total_cost -= lot_qty * lot_cost
            else:
                                 
                cost_basis += remaining * lot_cost
                new_lot_qty = lot_qty - remaining
                new_lots.append((new_lot_qty, lot_cost, lot_time))
                self.total_quantity -= remaining
                self.total_cost -= remaining * lot_cost
                remaining = Decimal("0")
        
        self.lots = new_lots
        avg_cost = cost_basis / quantity if quantity > 0 else Decimal("0")
        return cost_basis, avg_cost
    
    def get_average_cost(self) -> Decimal:
        """Get current average cost basis"""
        if self.total_quantity <= 0:
            return Decimal("0")
        return self.total_cost / self.total_quantity
    
    def get_unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized P&L at current price"""
        current_value = self.total_quantity * current_price
        return current_value - self.total_cost
    
    def get_quantity(self) -> Decimal:
        """Get total quantity held"""
        return self.total_quantity


                                                    

class UserAnalytics:
    """Per-user analytics state and tracking"""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.cost_basis_method = CostBasisMethod.FIFO
        
                                       
        self.token_trackers: Dict[str, CostBasisTracker] = {}
        
                                            
        self.snapshots: List[PortfolioSnapshot] = []
        self.daily_snapshots: List[PortfolioSnapshot] = []
        self.weekly_snapshots: List[PortfolioSnapshot] = []
        
                                     
        self.trades: List[Dict[str, Any]] = []
        
                              
        self.realized_pnl_records: List[Dict[str, Any]] = []
        
                     
        self.fee_records: List[Dict[str, Any]] = []
        
                        
        self.total_realized_pnl = Decimal("0")
        self.total_fees_paid = Decimal("0")
        self.total_volume = Decimal("0")
        
                                              
        self.initial_balance: Optional[Decimal] = None
        self.initial_balance_time: Optional[datetime] = None
        
                                             
        self.peak_value = Decimal("0")
        self.peak_time: Optional[datetime] = None
        
                               
        self.last_update = datetime.utcnow()
    
    def get_or_create_tracker(self, token_address: str) -> CostBasisTracker:
        """Get or create cost basis tracker for a token"""
        if token_address not in self.token_trackers:
            self.token_trackers[token_address] = CostBasisTracker(self.cost_basis_method)
        return self.token_trackers[token_address]
    
    def set_cost_basis_method(self, method: CostBasisMethod):
        """Update cost basis method for all trackers"""
        self.cost_basis_method = method
        for tracker in self.token_trackers.values():
            tracker.method = method



                                                

class AnalyticsEngine:
    """
    Production-grade analytics engine for trading bot.
    Handles P&L tracking, portfolio analytics, and reporting.
    """
    
    def __init__(self, database=None, price_fetcher=None):
        """
        Initialize analytics engine.
        
        Args:
            database: Database instance for persistence
            price_fetcher: Async function to fetch current prices
        """
        self.db = database
        self.price_fetcher = price_fetcher
        
                                  
        self.user_analytics: Dict[int, UserAnalytics] = {}
        
                             
        self.sol_price_usd = Decimal("150")                              
        
                                
        self.snapshot_interval = timedelta(hours=1)
        self.daily_snapshot_hour = 0                
        
                               
        self.raw_retention_days = 30
        self.hourly_retention_days = 90
        self.daily_retention_days = 365
        
                         
        self.alert_callbacks: List[Callable[[int, Alert], None]] = []
        
                          
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        logger.info("Analytics engine initialized")
    
                                             
    
    async def start(self):
        """Start analytics engine background tasks"""
        self._running = True
        self._tasks = [
            asyncio.create_task(self._snapshot_loop()),
            asyncio.create_task(self._aggregation_loop()),
            asyncio.create_task(self._sol_price_loop())
        ]
        logger.info("Analytics engine started")
    
    async def stop(self):
        """Stop analytics engine"""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("Analytics engine stopped")
    
    def get_user_analytics(self, user_id: int) -> UserAnalytics:
        """Get or create user analytics state"""
        if user_id not in self.user_analytics:
            self.user_analytics[user_id] = UserAnalytics(user_id)
        return self.user_analytics[user_id]
    
    async def set_cost_basis_method(self, user_id: int, method: CostBasisMethod):
        """Set cost basis calculation method for user"""
        ua = self.get_user_analytics(user_id)
        ua.set_cost_basis_method(method)
        logger.info(f"User {user_id} cost basis method set to {method.value}")
    
                                                   
    
    async def record_trade(
        self,
        user_id: int,
        trade_type: str,                           
        token_address: str,
        token_symbol: str,
        amount: Decimal,
        price_sol: Decimal,
        total_sol: Decimal,
        fees: Dict[str, Decimal],
        tx_signature: str,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Record a trade and update analytics.
        
        Returns dict with realized P&L info for sells.
        """
        ua = self.get_user_analytics(user_id)
        timestamp = timestamp or datetime.utcnow()
        
        result = {
            "trade_type": trade_type,
            "token_address": token_address,
            "token_symbol": token_symbol,
            "amount": str(amount),
            "price_sol": str(price_sol),
            "total_sol": str(total_sol),
            "realized_pnl": None,
            "cost_basis": None
        }
        
        tracker = ua.get_or_create_tracker(token_address)
        
        if trade_type == "buy":
                               
            tracker.add_purchase(amount, price_sol, timestamp)
            
        elif trade_type == "sell":
                                    
            cost_basis, avg_cost = tracker.calculate_sale_cost_basis(amount)
            realized_pnl = total_sol - cost_basis
            
            result["realized_pnl"] = str(realized_pnl)
            result["cost_basis"] = str(cost_basis)
            
                                 
            ua.realized_pnl_records.append({
                "timestamp": timestamp,
                "token_address": token_address,
                "token_symbol": token_symbol,
                "amount_sold": amount,
                "sale_value": total_sol,
                "cost_basis": cost_basis,
                "realized_pnl": realized_pnl
            })
            ua.total_realized_pnl += realized_pnl
        
                     
        total_fees = sum(fees.values())
        ua.fee_records.append({
            "timestamp": timestamp,
            "tx_signature": tx_signature,
            "platform_fee": fees.get("platform", Decimal("0")),
            "network_fee": fees.get("network", Decimal("0")),
            "priority_fee": fees.get("priority", Decimal("0")),
            "total_fee": total_fees
        })
        ua.total_fees_paid += total_fees
        
                                    
        trade_record = {
            "timestamp": timestamp,
            "trade_type": trade_type,
            "token_address": token_address,
            "token_symbol": token_symbol,
            "amount": amount,
            "price_sol": price_sol,
            "total_sol": total_sol,
            "fees": total_fees,
            "tx_signature": tx_signature
        }
        ua.trades.append(trade_record)
        ua.total_volume += total_sol
        ua.last_update = timestamp
        
                                               
        if ua.initial_balance is None:
            await self._initialize_user_balance(user_id)
        
        logger.debug(f"Recorded {trade_type} for user {user_id}: {amount} {token_symbol}")
        
        return result
    
    async def _initialize_user_balance(self, user_id: int):
        """Initialize user's starting balance for ROI calculation"""
        ua = self.get_user_analytics(user_id)
        if self.db:
            try:
                wallet = await self.db.get_user_wallet(user_id)
                if wallet:
                    ua.initial_balance = Decimal(str(wallet.get("sol_balance", 0)))
                    ua.initial_balance_time = datetime.utcnow()
            except Exception as e:
                logger.error(f"Failed to initialize balance for user {user_id}: {e}")


                                                      
    
    async def get_portfolio(self, user_id: int) -> PortfolioSnapshot:
        """Get current portfolio state"""
        ua = self.get_user_analytics(user_id)
        now = datetime.utcnow()
        
        holdings = []
        total_token_value_sol = Decimal("0")
        total_unrealized_pnl = Decimal("0")
        
                                                 
        positions = []
        if self.db:
            try:
                positions = await self.db.get_user_positions(user_id)
            except Exception:
                pass
        
                                              
        if not positions:
            for token_addr, tracker in ua.token_trackers.items():
                if tracker.get_quantity() > 0:
                    positions.append({
                        "token_address": token_addr,
                        "token_symbol": "UNKNOWN",
                        "balance": tracker.get_quantity(),
                        "first_buy_time": now,
                        "last_trade_time": now
                    })
        
        for pos in positions:
            token_address = pos.get("token_address", "")
            token_symbol = pos.get("token_symbol", "UNKNOWN")
            balance = Decimal(str(pos.get("balance", 0)))
            
            if balance <= 0:
                continue
            
                               
            current_price = Decimal("0")
            if self.price_fetcher:
                try:
                    price_data = await self.price_fetcher(token_address)
                    current_price = Decimal(str(price_data.get("price_sol", 0)))
                except Exception:
                    pass
            
                                         
            tracker = ua.token_trackers.get(token_address)
            avg_cost = tracker.get_average_cost() if tracker else Decimal("0")
            
                              
            value_sol = balance * current_price
            value_usd = value_sol * self.sol_price_usd
            cost_value = balance * avg_cost
            unrealized_pnl = value_sol - cost_value
            pnl_percent = ((current_price - avg_cost) / avg_cost * 100) if avg_cost > 0 else Decimal("0")
            
            holding = TokenHolding(
                token_address=token_address,
                token_symbol=token_symbol,
                balance=balance,
                avg_cost_basis=avg_cost,
                current_price=current_price,
                value_sol=value_sol,
                value_usd=value_usd,
                unrealized_pnl_sol=unrealized_pnl,
                unrealized_pnl_percent=pnl_percent,
                first_buy_time=pos.get("first_buy_time", now),
                last_trade_time=pos.get("last_trade_time", now)
            )
            holdings.append(holding)
            total_token_value_sol += value_sol
            total_unrealized_pnl += unrealized_pnl
        
                         
        sol_balance = Decimal("0")
        if self.db:
            try:
                wallet = await self.db.get_user_wallet(user_id)
                if wallet:
                    sol_balance = Decimal(str(wallet.get("sol_balance", 0)))
            except Exception:
                pass
        
        total_value_sol = sol_balance + total_token_value_sol
        total_value_usd = total_value_sol * self.sol_price_usd
        
        snapshot = PortfolioSnapshot(
            timestamp=now,
            total_value_sol=total_value_sol,
            total_value_usd=total_value_usd,
            sol_balance=sol_balance,
            token_holdings=holdings,
            unrealized_pnl=total_unrealized_pnl,
            realized_pnl=ua.total_realized_pnl,
            total_pnl=total_unrealized_pnl + ua.total_realized_pnl,
            num_positions=len(holdings)
        )
        
                                                    
        if total_value_sol > ua.peak_value:
            ua.peak_value = total_value_sol
            ua.peak_time = now
        
        return snapshot
    
    async def get_portfolio_history(
        self,
        user_id: int,
        period: TimePeriod = TimePeriod.WEEK
    ) -> List[PortfolioSnapshot]:
        """Get historical portfolio snapshots"""
        ua = self.get_user_analytics(user_id)
        
        delta = period.to_timedelta()
        cutoff = datetime.utcnow() - delta if delta else datetime.min
        
                                                          
        if period in [TimePeriod.HOUR, TimePeriod.DAY]:
            snapshots = ua.snapshots
        elif period in [TimePeriod.WEEK, TimePeriod.MONTH]:
            snapshots = ua.daily_snapshots if ua.daily_snapshots else ua.snapshots
        else:
            snapshots = ua.weekly_snapshots if ua.weekly_snapshots else ua.daily_snapshots
        
                        
        filtered = [s for s in snapshots if s.timestamp >= cutoff]
        
        return filtered


                                                    
    
    async def calculate_pnl(
        self,
        user_id: int,
        period: TimePeriod = TimePeriod.DAY
    ) -> PnLSummary:
        """Calculate P&L for a period"""
        ua = self.get_user_analytics(user_id)
        now = datetime.utcnow()
        delta = period.to_timedelta()
        start_time = now - delta if delta else (ua.initial_balance_time or now)
        
                               
        current_portfolio = await self.get_portfolio(user_id)
        
                                                        
        starting_balance = ua.initial_balance or Decimal("0")
        historical = await self.get_portfolio_history(user_id, period)
        if historical:
            starting_balance = historical[0].total_value_sol
        
                                          
        realized_in_period = sum(
            r["realized_pnl"]
            for r in ua.realized_pnl_records
            if r["timestamp"] >= start_time
        )
        
                                  
        fees_in_period = sum(
            f["total_fee"]
            for f in ua.fee_records
            if f["timestamp"] >= start_time
        )
        
                                                                  
        deposits = Decimal("0")
        withdrawals = Decimal("0")
        
        total_pnl = current_portfolio.total_value_sol - starting_balance
        pnl_percent = (total_pnl / starting_balance * 100) if starting_balance > 0 else Decimal("0")
        
        return PnLSummary(
            period=period.value,
            start_time=start_time,
            end_time=now,
            starting_balance=starting_balance,
            ending_balance=current_portfolio.total_value_sol,
            realized_pnl=realized_in_period,
            unrealized_pnl=current_portfolio.unrealized_pnl,
            total_pnl=total_pnl,
            pnl_percent=pnl_percent,
            deposits=deposits,
            withdrawals=withdrawals,
            fees_paid=fees_in_period,
            net_pnl=total_pnl - fees_in_period
        )
    
    async def get_realized_pnl(self, user_id: int) -> Decimal:
        """Get total realized P&L"""
        ua = self.get_user_analytics(user_id)
        return ua.total_realized_pnl
    
    async def get_unrealized_pnl(self, user_id: int) -> Decimal:
        """Get total unrealized P&L"""
        portfolio = await self.get_portfolio(user_id)
        return portfolio.unrealized_pnl
    
    async def get_pnl_by_token(self, user_id: int) -> List[TokenPnL]:
        """Get P&L breakdown by token"""
        ua = self.get_user_analytics(user_id)
        token_pnls = []
        
                                   
        token_data: Dict[str, Dict] = defaultdict(lambda: {
            "buys": [], "sells": [], "symbol": "UNKNOWN"
        })
        
        for trade in ua.trades:
            token_addr = trade["token_address"]
            token_data[token_addr]["symbol"] = trade["token_symbol"]
            if trade["trade_type"] == "buy":
                token_data[token_addr]["buys"].append(trade)
            else:
                token_data[token_addr]["sells"].append(trade)
        
        for token_address, data in token_data.items():
            buys = data["buys"]
            sells = data["sells"]
            
            total_bought = sum(t["amount"] for t in buys)
            total_sold = sum(t["amount"] for t in sells)
            total_buy_value = sum(t["total_sol"] for t in buys)
            total_sell_value = sum(t["total_sol"] for t in sells)
            
            remaining_balance = total_bought - total_sold
            
                                                     
            current_price = Decimal("0")
            if remaining_balance > 0 and self.price_fetcher:
                try:
                    price_data = await self.price_fetcher(token_address)
                    current_price = Decimal(str(price_data.get("price_sol", 0)))
                except Exception:
                    pass
            
            remaining_value = remaining_balance * current_price
            
                           
            avg_buy_price = total_buy_value / total_bought if total_bought > 0 else Decimal("0")
            avg_sell_price = total_sell_value / total_sold if total_sold > 0 else Decimal("0")
            
                                     
            realized_cost = total_sold * avg_buy_price if total_bought > 0 else Decimal("0")
            realized_pnl = total_sell_value - realized_cost
            
                                         
            unrealized_cost = remaining_balance * avg_buy_price
            unrealized_pnl = remaining_value - unrealized_cost
            
            total_pnl = realized_pnl + unrealized_pnl
            roi = (total_pnl / total_buy_value * 100) if total_buy_value > 0 else Decimal("0")
            
            all_trades = buys + sells
            first_trade = min(t["timestamp"] for t in all_trades) if all_trades else datetime.utcnow()
            last_trade = max(t["timestamp"] for t in all_trades) if all_trades else datetime.utcnow()
            
            token_pnls.append(TokenPnL(
                token_address=token_address,
                token_symbol=data["symbol"],
                total_bought=total_bought,
                total_sold=total_sold,
                total_buy_value=total_buy_value,
                total_sell_value=total_sell_value,
                remaining_balance=remaining_balance,
                remaining_value=remaining_value,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                total_pnl=total_pnl,
                roi_percent=roi,
                num_buys=len(buys),
                num_sells=len(sells),
                avg_buy_price=avg_buy_price,
                avg_sell_price=avg_sell_price,
                first_trade=first_trade,
                last_trade=last_trade
            ))
        
                                      
        token_pnls.sort(key=lambda x: x.total_pnl, reverse=True)
        
        return token_pnls


                                                   
    
    async def get_trade_stats(
        self,
        user_id: int,
        period: TimePeriod = TimePeriod.ALL_TIME
    ) -> TradeStats:
        """Get trading statistics"""
        ua = self.get_user_analytics(user_id)
        now = datetime.utcnow()
        delta = period.to_timedelta()
        cutoff = now - delta if delta else datetime.min
        
                                 
        trades = [t for t in ua.trades if t["timestamp"] >= cutoff]
        
        if not trades:
            return self._empty_trade_stats()
        
                                                 
        completed_trades = []
        token_buys: Dict[str, List] = defaultdict(list)
        
        for trade in sorted(trades, key=lambda t: t["timestamp"]):
            token = trade["token_address"]
            if trade["trade_type"] == "buy":
                token_buys[token].append(trade)
            elif trade["trade_type"] == "sell" and token_buys[token]:
                buy = token_buys[token].pop(0)
                sell = trade
                
                hold_time = sell["timestamp"] - buy["timestamp"]
                pnl = sell["total_sol"] - buy["total_sol"]
                
                completed_trades.append({
                    "token_address": token,
                    "token_symbol": trade["token_symbol"],
                    "buy_time": buy["timestamp"],
                    "sell_time": sell["timestamp"],
                    "hold_time": hold_time,
                    "buy_value": buy["total_sol"],
                    "sell_value": sell["total_sol"],
                    "pnl": pnl,
                    "pnl_percent": (pnl / buy["total_sol"] * 100) if buy["total_sol"] > 0 else Decimal("0")
                })
        
        if not completed_trades:
            return self._empty_trade_stats()
        
                              
        wins = [t for t in completed_trades if t["pnl"] > 0]
        losses = [t for t in completed_trades if t["pnl"] < 0]
        breakeven = [t for t in completed_trades if t["pnl"] == 0]
        
        win_rate = Decimal(len(wins)) / Decimal(len(completed_trades)) * 100
        avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else Decimal("0")
        avg_loss = abs(sum(t["pnl"] for t in losses) / len(losses)) if losses else Decimal("0")
        
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else Decimal("999")
        
        hold_times = [t["hold_time"] for t in completed_trades]
        avg_hold = sum(hold_times, timedelta()) / len(hold_times)
        
                       
        token_pnl: Dict[str, Decimal] = defaultdict(Decimal)
        for t in completed_trades:
            token_pnl[t["token_symbol"]] += t["pnl"]
        
        best_token = max(token_pnl.keys(), key=lambda k: token_pnl[k]) if token_pnl else None
        worst_token = min(token_pnl.keys(), key=lambda k: token_pnl[k]) if token_pnl else None
        
        pnls = [t["pnl"] for t in completed_trades]
        total_volume = sum(t["buy_value"] + t["sell_value"] for t in completed_trades)
        
        return TradeStats(
            total_trades=len(completed_trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            breakeven_trades=len(breakeven),
            win_rate=win_rate,
            average_win=avg_win,
            average_loss=avg_loss,
            profit_factor=profit_factor,
            largest_win=max(pnls) if pnls else Decimal("0"),
            largest_loss=min(pnls) if pnls else Decimal("0"),
            average_hold_time=avg_hold,
            shortest_hold=min(hold_times) if hold_times else timedelta(),
            longest_hold=max(hold_times) if hold_times else timedelta(),
            best_token=best_token,
            best_token_pnl=token_pnl.get(best_token, Decimal("0")) if best_token else Decimal("0"),
            worst_token=worst_token,
            worst_token_pnl=token_pnl.get(worst_token, Decimal("0")) if worst_token else Decimal("0"),
            total_volume_sol=total_volume,
            average_trade_size=total_volume / len(completed_trades) / 2
        )
    
    def _empty_trade_stats(self) -> TradeStats:
        """Return empty trade stats"""
        return TradeStats(
            total_trades=0, winning_trades=0, losing_trades=0, breakeven_trades=0,
            win_rate=Decimal("0"), average_win=Decimal("0"), average_loss=Decimal("0"),
            profit_factor=Decimal("0"), largest_win=Decimal("0"), largest_loss=Decimal("0"),
            average_hold_time=timedelta(), shortest_hold=timedelta(), longest_hold=timedelta(),
            best_token=None, best_token_pnl=Decimal("0"),
            worst_token=None, worst_token_pnl=Decimal("0"),
            total_volume_sol=Decimal("0"), average_trade_size=Decimal("0")
        )
    
                                                
    
    async def get_fee_analysis(
        self,
        user_id: int,
        period: TimePeriod = TimePeriod.ALL_TIME
    ) -> FeeAnalysis:
        """Get detailed fee analysis"""
        ua = self.get_user_analytics(user_id)
        now = datetime.utcnow()
        delta = period.to_timedelta()
        cutoff = now - delta if delta else datetime.min
        
        fees = [f for f in ua.fee_records if f["timestamp"] >= cutoff]
        
        if not fees:
            return FeeAnalysis(
                platform_fees_sol=Decimal("0"), platform_fees_usd=Decimal("0"),
                network_fees_sol=Decimal("0"), network_fees_usd=Decimal("0"),
                priority_fees_sol=Decimal("0"), priority_fees_usd=Decimal("0"),
                slippage_cost_sol=Decimal("0"), slippage_cost_usd=Decimal("0"),
                total_fees_sol=Decimal("0"), total_fees_usd=Decimal("0"),
                fees_as_percent_of_volume=Decimal("0"), average_fee_per_trade=Decimal("0")
            )
        
        platform_fees = sum(f["platform_fee"] for f in fees)
        network_fees = sum(f["network_fee"] for f in fees)
        priority_fees = sum(f["priority_fee"] for f in fees)
        total_fees = sum(f["total_fee"] for f in fees)
        slippage = Decimal("0")                                       
        
        trades = [t for t in ua.trades if t["timestamp"] >= cutoff]
        volume = sum(t["total_sol"] for t in trades)
        
        fees_percent = (total_fees / volume * 100) if volume > 0 else Decimal("0")
        avg_fee = total_fees / len(fees) if fees else Decimal("0")
        
        return FeeAnalysis(
            platform_fees_sol=platform_fees,
            platform_fees_usd=platform_fees * self.sol_price_usd,
            network_fees_sol=network_fees,
            network_fees_usd=network_fees * self.sol_price_usd,
            priority_fees_sol=priority_fees,
            priority_fees_usd=priority_fees * self.sol_price_usd,
            slippage_cost_sol=slippage,
            slippage_cost_usd=slippage * self.sol_price_usd,
            total_fees_sol=total_fees,
            total_fees_usd=total_fees * self.sol_price_usd,
            fees_as_percent_of_volume=fees_percent,
            average_fee_per_trade=avg_fee
        )


                                                       
    
    async def get_performance_metrics(
        self,
        user_id: int,
        period: TimePeriod = TimePeriod.ALL_TIME
    ) -> PerformanceMetrics:
        """Calculate advanced performance metrics"""
        ua = self.get_user_analytics(user_id)
        now = datetime.utcnow()
        
        history = await self.get_portfolio_history(user_id, period)
        current = await self.get_portfolio(user_id)
        
        if not history:
            history = [current]
        
        initial_value = ua.initial_balance or history[0].total_value_sol
        current_value = current.total_value_sol
        
                   
        roi_sol = current_value - initial_value
        roi_percent = (roi_sol / initial_value * 100) if initial_value > 0 else Decimal("0")
        
                                                      
        daily_returns = []
        for i in range(1, len(history)):
            prev_val = history[i-1].total_value_sol
            curr_val = history[i].total_value_sol
            if prev_val > 0:
                daily_return = float((curr_val - prev_val) / prev_val)
                daily_returns.append(daily_return)
        
                                            
        cagr = None
        if ua.initial_balance_time:
            days_elapsed = (now - ua.initial_balance_time).days
            if days_elapsed > 0 and initial_value > 0:
                years = Decimal(days_elapsed) / Decimal(365)
                if years > 0 and current_value > 0:
                    ratio = float(current_value / initial_value)
                    if ratio > 0:
                        cagr = Decimal(str((ratio ** (1/float(years)) - 1) * 100))
        
                                            
        sharpe = None
        sortino = None
        volatility = None
        
        if len(daily_returns) >= 7:
            try:
                vol = statistics.stdev(daily_returns)
                volatility = Decimal(str(vol * 100))
                avg_return = statistics.mean(daily_returns)
                
                                                           
                if vol > 0:
                    sharpe = Decimal(str((avg_return / vol) * (252 ** 0.5)))
                
                                                    
                negative_returns = [r for r in daily_returns if r < 0]
                if negative_returns:
                    downside_dev = statistics.stdev(negative_returns)
                    if downside_dev > 0:
                        sortino = Decimal(str((avg_return / downside_dev) * (252 ** 0.5)))
            except Exception:
                pass
        
                               
        peak = initial_value
        max_drawdown_sol = Decimal("0")
        max_drawdown_percent = Decimal("0")
        
        for snapshot in history:
            if snapshot.total_value_sol > peak:
                peak = snapshot.total_value_sol
            
            drawdown = peak - snapshot.total_value_sol
            drawdown_percent = (drawdown / peak * 100) if peak > 0 else Decimal("0")
            
            if drawdown > max_drawdown_sol:
                max_drawdown_sol = drawdown
                max_drawdown_percent = drawdown_percent
        
        current_dd = (ua.peak_value - current_value) / ua.peak_value * 100 if ua.peak_value > 0 else Decimal("0")
        
                                          
        recovery_factor = None
        calmar = None
        if max_drawdown_sol > 0:
            recovery_factor = roi_sol / max_drawdown_sol
            if cagr and max_drawdown_percent > 0:
                calmar = cagr / max_drawdown_percent
        
                     
        profitable_days = sum(1 for r in daily_returns if r > 0)
        unprofitable_days = sum(1 for r in daily_returns if r < 0)
        avg_daily = Decimal(str(statistics.mean(daily_returns) * 100)) if daily_returns else Decimal("0")
        best_day = Decimal(str(max(daily_returns) * 100)) if daily_returns else Decimal("0")
        worst_day = Decimal(str(min(daily_returns) * 100)) if daily_returns else Decimal("0")
        
        return PerformanceMetrics(
            roi_percent=roi_percent,
            roi_sol=roi_sol,
            cagr_percent=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_percent=max_drawdown_percent,
            max_drawdown_sol=max_drawdown_sol,
            current_drawdown_percent=current_dd,
            recovery_factor=recovery_factor,
            calmar_ratio=calmar,
            risk_of_ruin=None,
            avg_daily_return=avg_daily,
            volatility=volatility,
            best_day=best_day,
            worst_day=worst_day,
            profitable_days=profitable_days,
            unprofitable_days=unprofitable_days
        )


                                               
    
    async def get_equity_curve(
        self,
        user_id: int,
        period: TimePeriod = TimePeriod.MONTH
    ) -> List[ChartDataPoint]:
        """Get equity curve data for charting"""
        history = await self.get_portfolio_history(user_id, period)
        
        return [
            ChartDataPoint(
                timestamp=s.timestamp,
                value=s.total_value_sol,
                label=f"{s.total_value_sol:.4f} SOL"
            )
            for s in history
        ]
    
    async def get_pnl_chart(
        self,
        user_id: int,
        period: TimePeriod = TimePeriod.MONTH
    ) -> List[ChartDataPoint]:
        """Get P&L over time for charting"""
        history = await self.get_portfolio_history(user_id, period)
        
        return [
            ChartDataPoint(
                timestamp=s.timestamp,
                value=s.total_pnl,
                label=f"P&L: {s.total_pnl:+.4f} SOL"
            )
            for s in history
        ]
    
    async def get_token_allocation(self, user_id: int) -> List[Dict[str, Any]]:
        """Get token allocation for pie chart"""
        portfolio = await self.get_portfolio(user_id)
        allocations = []
        
                 
        if portfolio.sol_balance > 0:
            sol_pct = (portfolio.sol_balance / portfolio.total_value_sol * 100) if portfolio.total_value_sol > 0 else Decimal("0")
            allocations.append({
                "token": "SOL",
                "value_sol": str(portfolio.sol_balance),
                "percent": str(sol_pct)
            })
        
                            
        for holding in portfolio.token_holdings:
            pct = (holding.value_sol / portfolio.total_value_sol * 100) if portfolio.total_value_sol > 0 else Decimal("0")
            allocations.append({
                "token": holding.token_symbol,
                "address": holding.token_address,
                "value_sol": str(holding.value_sol),
                "percent": str(pct)
            })
        
        return allocations
    
    async def get_trade_distribution(
        self,
        user_id: int,
        period: TimePeriod = TimePeriod.ALL_TIME
    ) -> Dict[str, Any]:
        """Get trade P&L distribution for histogram"""
        ua = self.get_user_analytics(user_id)
        
        pnls = []
        for record in ua.realized_pnl_records:
            pnl_pct = float(record["realized_pnl"] / record["cost_basis"] * 100) if record["cost_basis"] > 0 else 0
            pnls.append(pnl_pct)
        
        if not pnls:
            return {"bins": [], "counts": [], "avg_pnl_percent": 0, "median_pnl_percent": 0}
        
                               
        min_pnl = min(pnls)
        max_pnl = max(pnls)
        num_bins = 20
        bin_width = (max_pnl - min_pnl) / num_bins if max_pnl != min_pnl else 1
        
        bins = []
        counts = []
        for i in range(num_bins):
            bin_start = min_pnl + i * bin_width
            bin_end = bin_start + bin_width
            count = sum(1 for p in pnls if bin_start <= p < bin_end)
            bins.append(f"{bin_start:.1f}% to {bin_end:.1f}%")
            counts.append(count)
        
        return {
            "bins": bins,
            "counts": counts,
            "avg_pnl_percent": sum(pnls) / len(pnls) if pnls else 0,
            "median_pnl_percent": statistics.median(pnls) if pnls else 0
        }


                                                    
    
    async def export_trades_csv(
        self,
        user_id: int,
        period: TimePeriod = TimePeriod.ALL_TIME
    ) -> str:
        """Export trade history to CSV"""
        ua = self.get_user_analytics(user_id)
        now = datetime.utcnow()
        delta = period.to_timedelta()
        cutoff = now - delta if delta else datetime.min
        
        trades = [t for t in ua.trades if t["timestamp"] >= cutoff]
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        writer.writerow([
            "Timestamp", "Type", "Token Symbol", "Token Address",
            "Amount", "Price (SOL)", "Total (SOL)", "Fees (SOL)", "Transaction"
        ])
        
        for trade in trades:
            writer.writerow([
                trade["timestamp"].isoformat(),
                trade["trade_type"].upper(),
                trade["token_symbol"],
                trade["token_address"],
                str(trade["amount"]),
                str(trade["price_sol"]),
                str(trade["total_sol"]),
                str(trade["fees"]),
                trade["tx_signature"]
            ])
        
        return output.getvalue()
    
    async def export_portfolio_csv(self, user_id: int) -> str:
        """Export current holdings to CSV"""
        portfolio = await self.get_portfolio(user_id)
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        writer.writerow([
            "Token Symbol", "Token Address", "Balance",
            "Avg Cost (SOL)", "Current Price (SOL)",
            "Value (SOL)", "Value (USD)",
            "Unrealized P&L (SOL)", "Unrealized P&L (%)"
        ])
        
                     
        writer.writerow([
            "SOL", "Native", str(portfolio.sol_balance),
            "1.0", "1.0",
            str(portfolio.sol_balance), str(portfolio.sol_balance * self.sol_price_usd),
            "0", "0"
        ])
        
        for holding in portfolio.token_holdings:
            writer.writerow([
                holding.token_symbol,
                holding.token_address,
                str(holding.balance),
                str(holding.avg_cost_basis),
                str(holding.current_price),
                str(holding.value_sol),
                str(holding.value_usd),
                str(holding.unrealized_pnl_sol),
                str(holding.unrealized_pnl_percent)
            ])
        
        return output.getvalue()
    
    async def export_pnl_report(
        self,
        user_id: int,
        period: TimePeriod = TimePeriod.MONTH
    ) -> str:
        """Generate detailed P&L report"""
        pnl = await self.calculate_pnl(user_id, period)
        stats = await self.get_trade_stats(user_id, period)
        fees = await self.get_fee_analysis(user_id, period)
        metrics = await self.get_performance_metrics(user_id, period)
        token_pnls = await self.get_pnl_by_token(user_id)
        
        lines = []
        lines.append("=" * 60)
        lines.append("TRADING P&L REPORT")
        lines.append(f"Period: {pnl.period}")
        lines.append(f"Generated: {datetime.utcnow().isoformat()}")
        lines.append("=" * 60)
        lines.append("")
        
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Starting Balance:    {pnl.starting_balance:>12.4f} SOL")
        lines.append(f"Ending Balance:      {pnl.ending_balance:>12.4f} SOL")
        lines.append(f"Total P&L:           {pnl.total_pnl:>+12.4f} SOL ({pnl.pnl_percent:+.2f}%)")
        lines.append(f"  Realized:          {pnl.realized_pnl:>+12.4f} SOL")
        lines.append(f"  Unrealized:        {pnl.unrealized_pnl:>+12.4f} SOL")
        lines.append(f"Fees Paid:           {pnl.fees_paid:>12.4f} SOL")
        lines.append(f"Net P&L:             {pnl.net_pnl:>+12.4f} SOL")
        lines.append("")
        
        lines.append("TRADING STATISTICS")
        lines.append("-" * 40)
        lines.append(f"Total Trades:        {stats.total_trades:>12}")
        lines.append(f"Winning Trades:      {stats.winning_trades:>12}")
        lines.append(f"Losing Trades:       {stats.losing_trades:>12}")
        lines.append(f"Win Rate:            {stats.win_rate:>12.2f}%")
        lines.append(f"Profit Factor:       {stats.profit_factor:>12.2f}")
        lines.append(f"Avg Win:             {stats.average_win:>+12.4f} SOL")
        lines.append(f"Avg Loss:            {stats.average_loss:>12.4f} SOL")
        lines.append(f"Largest Win:         {stats.largest_win:>+12.4f} SOL")
        lines.append(f"Largest Loss:        {stats.largest_loss:>12.4f} SOL")
        lines.append(f"Total Volume:        {stats.total_volume_sol:>12.4f} SOL")
        lines.append("")
        
        lines.append("PERFORMANCE METRICS")
        lines.append("-" * 40)
        lines.append(f"ROI:                 {metrics.roi_percent:>+12.2f}%")
        if metrics.cagr_percent:
            lines.append(f"CAGR:                {metrics.cagr_percent:>+12.2f}%")
        if metrics.sharpe_ratio:
            lines.append(f"Sharpe Ratio:        {metrics.sharpe_ratio:>12.2f}")
        lines.append(f"Max Drawdown:        {metrics.max_drawdown_percent:>12.2f}%")
        lines.append(f"Best Day:            {metrics.best_day:>+12.2f}%")
        lines.append(f"Worst Day:           {metrics.worst_day:>+12.2f}%")
        lines.append("")
        
        lines.append("FEE BREAKDOWN")
        lines.append("-" * 40)
        lines.append(f"Platform Fees:       {fees.platform_fees_sol:>12.4f} SOL")
        lines.append(f"Network Fees:        {fees.network_fees_sol:>12.4f} SOL")
        lines.append(f"Priority Fees:       {fees.priority_fees_sol:>12.4f} SOL")
        lines.append(f"Total Fees:          {fees.total_fees_sol:>12.4f} SOL")
        lines.append(f"Fees % of Volume:    {fees.fees_as_percent_of_volume:>12.4f}%")
        lines.append("")
        
        lines.append("TOP PERFORMING TOKENS")
        lines.append("-" * 40)
        for tp in token_pnls[:5]:
            lines.append(f"{tp.token_symbol:<10} P&L: {tp.total_pnl:>+10.4f} SOL ({tp.roi_percent:>+7.2f}%)")
        lines.append("")
        
        if len(token_pnls) > 5:
            lines.append("WORST PERFORMING TOKENS")
            lines.append("-" * 40)
            for tp in token_pnls[-5:][::-1]:
                lines.append(f"{tp.token_symbol:<10} P&L: {tp.total_pnl:>+10.4f} SOL ({tp.roi_percent:>+7.2f}%)")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    async def export_tax_report(self, user_id: int, year: int) -> str:
        """Generate tax-friendly report for a year"""
        ua = self.get_user_analytics(user_id)
        
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23, 59, 59)
        
        trades = [t for t in ua.trades if start_date <= t["timestamp"] <= end_date]
        realized = [r for r in ua.realized_pnl_records if start_date <= r["timestamp"] <= end_date]
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        writer.writerow([
            "Date Acquired", "Date Sold", "Asset",
            "Quantity", "Cost Basis (SOL)", "Proceeds (SOL)",
            "Gain/Loss (SOL)", "Holding Period"
        ])
        
        for rec in realized:
            token = rec["token_symbol"]
            sell_date = rec["timestamp"]
            
            matching_buys = [
                t for t in trades
                if t["token_symbol"] == token and t["trade_type"] == "buy" and t["timestamp"] < sell_date
            ]
            
            buy_date = matching_buys[0]["timestamp"] if matching_buys else sell_date
            hold_days = (sell_date - buy_date).days
            holding_period = "Long-term" if hold_days > 365 else "Short-term"
            
            writer.writerow([
                buy_date.strftime("%Y-%m-%d"),
                sell_date.strftime("%Y-%m-%d"),
                token,
                str(rec["amount_sold"]),
                str(rec["cost_basis"]),
                str(rec["sale_value"]),
                str(rec["realized_pnl"]),
                holding_period
            ])
        
        total_gains = sum(r["realized_pnl"] for r in realized if r["realized_pnl"] > 0)
        total_losses = sum(r["realized_pnl"] for r in realized if r["realized_pnl"] < 0)
        
        writer.writerow([])
        writer.writerow(["SUMMARY"])
        writer.writerow(["Total Gains", str(total_gains)])
        writer.writerow(["Total Losses", str(total_losses)])
        writer.writerow(["Net Gain/Loss", str(total_gains + total_losses)])
        writer.writerow([])
        writer.writerow(["Note: This report is for informational purposes only."])
        writer.writerow(["Consult a tax professional for official tax reporting."])
        
        return output.getvalue()


                                          
    
    def register_alert_callback(self, callback: Callable[[int, Alert], None]):
        """Register callback for alerts"""
        self.alert_callbacks.append(callback)
    
    async def _send_alert(self, user_id: int, alert: Alert):
        """Send alert to registered callbacks"""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(user_id, alert)
                else:
                    callback(user_id, alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    async def generate_daily_summary(self, user_id: int) -> Alert:
        """Generate daily P&L summary alert"""
        pnl = await self.calculate_pnl(user_id, TimePeriod.DAY)
        portfolio = await self.get_portfolio(user_id)
        
        pnl_emoji = "[UP]" if pnl.total_pnl >= 0 else "[DOWN]"
        
        message = f"""
{pnl_emoji} Daily Summary

Portfolio Value: {portfolio.total_value_sol:.4f} SOL (${portfolio.total_value_usd:.2f})
Today's P&L: {pnl.total_pnl:+.4f} SOL ({pnl.pnl_percent:+.2f}%)
  Realized: {pnl.realized_pnl:+.4f} SOL
  Unrealized: {pnl.unrealized_pnl:+.4f} SOL
Fees: {pnl.fees_paid:.4f} SOL
Positions: {portfolio.num_positions}
        """.strip()
        
        return Alert(
            alert_type="daily_summary",
            title="Daily P&L Summary",
            message=message,
            severity="info",
            timestamp=datetime.utcnow(),
            data=pnl.to_dict()
        )
    
    async def generate_weekly_report(self, user_id: int) -> Alert:
        """Generate weekly performance report"""
        pnl = await self.calculate_pnl(user_id, TimePeriod.WEEK)
        stats = await self.get_trade_stats(user_id, TimePeriod.WEEK)
        metrics = await self.get_performance_metrics(user_id, TimePeriod.WEEK)
        
        message = f"""
Weekly Performance Report

Total P&L: {pnl.total_pnl:+.4f} SOL ({pnl.pnl_percent:+.2f}%)
Total Trades: {stats.total_trades}
Win Rate: {stats.win_rate:.1f}%
Profit Factor: {stats.profit_factor:.2f}
Max Drawdown: {metrics.max_drawdown_percent:.2f}%
Best Token: {stats.best_token or 'N/A'}
        """.strip()
        
        severity = "success" if pnl.total_pnl > 0 else "warning" if pnl.total_pnl == 0 else "info"
        
        return Alert(
            alert_type="weekly_report",
            title="Weekly Performance Report",
            message=message,
            severity=severity,
            timestamp=datetime.utcnow(),
            data={
                "pnl": pnl.to_dict(),
                "stats": stats.to_dict(),
                "metrics": metrics.to_dict()
            }
        )
    
    async def check_high_performer_alert(self, user_id: int) -> Optional[Alert]:
        """Check for high-performing token alerts"""
        token_pnls = await self.get_pnl_by_token(user_id)
        
        for tp in token_pnls:
            if tp.roi_percent >= 50:
                return Alert(
                    alert_type="high_performer",
                    title=f"{tp.token_symbol} Up {tp.roi_percent:.1f}%!",
                    message=f"{tp.token_symbol} is performing well! Current P&L: {tp.total_pnl:+.4f} SOL",
                    severity="success",
                    timestamp=datetime.utcnow(),
                    data=tp.to_dict()
                )
        return None
    
    async def check_portfolio_alerts(self, user_id: int) -> List[Alert]:
        """Check for various portfolio alerts"""
        alerts = []
        
        portfolio = await self.get_portfolio(user_id)
        metrics = await self.get_performance_metrics(user_id, TimePeriod.DAY)
        
                        
        if metrics.current_drawdown_percent >= 10:
            alerts.append(Alert(
                alert_type="drawdown_warning",
                title="Drawdown Alert",
                message=f"Portfolio is {metrics.current_drawdown_percent:.1f}% below peak",
                severity="warning",
                timestamp=datetime.utcnow(),
                data={"drawdown_percent": str(metrics.current_drawdown_percent)}
            ))
        
                              
        high_perf = await self.check_high_performer_alert(user_id)
        if high_perf:
            alerts.append(high_perf)
        
        return alerts
    
                                                    
    
    async def _snapshot_loop(self):
        """Take regular portfolio snapshots"""
        while self._running:
            try:
                for user_id in list(self.user_analytics.keys()):
                    ua = self.user_analytics[user_id]
                    snapshot = await self.get_portfolio(user_id)
                    ua.snapshots.append(snapshot)
                    
                                            
                    cutoff = datetime.utcnow() - timedelta(days=self.raw_retention_days)
                    ua.snapshots = [s for s in ua.snapshots if s.timestamp >= cutoff]
                    
                await asyncio.sleep(self.snapshot_interval.total_seconds())
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Snapshot loop error: {e}")
                await asyncio.sleep(60)
    
    async def _aggregation_loop(self):
        """Aggregate snapshots to daily/weekly"""
        while self._running:
            try:
                now = datetime.utcnow()
                
                                                   
                if now.hour == 0:
                    for user_id in list(self.user_analytics.keys()):
                        ua = self.user_analytics[user_id]
                        snapshot = await self.get_portfolio(user_id)
                        ua.daily_snapshots.append(snapshot)
                        
                                                  
                        cutoff = now - timedelta(days=self.daily_retention_days)
                        ua.daily_snapshots = [s for s in ua.daily_snapshots if s.timestamp >= cutoff]
                        
                                          
                        if now.weekday() == 6:
                            ua.weekly_snapshots.append(snapshot)
                        
                                            
                        alert = await self.generate_daily_summary(user_id)
                        await self._send_alert(user_id, alert)
                        
                                                 
                        if now.weekday() == 6:
                            weekly = await self.generate_weekly_report(user_id)
                            await self._send_alert(user_id, weekly)
                
                await asyncio.sleep(3600)                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Aggregation loop error: {e}")
                await asyncio.sleep(3600)
    
    async def _sol_price_loop(self):
        """Update SOL price regularly"""
        while self._running:
            try:
                if self.price_fetcher:
                    try:
                        price_data = await self.price_fetcher("SOL")
                        if price_data and "price_usd" in price_data:
                            self.sol_price_usd = Decimal(str(price_data["price_usd"]))
                    except Exception:
                        pass
                
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"SOL price loop error: {e}")
                await asyncio.sleep(60)


                                                

async def create_analytics_engine(database=None, price_fetcher=None) -> AnalyticsEngine:
    """Create and start analytics engine"""
    engine = AnalyticsEngine(database=database, price_fetcher=price_fetcher)
    await engine.start()
    return engine
