import aiosqlite
import asyncio
import json
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, asdict, field
from contextlib import asynccontextmanager
from pathlib import Path
import logging
from enum import Enum
import time

logger = logging.getLogger(__name__)


class TradeStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OrderType(Enum):
    BUY = "buy"
    SELL = "sell"
    DCA = "dca"
    COPY = "copy"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


@dataclass
class Trade:
    wallet_address: str
    token_address: str
    order_type: str
    amount_sol: float
    amount_token: float
    price: float
    status: str = "pending"
    tx_signature: Optional[str] = None
    timestamp: Optional[datetime] = None
    slippage: float = 0.0
    fees: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


@dataclass
class WalletData:
    address: str
    encrypted_key: bytes
    salt: bytes
    created_at: datetime
    last_accessed: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class DatabaseError(Exception):
    pass


class ConnectionPoolExhausted(DatabaseError):
    pass


class QueryTimeout(DatabaseError):
    pass


class Database:
    def __init__(
        self,
        db_path: str = "trading_bot.db",
        max_connections: int = 10,
        query_timeout: float = 30.0,
        enable_wal: bool = True,
        integrity_check_interval: int = 3600
    ):
        self.db_path = Path(db_path)
        self.max_connections = max_connections
        self.query_timeout = query_timeout
        self.enable_wal = enable_wal
        self.integrity_check_interval = integrity_check_interval
        self._pool: List[aiosqlite.Connection] = []
        self._pool_lock = asyncio.Lock()
        self._initialized = False
        self._last_integrity_check = 0
        self._query_count = 0
        self._error_count = 0
        
    async def initialize(self) -> None:
        if self._initialized:
            return
            
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with self._get_connection() as conn:
            if self.enable_wal:
                await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute("PRAGMA synchronous=NORMAL")
            
            await conn.execute("PRAGMA foreign_keys=ON")
            await conn.execute("PRAGMA busy_timeout=30000")
            await conn.execute("PRAGMA cache_size=-64000")
            
            await self._create_tables(conn)
            await self._create_indexes(conn)
            await conn.commit()
            
        self._initialized = True
        logger.info(f"Database initialized: {self.db_path}")
        
    async def _create_tables(self, conn: aiosqlite.Connection) -> None:
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet_address TEXT NOT NULL,
                token_address TEXT NOT NULL,
                order_type TEXT NOT NULL,
                amount_sol REAL NOT NULL,
                amount_token REAL NOT NULL,
                price REAL NOT NULL,
                status TEXT DEFAULT 'pending',
                tx_signature TEXT UNIQUE,
                timestamp TEXT NOT NULL,
                slippage REAL DEFAULT 0,
                fees REAL DEFAULT 0,
                error_message TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                checksum TEXT NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS wallets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                address TEXT UNIQUE NOT NULL,
                encrypted_key BLOB NOT NULL,
                salt BLOB NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_accessed TEXT,
                is_active INTEGER DEFAULT 1,
                metadata TEXT DEFAULT '{}',
                checksum TEXT NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS token_cache (
                address TEXT PRIMARY KEY,
                symbol TEXT,
                name TEXT,
                decimals INTEGER,
                price REAL,
                market_cap REAL,
                volume_24h REAL,
                last_updated TEXT,
                metadata TEXT DEFAULT '{}'
            );
            
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                token_address TEXT,
                condition TEXT NOT NULL,
                threshold REAL,
                is_active INTEGER DEFAULT 1,
                triggered_count INTEGER DEFAULT 0,
                last_triggered TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                user_id TEXT,
                action TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                checksum TEXT NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS rate_limits (
                key TEXT PRIMARY KEY,
                count INTEGER DEFAULT 0,
                window_start TEXT,
                last_request TEXT
            );
            
            CREATE TABLE IF NOT EXISTS dca_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet_address TEXT NOT NULL,
                token_address TEXT NOT NULL,
                amount_per_order REAL NOT NULL,
                interval_seconds INTEGER NOT NULL,
                total_orders INTEGER NOT NULL,
                completed_orders INTEGER DEFAULT 0,
                is_active INTEGER DEFAULT 1,
                next_execution TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT DEFAULT '{}'
            );
            
            CREATE TABLE IF NOT EXISTS copy_trading (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                follower_wallet TEXT NOT NULL,
                leader_wallet TEXT NOT NULL,
                allocation_percent REAL NOT NULL,
                max_trade_size REAL,
                is_active INTEGER DEFAULT 1,
                total_copied INTEGER DEFAULT 0,
                total_profit REAL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
    async def _create_indexes(self, conn: aiosqlite.Connection) -> None:
        await conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_trades_wallet ON trades(wallet_address);
            CREATE INDEX IF NOT EXISTS idx_trades_token ON trades(token_address);
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
            CREATE INDEX IF NOT EXISTS idx_wallets_address ON wallets(address);
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
            CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id);
            CREATE INDEX IF NOT EXISTS idx_dca_wallet ON dca_orders(wallet_address);
            CREATE INDEX IF NOT EXISTS idx_copy_follower ON copy_trading(follower_wallet);
        """)
        
    def _compute_checksum(self, data: Dict[str, Any]) -> str:
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
        
    def _verify_checksum(self, data: Dict[str, Any], checksum: str) -> bool:
        return hmac.compare_digest(self._compute_checksum(data), checksum)
        
    @asynccontextmanager
    async def _get_connection(self):
        conn = None
        try:
            async with asyncio.timeout(self.query_timeout):
                async with self._pool_lock:
                    if self._pool:
                        conn = self._pool.pop()
                    else:
                        conn = await aiosqlite.connect(
                            self.db_path,
                            timeout=self.query_timeout
                        )
                        conn.row_factory = aiosqlite.Row
                yield conn
        except asyncio.TimeoutError:
            self._error_count += 1
            raise QueryTimeout(f"Query timeout after {self.query_timeout}s")
        finally:
            if conn:
                async with self._pool_lock:
                    if len(self._pool) < self.max_connections:
                        self._pool.append(conn)
                    else:
                        await conn.close()
                        
    async def add_trade(self, trade: Trade) -> int:
        if not self._initialized:
            await self.initialize()
            
        trade_dict = asdict(trade)
        trade_dict["timestamp"] = trade.timestamp.isoformat()
        checksum = self._compute_checksum(trade_dict)
        
        async with self._get_connection() as conn:
            cursor = await conn.execute("""
                INSERT INTO trades (
                    wallet_address, token_address, order_type, amount_sol,
                    amount_token, price, status, tx_signature, timestamp,
                    slippage, fees, error_message, metadata, checksum
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.wallet_address, trade.token_address, trade.order_type,
                trade.amount_sol, trade.amount_token, trade.price, trade.status,
                trade.tx_signature, trade.timestamp.isoformat(), trade.slippage,
                trade.fees, trade.error_message, json.dumps(trade.metadata), checksum
            ))
            await conn.commit()
            self._query_count += 1
            
            trade_id = cursor.lastrowid
            logger.info(f"Trade added: {trade_id} - {trade.order_type} {trade.amount_sol} SOL")
            return trade_id
            
    async def get_trade(self, trade_id: int) -> Optional[Trade]:
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM trades WHERE id = ?", (trade_id,)
            )
            row = await cursor.fetchone()
            self._query_count += 1
            
            if row:
                return Trade(
                    wallet_address=row["wallet_address"],
                    token_address=row["token_address"],
                    order_type=row["order_type"],
                    amount_sol=row["amount_sol"],
                    amount_token=row["amount_token"],
                    price=row["price"],
                    status=row["status"],
                    tx_signature=row["tx_signature"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    slippage=row["slippage"],
                    fees=row["fees"],
                    error_message=row["error_message"],
                    metadata=json.loads(row["metadata"] or "{}")
                )
            return None
            
    async def get_trades_by_wallet(
        self,
        wallet_address: str,
        limit: int = 100,
        offset: int = 0,
        status: Optional[str] = None
    ) -> List[Trade]:
        query = "SELECT * FROM trades WHERE wallet_address = ?"
        params = [wallet_address]
        
        if status:
            query += " AND status = ?"
            params.append(status)
            
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        async with self._get_connection() as conn:
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
            self._query_count += 1
            
            return [Trade(
                wallet_address=row["wallet_address"],
                token_address=row["token_address"],
                order_type=row["order_type"],
                amount_sol=row["amount_sol"],
                amount_token=row["amount_token"],
                price=row["price"],
                status=row["status"],
                tx_signature=row["tx_signature"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                slippage=row["slippage"],
                fees=row["fees"],
                error_message=row["error_message"],
                metadata=json.loads(row["metadata"] or "{}")
            ) for row in rows]
            
    async def update_trade_status(
        self,
        trade_id: int,
        status: str,
        tx_signature: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> bool:
        async with self._get_connection() as conn:
            await conn.execute("""
                UPDATE trades 
                SET status = ?, tx_signature = COALESCE(?, tx_signature),
                    error_message = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (status, tx_signature, error_message, trade_id))
            await conn.commit()
            self._query_count += 1
            return True
            
    async def save_wallet(self, wallet: WalletData) -> bool:
        wallet_dict = {
            "address": wallet.address,
            "created_at": wallet.created_at.isoformat()
        }
        checksum = self._compute_checksum(wallet_dict)
        
        async with self._get_connection() as conn:
            await conn.execute("""
                INSERT OR REPLACE INTO wallets (
                    address, encrypted_key, salt, created_at, 
                    last_accessed, is_active, metadata, checksum
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                wallet.address, wallet.encrypted_key, wallet.salt,
                wallet.created_at.isoformat(),
                wallet.last_accessed.isoformat() if wallet.last_accessed else None,
                1 if wallet.is_active else 0,
                json.dumps(wallet.metadata), checksum
            ))
            await conn.commit()
            self._query_count += 1
            logger.info(f"Wallet saved: {wallet.address[:8]}...")
            return True
            
    async def get_wallet(self, address: str) -> Optional[WalletData]:
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM wallets WHERE address = ? AND is_active = 1",
                (address,)
            )
            row = await cursor.fetchone()
            self._query_count += 1
            
            if row:
                await conn.execute(
                    "UPDATE wallets SET last_accessed = CURRENT_TIMESTAMP WHERE address = ?",
                    (address,)
                )
                await conn.commit()
                
                return WalletData(
                    address=row["address"],
                    encrypted_key=row["encrypted_key"],
                    salt=row["salt"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else None,
                    is_active=bool(row["is_active"]),
                    metadata=json.loads(row["metadata"] or "{}")
                )
            return None
            
    async def add_audit_log(
        self,
        event_type: str,
        action: str,
        user_id: Optional[str] = None,
        details: Optional[Dict] = None,
        ip_address: Optional[str] = None
    ) -> int:
        log_data = {
            "event_type": event_type,
            "action": action,
            "timestamp": datetime.utcnow().isoformat()
        }
        checksum = self._compute_checksum(log_data)
        
        async with self._get_connection() as conn:
            cursor = await conn.execute("""
                INSERT INTO audit_log (event_type, user_id, action, details, ip_address, checksum)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (event_type, user_id, action, json.dumps(details) if details else None, ip_address, checksum))
            await conn.commit()
            self._query_count += 1
            return cursor.lastrowid
            
    async def get_statistics(self, wallet_address: Optional[str] = None) -> Dict[str, Any]:
        async with self._get_connection() as conn:
            where_clause = "WHERE wallet_address = ?" if wallet_address else ""
            params = (wallet_address,) if wallet_address else ()
            
            cursor = await conn.execute(f"""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN order_type = 'buy' THEN 1 ELSE 0 END) as buy_count,
                    SUM(CASE WHEN order_type = 'sell' THEN 1 ELSE 0 END) as sell_count,
                    SUM(CASE WHEN status = 'confirmed' THEN 1 ELSE 0 END) as confirmed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    SUM(amount_sol) as total_volume_sol,
                    SUM(fees) as total_fees,
                    AVG(slippage) as avg_slippage
                FROM trades {where_clause}
            """, params)
            
            row = await cursor.fetchone()
            self._query_count += 1
            
            return {
                "total_trades": row["total_trades"] or 0,
                "buy_count": row["buy_count"] or 0,
                "sell_count": row["sell_count"] or 0,
                "confirmed": row["confirmed"] or 0,
                "failed": row["failed"] or 0,
                "total_volume_sol": row["total_volume_sol"] or 0,
                "total_fees": row["total_fees"] or 0,
                "avg_slippage": row["avg_slippage"] or 0,
                "success_rate": (row["confirmed"] / row["total_trades"] * 100) if row["total_trades"] else 0
            }
            
    async def check_rate_limit(self, key: str, max_requests: int, window_seconds: int) -> bool:
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=window_seconds)
        
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT count, window_start FROM rate_limits WHERE key = ?",
                (key,)
            )
            row = await cursor.fetchone()
            
            if row:
                stored_window = datetime.fromisoformat(row["window_start"])
                if stored_window < window_start:
                    await conn.execute(
                        "UPDATE rate_limits SET count = 1, window_start = ?, last_request = ? WHERE key = ?",
                        (now.isoformat(), now.isoformat(), key)
                    )
                    await conn.commit()
                    return True
                elif row["count"] < max_requests:
                    await conn.execute(
                        "UPDATE rate_limits SET count = count + 1, last_request = ? WHERE key = ?",
                        (now.isoformat(), key)
                    )
                    await conn.commit()
                    return True
                else:
                    return False
            else:
                await conn.execute(
                    "INSERT INTO rate_limits (key, count, window_start, last_request) VALUES (?, 1, ?, ?)",
                    (key, now.isoformat(), now.isoformat())
                )
                await conn.commit()
                return True
                
    async def integrity_check(self) -> Dict[str, Any]:
        now = time.time()
        if now - self._last_integrity_check < self.integrity_check_interval:
            return {"status": "skipped", "reason": "too_recent"}
            
        self._last_integrity_check = now
        
        async with self._get_connection() as conn:
            cursor = await conn.execute("PRAGMA integrity_check")
            result = await cursor.fetchone()
            
            cursor = await conn.execute("PRAGMA quick_check")
            quick_result = await cursor.fetchone()
            
            return {
                "status": "ok" if result[0] == "ok" else "error",
                "integrity_check": result[0],
                "quick_check": quick_result[0],
                "query_count": self._query_count,
                "error_count": self._error_count,
                "pool_size": len(self._pool)
            }
            
    async def vacuum(self) -> None:
        async with self._get_connection() as conn:
            await conn.execute("VACUUM")
            logger.info("Database vacuumed")
            
    async def close(self) -> None:
        async with self._pool_lock:
            for conn in self._pool:
                await conn.close()
            self._pool.clear()
        self._initialized = False
        logger.info("Database connections closed")
        
    async def __aenter__(self):
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()