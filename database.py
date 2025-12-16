"""
Solana Trading Bot - Database Module
SQLite database for storing trades, positions, and bot state.
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Trade record."""
    id: Optional[int] = None
    timestamp: str = ""
    token_address: str = ""
    token_symbol: str = ""
    side: str = ""  # "buy" or "sell"
    amount_sol: float = 0.0
    amount_token: float = 0.0
    price: float = 0.0
    tx_signature: str = ""
    status: str = "pending"  # pending, confirmed, failed
    pnl: float = 0.0
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Position:
    """Open position record."""
    id: Optional[int] = None
    token_address: str = ""
    token_symbol: str = ""
    entry_price: float = 0.0
    amount_token: float = 0.0
    amount_sol_invested: float = 0.0
    entry_timestamp: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: str = "open"  # open, closed
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Database:
    """SQLite database manager for the trading bot."""
    
    def __init__(self, db_path: str = "data/trading_bot.db"):
        """Initialize database connection."""
        self.db_path = db_path
        self._ensure_directory()
        self._init_tables()
    
    def _ensure_directory(self):
        """Ensure database directory exists."""
        import os
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
    
    @contextmanager
    def get_connection(self):
        """Get database connection context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _init_tables(self):
        """Initialize database tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    token_address TEXT NOT NULL,
                    token_symbol TEXT,
                    side TEXT NOT NULL,
                    amount_sol REAL,
                    amount_token REAL,
                    price REAL,
                    tx_signature TEXT,
                    status TEXT DEFAULT 'pending',
                    pnl REAL DEFAULT 0,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Positions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_address TEXT NOT NULL,
                    token_symbol TEXT,
                    entry_price REAL,
                    amount_token REAL,
                    amount_sol_invested REAL,
                    entry_timestamp TEXT,
                    stop_loss REAL,
                    take_profit REAL,
                    status TEXT DEFAULT 'open',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Bot state table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bot_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_token ON trades(token_address)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_token ON positions(token_address)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)")
            
            logger.info("Database tables initialized")
    
    # ==================== TRADE OPERATIONS ====================
    
    def add_trade(self, trade: Trade) -> int:
        """Add a new trade record."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (timestamp, token_address, token_symbol, side, 
                                   amount_sol, amount_token, price, tx_signature, 
                                   status, pnl, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.timestamp or datetime.utcnow().isoformat(),
                trade.token_address,
                trade.token_symbol,
                trade.side,
                trade.amount_sol,
                trade.amount_token,
                trade.price,
                trade.tx_signature,
                trade.status,
                trade.pnl,
                trade.notes
            ))
            return cursor.lastrowid
    
    def get_trades(self, limit: int = 100, token_address: str = None) -> List[Trade]:
        """Get recent trades, optionally filtered by token."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if token_address:
                cursor.execute(
                    "SELECT * FROM trades WHERE token_address = ? ORDER BY timestamp DESC LIMIT ?",
                    (token_address, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )
            
            return [Trade(**dict(row)) for row in cursor.fetchall()]
    
    def get_daily_trade_count(self) -> int:
        """Get number of trades today."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM trades WHERE timestamp LIKE ?",
                (f"{today}%",)
            )
            return cursor.fetchone()[0]
    
    # ==================== POSITION OPERATIONS ====================
    
    def add_position(self, position: Position) -> int:
        """Add a new position."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO positions (token_address, token_symbol, entry_price,
                                      amount_token, amount_sol_invested, entry_timestamp,
                                      stop_loss, take_profit, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position.token_address,
                position.token_symbol,
                position.entry_price,
                position.amount_token,
                position.amount_sol_invested,
                position.entry_timestamp or datetime.utcnow().isoformat(),
                position.stop_loss,
                position.take_profit,
                position.status
            ))
            return cursor.lastrowid
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM positions WHERE status = 'open'")
            return [Position(**dict(row)) for row in cursor.fetchall()]
    
    def get_position_by_token(self, token_address: str) -> Optional[Position]:
        """Get open position for a specific token."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM positions WHERE token_address = ? AND status = 'open' LIMIT 1",
                (token_address,)
            )
            row = cursor.fetchone()
            return Position(**dict(row)) if row else None
    
    def close_position(self, position_id: int, pnl: float = 0.0) -> bool:
        """Close a position."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE positions SET status = 'closed' WHERE id = ?",
                (position_id,)
            )
            return cursor.rowcount > 0
    
    def get_open_position_count(self) -> int:
        """Get count of open positions."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM positions WHERE status = 'open'")
            return cursor.fetchone()[0]
    
    # ==================== STATE OPERATIONS ====================
    
    def set_state(self, key: str, value: Any):
        """Set a bot state value."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO bot_state (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, json.dumps(value), datetime.utcnow().isoformat()))
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a bot state value."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM bot_state WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return default
    
    # ==================== STATISTICS ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get trading statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total trades
            cursor.execute("SELECT COUNT(*) FROM trades")
            total_trades = cursor.fetchone()[0]
            
            # Total PnL
            cursor.execute("SELECT SUM(pnl) FROM trades WHERE status = 'confirmed'")
            total_pnl = cursor.fetchone()[0] or 0.0
            
            # Win rate
            cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl > 0 AND status = 'confirmed'")
            winning_trades = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'confirmed'")
            confirmed_trades = cursor.fetchone()[0]
            
            win_rate = (winning_trades / confirmed_trades * 100) if confirmed_trades > 0 else 0
            
            # Open positions
            cursor.execute("SELECT COUNT(*) FROM positions WHERE status = 'open'")
            open_positions = cursor.fetchone()[0]
            
            return {
                "total_trades": total_trades,
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "winning_trades": winning_trades,
                "open_positions": open_positions,
                "daily_trades": self.get_daily_trade_count()
            }


# Create global database instance
db: Optional[Database] = None


def get_database(db_path: str = "data/trading_bot.db") -> Database:
    """Get or create database instance."""
    global db
    if db is None:
        db = Database(db_path)
    return db


if __name__ == "__main__":
    # Test database operations
    logging.basicConfig(level=logging.INFO)
    
    test_db = Database("data/test_trading_bot.db")
    
    # Test adding a trade
    trade = Trade(
        timestamp=datetime.utcnow().isoformat(),
        token_address="DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
        token_symbol="BONK",
        side="buy",
        amount_sol=0.1,
        amount_token=1000000,
        price=0.0000001,
        status="confirmed"
    )
    trade_id = test_db.add_trade(trade)
    print(f"Added trade with ID: {trade_id}")
    
    # Test getting statistics
    stats = test_db.get_statistics()
    print(f"Statistics: {stats}")
