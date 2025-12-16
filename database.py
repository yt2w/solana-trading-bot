"""
SQLite database management for user wallets and settings.
"""

import sqlite3
import json
import os
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import contextmanager


class Database:
    """SQLite database manager for the trading bot."""
    
    def __init__(self, db_path: str = "data/bot_database.db"):
        """Initialize database connection."""
        self.db_path = db_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    telegram_id INTEGER PRIMARY KEY,
                    username TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Wallets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS wallets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telegram_id INTEGER NOT NULL,
                    wallet_name TEXT NOT NULL,
                    public_key TEXT NOT NULL,
                    encrypted_private_key TEXT NOT NULL,
                    is_default BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (telegram_id) REFERENCES users(telegram_id),
                    UNIQUE(telegram_id, wallet_name)
                )
            """)
            
            # User settings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    telegram_id INTEGER PRIMARY KEY,
                    slippage REAL DEFAULT 1.0,
                    priority_fee INTEGER DEFAULT 10000,
                    auto_buy_amount REAL DEFAULT 0.1,
                    settings_json TEXT DEFAULT '{}',
                    FOREIGN KEY (telegram_id) REFERENCES users(telegram_id)
                )
            """)
            
            # Transactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telegram_id INTEGER NOT NULL,
                    tx_signature TEXT UNIQUE,
                    tx_type TEXT NOT NULL,
                    token_address TEXT,
                    amount_in REAL,
                    amount_out REAL,
                    status TEXT DEFAULT 'pending',
                    fee_amount REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (telegram_id) REFERENCES users(telegram_id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_wallets_telegram ON wallets(telegram_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_telegram ON transactions(telegram_id)")
    
    # ==================== User Operations ====================
    
    def get_or_create_user(self, telegram_id: int, username: Optional[str] = None) -> Dict[str, Any]:
        """Get existing user or create new one."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Try to get existing user
            cursor.execute("SELECT * FROM users WHERE telegram_id = ?", (telegram_id,))
            user = cursor.fetchone()
            
            if user:
                # Update last active
                cursor.execute(
                    "UPDATE users SET last_active = ?, username = COALESCE(?, username) WHERE telegram_id = ?",
                    (datetime.now(), username, telegram_id)
                )
                return dict(user)
            
            # Create new user
            cursor.execute(
                "INSERT INTO users (telegram_id, username) VALUES (?, ?)",
                (telegram_id, username)
            )
            
            # Create default settings
            cursor.execute(
                "INSERT INTO user_settings (telegram_id) VALUES (?)",
                (telegram_id,)
            )
            
            return {
                "telegram_id": telegram_id,
                "username": username,
                "created_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat()
            }
    
    # ==================== Wallet Operations ====================
    
    def save_wallet(self, telegram_id: int, wallet_name: str, public_key: str, 
                    encrypted_private_key: str, is_default: bool = False) -> bool:
        """Save a new wallet for user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # If this is default, unset other defaults
            if is_default:
                cursor.execute(
                    "UPDATE wallets SET is_default = 0 WHERE telegram_id = ?",
                    (telegram_id,)
                )
            
            # Check if it's the first wallet (make it default)
            cursor.execute("SELECT COUNT(*) FROM wallets WHERE telegram_id = ?", (telegram_id,))
            if cursor.fetchone()[0] == 0:
                is_default = True
            
            cursor.execute("""
                INSERT OR REPLACE INTO wallets 
                (telegram_id, wallet_name, public_key, encrypted_private_key, is_default)
                VALUES (?, ?, ?, ?, ?)
            """, (telegram_id, wallet_name, public_key, encrypted_private_key, is_default))
            
            return True
    
    def get_wallets(self, telegram_id: int) -> List[Dict[str, Any]]:
        """Get all wallets for a user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM wallets WHERE telegram_id = ? ORDER BY is_default DESC, created_at",
                (telegram_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_default_wallet(self, telegram_id: int) -> Optional[Dict[str, Any]]:
        """Get user's default wallet."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM wallets WHERE telegram_id = ? AND is_default = 1",
                (telegram_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def delete_wallet(self, telegram_id: int, wallet_name: str) -> bool:
        """Delete a wallet."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM wallets WHERE telegram_id = ? AND wallet_name = ?",
                (telegram_id, wallet_name)
            )
            return cursor.rowcount > 0
    
    def set_default_wallet(self, telegram_id: int, wallet_name: str) -> bool:
        """Set a wallet as default."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Unset all defaults
            cursor.execute(
                "UPDATE wallets SET is_default = 0 WHERE telegram_id = ?",
                (telegram_id,)
            )
            
            # Set new default
            cursor.execute(
                "UPDATE wallets SET is_default = 1 WHERE telegram_id = ? AND wallet_name = ?",
                (telegram_id, wallet_name)
            )
            return cursor.rowcount > 0
    
    # ==================== Settings Operations ====================
    
    def get_user_settings(self, telegram_id: int) -> Dict[str, Any]:
        """Get user settings."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM user_settings WHERE telegram_id = ?",
                (telegram_id,)
            )
            row = cursor.fetchone()
            
            if row:
                settings = dict(row)
                settings["extra"] = json.loads(settings.get("settings_json", "{}"))
                return settings
            
            return {
                "slippage": 1.0,
                "priority_fee": 10000,
                "auto_buy_amount": 0.1,
                "extra": {}
            }
    
    def update_user_settings(self, telegram_id: int, **kwargs) -> bool:
        """Update user settings."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Build update query dynamically
            allowed_fields = ["slippage", "priority_fee", "auto_buy_amount"]
            updates = []
            values = []
            
            for field in allowed_fields:
                if field in kwargs:
                    updates.append(f"{field} = ?")
                    values.append(kwargs[field])
            
            if "extra" in kwargs:
                updates.append("settings_json = ?")
                values.append(json.dumps(kwargs["extra"]))
            
            if not updates:
                return False
            
            values.append(telegram_id)
            cursor.execute(
                f"UPDATE user_settings SET {', '.join(updates)} WHERE telegram_id = ?",
                values
            )
            return cursor.rowcount > 0
    
    # ==================== Transaction Operations ====================
    
    def log_transaction(self, telegram_id: int, tx_type: str, token_address: str = None,
                        amount_in: float = None, amount_out: float = None,
                        tx_signature: str = None, fee_amount: float = 0) -> int:
        """Log a transaction."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO transactions 
                (telegram_id, tx_type, token_address, amount_in, amount_out, tx_signature, fee_amount)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (telegram_id, tx_type, token_address, amount_in, amount_out, tx_signature, fee_amount))
            return cursor.lastrowid
    
    def update_transaction_status(self, tx_signature: str, status: str) -> bool:
        """Update transaction status."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE transactions SET status = ? WHERE tx_signature = ?",
                (status, tx_signature)
            )
            return cursor.rowcount > 0
    
    def get_user_transactions(self, telegram_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent transactions for a user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM transactions WHERE telegram_id = ? ORDER BY created_at DESC LIMIT ?",
                (telegram_id, limit)
            )
            return [dict(row) for row in cursor.fetchall()]
