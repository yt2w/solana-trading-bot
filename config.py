"""
Solana Trading Bot - Configuration Module
Clean, production-ready configuration with environment variable support.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_env(key: str, default: str = None, required: bool = False) -> Optional[str]:
    """Get environment variable with optional default and required check."""
    value = os.getenv(key, default)
    if required and not value:
        raise ValueError(f"Required environment variable {key} is not set")
    return value


def get_env_float(key: str, default: float = 0.0) -> float:
    """Get environment variable as float."""
    value = os.getenv(key)
    return float(value) if value else default


def get_env_int(key: str, default: int = 0) -> int:
    """Get environment variable as integer."""
    value = os.getenv(key)
    return int(value) if value else default


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get environment variable as boolean."""
    value = os.getenv(key, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    elif value in ("false", "0", "no", "off"):
        return False
    return default


@dataclass
class SolanaConfig:
    """Solana network configuration."""
    rpc_url: str = field(default_factory=lambda: get_env("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"))
    ws_url: str = field(default_factory=lambda: get_env("SOLANA_WS_URL", "wss://api.mainnet-beta.solana.com"))
    network: str = field(default_factory=lambda: get_env("SOLANA_NETWORK", "mainnet-beta"))
    commitment: str = "confirmed"
    

@dataclass
class WalletConfig:
    """Wallet configuration."""
    private_key: str = field(default_factory=lambda: get_env("WALLET_PRIVATE_KEY", required=False) or "")
    
    def is_configured(self) -> bool:
        """Check if wallet is properly configured."""
        return bool(self.private_key)


@dataclass  
class TradingConfig:
    """Trading parameters configuration."""
    max_position_size_sol: float = field(default_factory=lambda: get_env_float("MAX_POSITION_SIZE_SOL", 1.0))
    min_position_size_sol: float = field(default_factory=lambda: get_env_float("MIN_POSITION_SIZE_SOL", 0.01))
    max_slippage_bps: int = field(default_factory=lambda: get_env_int("MAX_SLIPPAGE_BPS", 100))
    stop_loss_percent: float = field(default_factory=lambda: get_env_float("STOP_LOSS_PERCENT", 10.0))
    take_profit_percent: float = field(default_factory=lambda: get_env_float("TAKE_PROFIT_PERCENT", 50.0))
    max_daily_trades: int = field(default_factory=lambda: get_env_int("MAX_DAILY_TRADES", 10))
    max_open_positions: int = field(default_factory=lambda: get_env_int("MAX_OPEN_POSITIONS", 5))
    auto_sell_enabled: bool = field(default_factory=lambda: get_env_bool("AUTO_SELL_ENABLED", True))
    paper_trading: bool = field(default_factory=lambda: get_env_bool("PAPER_TRADING", True))


@dataclass
class JupiterConfig:
    """Jupiter DEX aggregator configuration."""
    api_url: str = "https://quote-api.jup.ag/v6"
    max_accounts: int = 64
    priority_fee_lamports: int = field(default_factory=lambda: get_env_int("PRIORITY_FEE_LAMPORTS", 10000))


@dataclass
class TelegramConfig:
    """Telegram bot configuration."""
    bot_token: str = field(default_factory=lambda: get_env("TELEGRAM_BOT_TOKEN", ""))
    chat_id: str = field(default_factory=lambda: get_env("TELEGRAM_CHAT_ID", ""))
    enabled: bool = field(default_factory=lambda: get_env_bool("TELEGRAM_ENABLED", False))
    
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id and self.enabled)


@dataclass
class DatabaseConfig:
    """Database configuration."""
    path: str = field(default_factory=lambda: get_env("DATABASE_PATH", "data/trading_bot.db"))
    
    def ensure_directory(self):
        os.makedirs(os.path.dirname(self.path) if os.path.dirname(self.path) else ".", exist_ok=True)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = field(default_factory=lambda: get_env("LOG_LEVEL", "INFO"))
    file_path: str = field(default_factory=lambda: get_env("LOG_FILE", "logs/bot.log"))
    max_size_mb: int = 10
    backup_count: int = 5
    
    def ensure_directory(self):
        log_dir = os.path.dirname(self.file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)


@dataclass
class BotConfig:
    """Main bot configuration aggregating all sub-configs."""
    solana: SolanaConfig = field(default_factory=SolanaConfig)
    wallet: WalletConfig = field(default_factory=WalletConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    jupiter: JupiterConfig = field(default_factory=JupiterConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def validate(self) -> list[str]:
        issues = []
        if not self.wallet.is_configured():
            issues.append("WARNING: Wallet private key not configured")
        if self.trading.paper_trading:
            issues.append("INFO: Paper trading mode enabled")
        if not self.telegram.is_configured():
            issues.append("INFO: Telegram notifications disabled")
        return issues
    
    def initialize(self):
        self.database.ensure_directory()
        self.logging.ensure_directory()


config = BotConfig()


class Tokens:
    """Common Solana token addresses."""
    SOL = "So11111111111111111111111111111111111111112"
    USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    USDT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
    RAY = "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"
    BONK = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
