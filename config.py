"""
Production Configuration Module for Solana Trading Bot

This module provides comprehensive configuration management using Pydantic v2 BaseSettings.
All settings are loaded from environment variables with validation and type safety.

Usage:
    from config import settings
    print(settings.telegram.bot_token)
"""

from __future__ import annotations

import os
import sys
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Set

from pydantic import (
    AnyHttpUrl,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# ENUMS
# =============================================================================

class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging level options."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Network(str, Enum):
    """Solana network options."""
    MAINNET = "mainnet-beta"
    DEVNET = "devnet"
    TESTNET = "testnet"


# =============================================================================
# BASE CONFIGURATION
# =============================================================================

class BaseConfig(BaseSettings):
    """Base configuration with common settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


# =============================================================================
# TELEGRAM CONFIGURATION
# =============================================================================

class TelegramSettings(BaseConfig):
    """Telegram bot configuration settings."""
    
    model_config = SettingsConfigDict(
        env_prefix="TELEGRAM_",
        env_file=".env",
        extra="ignore",
    )
    
    bot_token: SecretStr = Field(
        ...,
        description="Telegram bot token from @BotFather",
        min_length=40,
    )
    
    admin_user_ids: Set[int] = Field(
        default_factory=set,
        description="Set of authorized admin user IDs",
    )
    
    allowed_user_ids: Set[int] = Field(
        default_factory=set,
        description="Set of allowed user IDs (empty = all allowed)",
    )
    
    rate_limit_messages: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Max messages per minute per user",
    )
    
    rate_limit_window: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Rate limit window in seconds",
    )
    
    webhook_enabled: bool = Field(
        default=False,
        description="Use webhook instead of polling",
    )
    
    webhook_url: Optional[AnyHttpUrl] = Field(
        default=None,
        description="Webhook URL if enabled",
    )
    
    webhook_port: int = Field(
        default=8443,
        ge=1,
        le=65535,
        description="Webhook server port",
    )
    
    command_timeout: int = Field(
        default=60,
        ge=5,
        le=300,
        description="Command execution timeout in seconds",
    )
    
    @field_validator("admin_user_ids", "allowed_user_ids", mode="before")
    @classmethod
    def parse_user_ids(cls, v: Any) -> Set[int]:
        """Parse comma-separated user IDs from environment variable."""
        if isinstance(v, set):
            return v
        if isinstance(v, (list, tuple)):
            return set(v)
        if isinstance(v, str):
            if not v.strip():
                return set()
            return {int(uid.strip()) for uid in v.split(",") if uid.strip()}
        return set()
    
    @model_validator(mode="after")
    def validate_webhook_config(self) -> "TelegramSettings":
        """Validate webhook configuration consistency."""
        if self.webhook_enabled and not self.webhook_url:
            raise ValueError("webhook_url required when webhook_enabled is True")
        return self


# =============================================================================
# SOLANA RPC CONFIGURATION
# =============================================================================

class SolanaRPCSettings(BaseConfig):
    """Solana RPC connection configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="SOLANA_",
        env_file=".env",
        extra="ignore",
    )
    
    network: Network = Field(
        default=Network.MAINNET,
        description="Solana network to connect to",
    )
    
    rpc_url: AnyHttpUrl = Field(
        default="https://api.mainnet-beta.solana.com",
        description="Primary RPC endpoint URL",
    )
    
    rpc_url_backup: Optional[AnyHttpUrl] = Field(
        default=None,
        description="Backup RPC endpoint URL",
    )
    
    ws_url: Optional[AnyHttpUrl] = Field(
        default=None,
        description="WebSocket endpoint URL for subscriptions",
    )
    
    commitment: str = Field(
        default="confirmed",
        pattern="^(processed|confirmed|finalized)$",
        description="Transaction commitment level",
    )
    
    timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="RPC request timeout in seconds",
    )
    
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum RPC retry attempts",
    )
    
    retry_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Delay between retries in seconds",
    )
    
    rate_limit_rps: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Rate limit requests per second",
    )
    
    @field_validator("rpc_url", "rpc_url_backup", "ws_url", mode="before")
    @classmethod
    def validate_url(cls, v: Any) -> Any:
        """Validate and normalize URL."""
        if v is None or v == "":
            return None
        return v


# =============================================================================
# WALLET CONFIGURATION
# =============================================================================

class WalletSettings(BaseConfig):
    """Wallet and key management configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="WALLET_",
        env_file=".env",
        extra="ignore",
    )
    
    private_key: SecretStr = Field(
        ...,
        description="Base58 encoded private key or path to keyfile",
        min_length=32,
    )
    
    use_keyfile: bool = Field(
        default=False,
        description="Treat private_key as path to keyfile",
    )
    
    keyfile_password: Optional[SecretStr] = Field(
        default=None,
        description="Password for encrypted keyfile",
    )
    
    auto_approve_below: float = Field(
        default=0.0,
        ge=0.0,
        description="Auto-approve transactions below this SOL amount",
    )
    
    require_confirmation: bool = Field(
        default=True,
        description="Require manual confirmation for trades",
    )


# =============================================================================
# JUPITER CONFIGURATION
# =============================================================================

class JupiterSettings(BaseConfig):
    """Jupiter aggregator API configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="JUPITER_",
        env_file=".env",
        extra="ignore",
    )
    
    api_url: AnyHttpUrl = Field(
        default="https://quote-api.jup.ag/v6",
        description="Jupiter API base URL",
    )
    
    slippage_bps: int = Field(
        default=50,
        ge=1,
        le=5000,
        description="Default slippage tolerance in basis points",
    )
    
    max_slippage_bps: int = Field(
        default=500,
        ge=10,
        le=10000,
        description="Maximum allowed slippage in basis points",
    )
    
    priority_fee_lamports: int = Field(
        default=10000,
        ge=0,
        le=10000000,
        description="Priority fee in lamports",
    )
    
    dynamic_priority_fee: bool = Field(
        default=True,
        description="Use dynamic priority fee estimation",
    )
    
    max_priority_fee_lamports: int = Field(
        default=1000000,
        ge=0,
        le=100000000,
        description="Maximum priority fee cap",
    )
    
    use_versioned_transactions: bool = Field(
        default=True,
        description="Use versioned transactions (v0)",
    )
    
    only_direct_routes: bool = Field(
        default=False,
        description="Only use direct swap routes",
    )
    
    exclude_dexes: Set[str] = Field(
        default_factory=set,
        description="DEXes to exclude from routing",
    )
    
    timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="API request timeout in seconds",
    )
    
    @field_validator("exclude_dexes", mode="before")
    @classmethod
    def parse_dexes(cls, v: Any) -> Set[str]:
        """Parse comma-separated DEX names."""
        if isinstance(v, set):
            return v
        if isinstance(v, (list, tuple)):
            return set(v)
        if isinstance(v, str):
            if not v.strip():
                return set()
            return {dex.strip() for dex in v.split(",") if dex.strip()}
        return set()
    
    @model_validator(mode="after")
    def validate_slippage(self) -> "JupiterSettings":
        """Ensure slippage_bps does not exceed max_slippage_bps."""
        if self.slippage_bps > self.max_slippage_bps:
            raise ValueError(
                f"slippage_bps ({self.slippage_bps}) cannot exceed "
                f"max_slippage_bps ({self.max_slippage_bps})"
            )
        return self


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

class DatabaseSettings(BaseConfig):
    """Database connection configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="DATABASE_",
        env_file=".env",
        extra="ignore",
    )
    
    url: SecretStr = Field(
        default=SecretStr("sqlite+aiosqlite:///./data/trading_bot.db"),
        description="Database connection URL",
    )
    
    echo: bool = Field(
        default=False,
        description="Echo SQL statements (debug)",
    )
    
    pool_size: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Connection pool size",
    )
    
    max_overflow: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Max connections above pool_size",
    )
    
    pool_timeout: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Pool connection timeout in seconds",
    )
    
    pool_recycle: int = Field(
        default=3600,
        ge=300,
        le=86400,
        description="Connection recycle time in seconds",
    )
    
    auto_migrate: bool = Field(
        default=True,
        description="Run migrations on startup",
    )
    
    @property
    def async_url(self) -> str:
        """Get async-compatible database URL."""
        url = self.url.get_secret_value()
        if url.startswith("sqlite:") and "aiosqlite" not in url:
            return url.replace("sqlite:", "sqlite+aiosqlite:")
        if url.startswith("postgresql:") and "asyncpg" not in url:
            return url.replace("postgresql:", "postgresql+asyncpg:")
        return url


# =============================================================================
# RISK MANAGEMENT CONFIGURATION
# =============================================================================

class RiskSettings(BaseConfig):
    """Trading risk management configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="RISK_",
        env_file=".env",
        extra="ignore",
    )
    
    max_position_size_sol: float = Field(
        default=1.0,
        gt=0.0,
        le=1000.0,
        description="Maximum position size in SOL",
    )
    
    max_position_size_pct: float = Field(
        default=10.0,
        gt=0.0,
        le=100.0,
        description="Maximum position as % of portfolio",
    )
    
    max_daily_trades: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum trades per day",
    )
    
    max_daily_volume_sol: float = Field(
        default=10.0,
        gt=0.0,
        le=10000.0,
        description="Maximum daily trading volume in SOL",
    )
    
    stop_loss_pct: float = Field(
        default=5.0,
        ge=0.0,
        le=50.0,
        description="Default stop-loss percentage",
    )
    
    take_profit_pct: float = Field(
        default=10.0,
        ge=0.0,
        le=500.0,
        description="Default take-profit percentage",
    )
    
    trailing_stop_enabled: bool = Field(
        default=False,
        description="Enable trailing stop-loss",
    )
    
    trailing_stop_pct: float = Field(
        default=3.0,
        ge=0.5,
        le=20.0,
        description="Trailing stop-loss percentage",
    )
    
    min_liquidity_usd: float = Field(
        default=10000.0,
        ge=0.0,
        description="Minimum token liquidity in USD",
    )
    
    blacklisted_tokens: Set[str] = Field(
        default_factory=set,
        description="Token addresses to never trade",
    )
    
    whitelisted_tokens: Set[str] = Field(
        default_factory=set,
        description="Only trade these tokens (empty = all)",
    )
    
    cool_down_seconds: int = Field(
        default=60,
        ge=0,
        le=3600,
        description="Cooldown between trades on same token",
    )
    
    @field_validator("blacklisted_tokens", "whitelisted_tokens", mode="before")
    @classmethod
    def parse_token_set(cls, v: Any) -> Set[str]:
        """Parse comma-separated token addresses."""
        if isinstance(v, set):
            return v
        if isinstance(v, (list, tuple)):
            return set(v)
        if isinstance(v, str):
            if not v.strip():
                return set()
            return {t.strip() for t in v.split(",") if t.strip()}
        return set()


# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

class SecuritySettings(BaseConfig):
    """Security and access control configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="SECURITY_",
        env_file=".env",
        extra="ignore",
    )
    
    encryption_key: Optional[SecretStr] = Field(
        default=None,
        description="Fernet encryption key for sensitive data",
    )
    
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key for external access",
    )
    
    jwt_secret: Optional[SecretStr] = Field(
        default=None,
        description="JWT signing secret",
    )
    
    jwt_expiry_hours: int = Field(
        default=24,
        ge=1,
        le=720,
        description="JWT token expiry in hours",
    )
    
    ip_whitelist: Set[str] = Field(
        default_factory=set,
        description="Whitelisted IP addresses",
    )
    
    max_failed_attempts: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Max failed auth attempts before lockout",
    )
    
    lockout_duration_minutes: int = Field(
        default=15,
        ge=1,
        le=1440,
        description="Account lockout duration in minutes",
    )
    
    require_2fa: bool = Field(
        default=False,
        description="Require 2FA for sensitive operations",
    )
    
    audit_logging: bool = Field(
        default=True,
        description="Enable audit logging",
    )
    
    @field_validator("ip_whitelist", mode="before")
    @classmethod
    def parse_ips(cls, v: Any) -> Set[str]:
        """Parse comma-separated IP addresses."""
        if isinstance(v, set):
            return v
        if isinstance(v, (list, tuple)):
            return set(v)
        if isinstance(v, str):
            if not v.strip():
                return set()
            return {ip.strip() for ip in v.split(",") if ip.strip()}
        return set()


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

class LoggingSettings(BaseConfig):
    """Logging configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=".env",
        extra="ignore",
    )
    
    level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level",
    )
    
    format: str = Field(
        default="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        description="Log message format",
    )
    
    date_format: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="Log date format",
    )
    
    file_enabled: bool = Field(
        default=True,
        description="Enable file logging",
    )
    
    file_path: Path = Field(
        default=Path("logs/bot.log"),
        description="Log file path",
    )
    
    file_max_bytes: int = Field(
        default=10 * 1024 * 1024,  # 10 MB
        ge=1024,
        description="Max log file size in bytes",
    )
    
    file_backup_count: int = Field(
        default=5,
        ge=0,
        le=100,
        description="Number of backup log files",
    )
    
    json_format: bool = Field(
        default=False,
        description="Use JSON log format",
    )
    
    include_trace: bool = Field(
        default=False,
        description="Include stack traces in logs",
    )


# =============================================================================
# MONITORING CONFIGURATION
# =============================================================================

class MonitoringSettings(BaseConfig):
    """Monitoring and alerting configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="MONITOR_",
        env_file=".env",
        extra="ignore",
    )
    
    enabled: bool = Field(
        default=True,
        description="Enable monitoring",
    )
    
    health_check_interval: int = Field(
        default=60,
        ge=10,
        le=3600,
        description="Health check interval in seconds",
    )
    
    metrics_enabled: bool = Field(
        default=False,
        description="Enable Prometheus metrics",
    )
    
    metrics_port: int = Field(
        default=9090,
        ge=1,
        le=65535,
        description="Metrics server port",
    )
    
    alert_on_error: bool = Field(
        default=True,
        description="Send alerts on errors",
    )
    
    alert_on_trade: bool = Field(
        default=True,
        description="Send alerts on trades",
    )
    
    alert_cooldown_seconds: int = Field(
        default=300,
        ge=0,
        le=3600,
        description="Minimum time between similar alerts",
    )


# =============================================================================
# APPLICATION SETTINGS (MAIN)
# =============================================================================

class Settings(BaseConfig):
    """
    Main application settings aggregating all configuration sections.
    
    Usage:
        settings = Settings()
        # or
        settings = get_settings()
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Application metadata
    app_name: str = Field(
        default="Solana Trading Bot",
        description="Application name",
    )
    
    app_version: str = Field(
        default="1.0.0",
        description="Application version",
    )
    
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment",
    )
    
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )
    
    # Sub-configurations
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    solana: SolanaRPCSettings = Field(default_factory=SolanaRPCSettings)
    wallet: WalletSettings = Field(default_factory=WalletSettings)
    jupiter: JupiterSettings = Field(default_factory=JupiterSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT
    
    def mask_secrets(self) -> dict[str, Any]:
        """
        Return settings dict with sensitive values masked.
        Safe for logging and debugging.
        """
        def mask_value(v: Any) -> Any:
            if isinstance(v, SecretStr):
                secret = v.get_secret_value()
                if len(secret) > 8:
                    return f"{secret[:4]}...{secret[-4:]}"
                return "***"
            elif isinstance(v, dict):
                return {k: mask_value(val) for k, val in v.items()}
            elif isinstance(v, (list, set, tuple)):
                return type(v)(mask_value(item) for item in v)
            return v
        
        data = self.model_dump()
        return mask_value(data)
    
    def to_safe_dict(self) -> dict[str, Any]:
        """Export settings without any secret values."""
        def remove_secrets(d: dict) -> dict:
            result = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    result[k] = remove_secrets(v)
                elif not any(secret in k.lower() for secret in 
                           ["key", "token", "secret", "password", "credential"]):
                    result[k] = v
                else:
                    result[k] = "[REDACTED]"
            return result
        
        return remove_secrets(self.model_dump())


# =============================================================================
# SINGLETON & CACHING
# =============================================================================

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings instance.
    
    Returns:
        Settings: Cached settings singleton
    
    Usage:
        settings = get_settings()
        print(settings.telegram.bot_token)
    """
    return Settings()


def reload_settings() -> Settings:
    """
    Clear settings cache and reload from environment.
    
    Returns:
        Settings: Fresh settings instance
    """
    get_settings.cache_clear()
    return get_settings()


# =============================================================================
# CLI UTILITIES
# =============================================================================

def print_settings_summary(mask_secrets: bool = True) -> None:
    """Print a summary of current settings."""
    settings = get_settings()
    
    print("=" * 60)
    print(f"  {settings.app_name} v{settings.app_version}")
    print(f"  Environment: {settings.environment.value}")
    print("=" * 60)
    
    data = settings.mask_secrets() if mask_secrets else settings.model_dump()
    
    sections = [
        ("Telegram", "telegram"),
        ("Solana RPC", "solana"),
        ("Jupiter", "jupiter"),
        ("Database", "database"),
        ("Risk Management", "risk"),
        ("Security", "security"),
        ("Logging", "logging"),
        ("Monitoring", "monitoring"),
    ]
    
    for title, key in sections:
        print(f"\n{title}:")
        print("-" * 40)
        section_data = data.get(key, {})
        for k, v in section_data.items():
            if isinstance(v, (set, list)):
                v = list(v)[:3]  # Limit displayed items
            print(f"  {k}: {v}")


def validate_settings() -> tuple[bool, list[str]]:
    """
    Validate all settings and return status with any errors.
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    try:
        settings = Settings()
        
        # Additional validation checks
        if settings.is_production:
            if settings.debug:
                errors.append("Debug mode should be disabled in production")
            
            if not settings.security.encryption_key:
                errors.append("Encryption key required in production")
            
            if settings.wallet.auto_approve_below > 0:
                errors.append("Auto-approve should be disabled in production")
            
            if settings.database.echo:
                errors.append("SQL echo should be disabled in production")
                
    except Exception as e:
        errors.append(f"Settings validation failed: {str(e)}")
    
    return len(errors) == 0, errors


def generate_env_template() -> str:
    """Generate a .env template with all available settings."""
    template = """# =============================================================================
# Solana Trading Bot Configuration
# Copy this file to .env and fill in your values
# =============================================================================

# Application
ENVIRONMENT=development
DEBUG=false

# -----------------------------------------------------------------------------
# Telegram Configuration
# -----------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_ADMIN_USER_IDS=123456789,987654321
TELEGRAM_ALLOWED_USER_IDS=
TELEGRAM_RATE_LIMIT_MESSAGES=30
TELEGRAM_WEBHOOK_ENABLED=false
TELEGRAM_WEBHOOK_URL=
TELEGRAM_WEBHOOK_PORT=8443

# -----------------------------------------------------------------------------
# Solana RPC Configuration
# -----------------------------------------------------------------------------
SOLANA_NETWORK=mainnet-beta
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
SOLANA_RPC_URL_BACKUP=
SOLANA_WS_URL=
SOLANA_COMMITMENT=confirmed
SOLANA_TIMEOUT=30
SOLANA_MAX_RETRIES=3
SOLANA_RATE_LIMIT_RPS=10

# -----------------------------------------------------------------------------
# Wallet Configuration
# -----------------------------------------------------------------------------
WALLET_PRIVATE_KEY=your_base58_private_key_here
WALLET_USE_KEYFILE=false
WALLET_KEYFILE_PASSWORD=
WALLET_AUTO_APPROVE_BELOW=0.0
WALLET_REQUIRE_CONFIRMATION=true

# -----------------------------------------------------------------------------
# Jupiter Configuration
# -----------------------------------------------------------------------------
JUPITER_API_URL=https://quote-api.jup.ag/v6
JUPITER_SLIPPAGE_BPS=50
JUPITER_MAX_SLIPPAGE_BPS=500
JUPITER_PRIORITY_FEE_LAMPORTS=10000
JUPITER_DYNAMIC_PRIORITY_FEE=true
JUPITER_MAX_PRIORITY_FEE_LAMPORTS=1000000
JUPITER_EXCLUDE_DEXES=

# -----------------------------------------------------------------------------
# Database Configuration
# -----------------------------------------------------------------------------
DATABASE_URL=sqlite+aiosqlite:///./data/trading_bot.db
DATABASE_ECHO=false
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10
DATABASE_AUTO_MIGRATE=true

# -----------------------------------------------------------------------------
# Risk Management Configuration
# -----------------------------------------------------------------------------
RISK_MAX_POSITION_SIZE_SOL=1.0
RISK_MAX_POSITION_SIZE_PCT=10.0
RISK_MAX_DAILY_TRADES=50
RISK_MAX_DAILY_VOLUME_SOL=10.0
RISK_STOP_LOSS_PCT=5.0
RISK_TAKE_PROFIT_PCT=10.0
RISK_TRAILING_STOP_ENABLED=false
RISK_TRAILING_STOP_PCT=3.0
RISK_MIN_LIQUIDITY_USD=10000.0
RISK_BLACKLISTED_TOKENS=
RISK_WHITELISTED_TOKENS=
RISK_COOL_DOWN_SECONDS=60

# -----------------------------------------------------------------------------
# Security Configuration
# -----------------------------------------------------------------------------
SECURITY_ENCRYPTION_KEY=
SECURITY_API_KEY=
SECURITY_JWT_SECRET=
SECURITY_JWT_EXPIRY_HOURS=24
SECURITY_IP_WHITELIST=
SECURITY_MAX_FAILED_ATTEMPTS=5
SECURITY_LOCKOUT_DURATION_MINUTES=15
SECURITY_REQUIRE_2FA=false
SECURITY_AUDIT_LOGGING=true

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
LOG_LEVEL=INFO
LOG_FILE_ENABLED=true
LOG_FILE_PATH=logs/bot.log
LOG_FILE_MAX_BYTES=10485760
LOG_FILE_BACKUP_COUNT=5
LOG_JSON_FORMAT=false

# -----------------------------------------------------------------------------
# Monitoring Configuration
# -----------------------------------------------------------------------------
MONITOR_ENABLED=true
MONITOR_HEALTH_CHECK_INTERVAL=60
MONITOR_METRICS_ENABLED=false
MONITOR_METRICS_PORT=9090
MONITOR_ALERT_ON_ERROR=true
MONITOR_ALERT_ON_TRADE=true
"""
    return template


# =============================================================================
# MODULE EXPORTS
# =============================================================================

# Convenience singleton - lazy loaded
settings: Settings = None  # type: ignore


def _get_settings_lazy() -> Settings:
    """Lazy load settings on first access."""
    global settings
    if settings is None:
        settings = get_settings()
    return settings


__all__ = [
    # Main classes
    "Settings",
    "TelegramSettings",
    "SolanaRPCSettings",
    "WalletSettings",
    "JupiterSettings",
    "DatabaseSettings",
    "RiskSettings",
    "SecuritySettings",
    "LoggingSettings",
    "MonitoringSettings",
    # Enums
    "Environment",
    "LogLevel",
    "Network",
    # Functions
    "get_settings",
    "reload_settings",
    "validate_settings",
    "print_settings_summary",
    "generate_env_template",
    # Singleton
    "settings",
]


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration management CLI")
    parser.add_argument(
        "--show", 
        action="store_true", 
        help="Show current settings"
    )
    parser.add_argument(
        "--validate", 
        action="store_true", 
        help="Validate settings"
    )
    parser.add_argument(
        "--generate-env", 
        action="store_true", 
        help="Generate .env template"
    )
    parser.add_argument(
        "--unmask", 
        action="store_true", 
        help="Show unmasked secrets (dangerous)"
    )
    
    args = parser.parse_args()
    
    if args.generate_env:
        print(generate_env_template())
    elif args.validate:
        is_valid, errors = validate_settings()
        if is_valid:
            print("[PASS] Settings validation passed")
        else:
            print("[FAIL] Settings validation failed:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
    elif args.show:
        print_settings_summary(mask_secrets=not args.unmask)
    else:
        parser.print_help()
