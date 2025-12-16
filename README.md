<div align="center">

# Solana Trading Bot

### A feature-rich Telegram bot for trading on Solana DEX

[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Solana](https://img.shields.io/badge/Solana-Mainnet-9945ff?style=for-the-badge&logo=solana&logoColor=white)](https://solana.com)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-26a5e4?style=for-the-badge&logo=telegram&logoColor=white)](https://core.telegram.org/bots)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<img src="https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/So11111111111111111111111111111111111111112/logo.png" width="120" alt="Solana">

**Trade tokens - Set alerts - Copy wallets - DCA strategies - Risk management**

[Features](#features) - [Quick Start](#quick-start) - [Configuration](#configuration) - [Commands](#commands) - [Architecture](#architecture)

</div>

---

## Overview

A production-ready Telegram bot that enables trading on Solana through the Jupiter DEX aggregator. Built with async Python for high performance, featuring encrypted wallet storage, comprehensive risk management, and a modular architecture.

```
+-------------------------------------------------------------+
|                    TELEGRAM INTERFACE                       |
+-------------------------------------------------------------+
|  User Commands -> Bot Handler -> Trading Engine -> Jupiter    |
|                                      v                      |
|                              Risk Manager                   |
|                                      v                      |
|                              Transaction Builder -> Solana   |
+-------------------------------------------------------------+
```

---

## Features

<table>
<tr>
<td width="50%">

### Trading
- **Instant Swaps** via Jupiter V6 aggregator
- **DCA Engine** for automated recurring buys
- **Copy Trading** to mirror successful wallets
- **Price Alerts** with custom triggers
- **Paper Trading** mode for testing

</td>
<td width="50%">

### Security
- **AES-256 Encryption** for wallet storage
- **PBKDF2 Key Derivation** (600k iterations)
- **Rate Limiting** to prevent abuse
- **Audit Logging** with tamper detection
- **Input Validation** on all parameters

</td>
</tr>
<tr>
<td>

### Risk Management
- **Position Limits** per trade and total
- **Stop-Loss / Take-Profit** automation
- **Daily Loss Limits** protection
- **Slippage Controls** on all swaps
- **Token Safety Scanner** (rug detection)

</td>
<td>

### Analytics
- **P&L Tracking** per token and overall
- **Trade History** with export options
- **Portfolio Overview** with real-time prices
- **Performance Metrics** and statistics
- **Gas Cost Analysis** tracking

</td>
</tr>
</table>

---

## Quick Start

### Prerequisites

- Python 3.11+
- Telegram Bot Token ([get one from @BotFather](https://t.me/BotFather))
- Solana RPC endpoint (free or paid)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/solana-trading-bot.git
cd solana-trading-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings
```

### Generate Security Keys

```bash
# Generate encryption secret (64 chars)
python -c "import secrets; print(secrets.token_hex(32))"

# Generate encryption salt (32 chars)
python -c "import secrets; print(secrets.token_hex(16))"
```

### Run

```bash
python main.py
```

---

## Configuration

Copy `.env.example` to `.env` and configure:

| Variable | Description | Required |
|----------|-------------|:--------:|
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather | Yes |
| `SOLANA_RPC_URL` | RPC endpoint URL | Yes |
| `ENCRYPTION_SECRET` | 64-char hex for wallet encryption | Yes |
| `ENCRYPTION_SALT` | 32-char hex for key derivation | Yes |
| `PAPER_TRADING` | Start in simulation mode | - |
| `PLATFORM_FEE_PERCENT` | Fee percentage (default: 0.5) | - |

<details>
<summary><b>All Configuration Options</b></summary>

```env
# Network
SOLANA_NETWORK=mainnet-beta
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com

# Trading
PAPER_TRADING=true
MAX_TRADE_SIZE_SOL=10.0
DEFAULT_SLIPPAGE=1.0

# Risk
MAX_POSITION_PERCENT=25.0
DAILY_LOSS_LIMIT_SOL=0

# Features
DCA_ENABLED=true
COPY_TRADING_ENABLED=true
ALERTS_ENABLED=true
```

</details>

---

## Commands

| Command | Description |
|---------|-------------|
| `/start` | Initialize bot and create wallet |
| `/wallet` | View wallet address and balances |
| `/buy <token> <amount>` | Buy token with SOL |
| `/sell <token> <amount>` | Sell token for SOL |
| `/portfolio` | View current holdings |
| `/pnl` | Profit & loss report |
| `/alert <token> <price>` | Set price alert |
| `/dca` | Configure DCA strategy |
| `/copy <wallet>` | Start copy trading |
| `/settings` | Bot configuration |
| `/export` | Export wallet (encrypted) |
| `/help` | Command reference |

---

## Architecture

```
solana_trading_bot/
+-- main.py              # Application entry point
+-- bot.py               # Telegram bot handlers
+-- config.py            # Pydantic configuration
+-- database.py          # SQLite with async support
|
+-- wallet_async.py      # Encrypted wallet management
+-- jupiter_async.py     # Jupiter DEX integration
+-- transaction.py       # Transaction builder + Jito
|
+-- risk_manager.py      # Position & risk controls
+-- token_scanner.py     # Rug pull detection
+-- dca_engine.py        # Dollar-cost averaging
+-- copy_trading.py      # Wallet mirroring
+-- alerts.py            # Price alert system
+-- analytics.py         # P&L and statistics
|
+-- rate_limiter.py      # API rate limiting
+-- audit_secure.py      # Tamper-proof logging
+-- validators.py        # Input sanitization
+-- exceptions.py        # Custom exceptions
+-- retry.py             # Retry with backoff
```

### Key Design Decisions

- **Async-first**: All I/O operations are non-blocking
- **Dependency injection**: Clean component initialization
- **Environment-based config**: No hardcoded values
- **Defensive coding**: Validation at every boundary

---

## RPC Providers

For production, use a paid RPC provider:

| Provider | Free Tier | Notes |
|----------|-----------|-------|
| [Helius](https://helius.xyz) | 100k req/day | Recommended |
| [QuickNode](https://quicknode.com) | Limited | Fast |
| [Triton](https://triton.one) | None | Enterprise |
| Public RPC | Rate limited | Testing only |

---

## Security Considerations

> **Important**: This bot handles real funds. Review these points:

1. **Never share your `.env` file** - Contains encryption keys
2. **Use paper trading first** - Test before real money
3. **Secure your server** - Firewall, SSH keys, updates
4. **Monitor audit logs** - Check for anomalies
5. **Set conservative limits** - Start with small amounts

### Encryption Details

- Wallet private keys encrypted with Fernet (AES-128-CBC)
- Key derived via PBKDF2-HMAC-SHA256 (600,000 iterations)
- Unique salt per installation
- Keys never stored in plaintext

---

## Development

```bash
# Run tests
pytest tests/ -v

# Type checking
mypy . --strict

# Lint
ruff check .
```

---

## Disclaimer

This software is provided for educational purposes. Trading cryptocurrencies involves significant risk. The authors are not responsible for any financial losses incurred through use of this software. Always do your own research and never trade with funds you cannot afford to lose.

---

## License

[MIT License](LICENSE) - feel free to use this in your own projects.

---

<div align="center">

**Built for the Solana ecosystem**

If this project helped you, consider giving it a star!

</div>
