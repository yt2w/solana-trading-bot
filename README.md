<div align="center">

# Solana Trading Bot

### A feature-rich Telegram bot for trading on Solana DEX

[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Solana](https://img.shields.io/badge/Solana-Mainnet-9945ff?style=for-the-badge&logo=solana&logoColor=white)](https://solana.com)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-26a5e4?style=for-the-badge&logo=telegram&logoColor=white)](https://core.telegram.org/bots)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Trade tokens - Set alerts - Copy wallets - DCA strategies - Risk management**

</div>

---

## Overview

A production-ready Telegram bot that enables trading on Solana through the Jupiter DEX aggregator. Built with async Python for high performance, featuring encrypted wallet storage, comprehensive risk management, and a modular architecture.

---

## Features

### Trading
- **Instant Swaps** via Jupiter V6 aggregator
- **DCA Engine** for automated recurring buys
- **Copy Trading** to mirror successful wallets
- **Price Alerts** with custom triggers
- **Paper Trading** mode for testing

### Security
- **AES-256 Encryption** for wallet storage
- **PBKDF2 Key Derivation** (100k iterations)
- **Rate Limiting** to prevent abuse
- **Audit Logging** with tamper detection
- **Input Validation** on all parameters

### Risk Management
- **Position Limits** per trade and total
- **Stop-Loss / Take-Profit** automation
- **Daily Loss Limits** protection
- **Slippage Controls** on all swaps
- **Token Safety Scanner** (rug detection)

### Analytics
- **P&L Tracking** per token and overall
- **Trade History** with export options
- **Portfolio Overview** with real-time prices
- **Performance Metrics** and statistics
- **Gas Cost Analysis** tracking

---

## Quick Start

### Prerequisites

- Python 3.11+
- Telegram Bot Token ([get one from @BotFather](https://t.me/BotFather))
- Solana RPC endpoint (free or paid)

### Installation

```bash
# Clone the repository
git clone https://github.com/yt2w/solana-trading-bot.git
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

### Run

```bash
python main.py
```

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
| `/help` | Command reference |

---

## License

[MIT License](LICENSE)
