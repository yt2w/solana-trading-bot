"""
Solana Trading Bot

A Telegram-based trading bot for Solana tokens using Jupiter aggregator.
"""

__version__ = "1.0.0"
__author__ = "Trading Bot Developer"

from .config import config, Tokens
from .database import get_database, Trade, Position
from .bot import get_bot, TradingBot

__all__ = [
    "config",
    "Tokens",
    "get_database",
    "Trade",
    "Position",
    "get_bot",
    "TradingBot",
]
