
import asyncio
import logging
import time
import re
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import httpx
import base58
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

from config import config
from database import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SOL_MINT = "So11111111111111111111111111111111111111112"
LAMPORTS_PER_SOL = 1_000_000_000
JUPITER_QUOTE_API = "https://quote-api.jup.ag/v6/quote"
JUPITER_SWAP_API = "https://quote-api.jup.ag/v6/swap"
JUPITER_TOKEN_API = "https://token.jup.ag/strict"

class RateLimiter:
    def __init__(self, max_requests: int = 20, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self._requests: Dict[int, List[float]] = defaultdict(list)

    def is_allowed(self, user_id: int) -> bool:
        now = time.time()
        self._requests[user_id] = [t for t in self._requests[user_id] if t > now - self.window]
        if len(self._requests[user_id]) >= self.max_requests:
            return False
        self._requests[user_id].append(now)
        return True

@dataclass
class TokenInfo:
    address: str
    symbol: str
    name: str
    decimals: int

@dataclass
class SwapQuote:
    input_mint: str
    output_mint: str
    in_amount: int
    out_amount: int
    price_impact: float
    route: str
    raw: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SwapResult:
    success: bool
    signature: str = ""
    error: str = ""

class SecureWallet:
    def __init__(self, private_key: str, rpc_url: str):
        self._rpc_url = rpc_url
        self._keypair = None
        self._public_key = ""
        try:
            from solders.keypair import Keypair
            key_bytes = base58.b58decode(private_key)
            self._keypair = Keypair.from_bytes(key_bytes)
            self._public_key = str(self._keypair.pubkey())
            logger.info(f"Wallet: {self._public_key[:8]}...{self._public_key[-4:]}")
        except Exception as e:
            logger.error(f"Wallet init failed: {e}")

    @property
    def public_key(self) -> str:
        return self._public_key

    @property
    def keypair(self):
        return self._keypair

    async def get_balance(self) -> float:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(self._rpc_url, json={
                    "jsonrpc": "2.0", "id": 1, "method": "getBalance",
                    "params": [self._public_key]
                })
                data = r.json()
                if "result" in data:
                    return data["result"]["value"] / LAMPORTS_PER_SOL
        except Exception as e:
            logger.error(f"Balance error: {e}")
        return 0.0

    async def get_token_balance(self, mint: str) -> Tuple[float, int]:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(self._rpc_url, json={
                    "jsonrpc": "2.0", "id": 1, "method": "getTokenAccountsByOwner",
                    "params": [self._public_key, {"mint": mint}, {"encoding": "jsonParsed"}]
                })
                data = r.json()
                if "result" in data and data["result"]["value"]:
                    info = data["result"]["value"][0]["account"]["data"]["parsed"]["info"]
                    return float(info["tokenAmount"]["uiAmount"] or 0), info["tokenAmount"]["decimals"]
        except Exception as e:
            logger.error(f"Token balance error: {e}")
        return 0.0, 0

class JupiterClient:
    def __init__(self, rpc_url: str):
        self._rpc_url = rpc_url
        self._tokens: Dict[str, TokenInfo] = {}
        self._cache_time = 0

    async def get_token(self, query: str) -> Optional[TokenInfo]:
        if time.time() - self._cache_time > 300:
            await self._load_tokens()

        if len(query) > 30:
            return self._tokens.get(query)

        query_upper = query.upper()
        for t in self._tokens.values():
            if t.symbol.upper() == query_upper:
                return t
        return None

    async def _load_tokens(self):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(JUPITER_TOKEN_API)
                if r.status_code == 200:
                    for t in r.json():
                        self._tokens[t["address"]] = TokenInfo(
                            address=t["address"], symbol=t["symbol"],
                            name=t["name"], decimals=t["decimals"]
                        )
                    self._cache_time = time.time()
                    logger.info(f"Loaded {len(self._tokens)} tokens")
        except Exception as e:
            logger.error(f"Token load error: {e}")

    async def get_quote(self, input_mint: str, output_mint: str, amount: int, slippage: int = 100) -> Optional[SwapQuote]:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(JUPITER_QUOTE_API, params={
                    "inputMint": input_mint, "outputMint": output_mint,
                    "amount": str(amount), "slippageBps": slippage
                })
                if r.status_code != 200:
                    return None
                data = r.json()
                route = " > ".join([s.get("swapInfo", {}).get("label", "?") for s in data.get("routePlan", [])])
                return SwapQuote(
                    input_mint=data["inputMint"], output_mint=data["outputMint"],
                    in_amount=int(data["inAmount"]), out_amount=int(data["outAmount"]),
                    price_impact=float(data.get("priceImpactPct", 0)), route=route or "Direct", raw=data
                )
        except Exception as e:
            logger.error(f"Quote error: {e}")
        return None

    async def execute_swap(self, quote: SwapQuote, wallet: SecureWallet, priority_fee: int = 50000) -> SwapResult:
        try:
            from solders.transaction import VersionedTransaction
            from solders.signature import Signature
            from solana.rpc.async_api import AsyncClient
            from solana.rpc.commitment import Confirmed

            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(JUPITER_SWAP_API, json={
                    "quoteResponse": quote.raw, "userPublicKey": wallet.public_key,
                    "wrapAndUnwrapSol": True, "computeUnitPriceMicroLamports": priority_fee
                })
                if r.status_code != 200:
                    return SwapResult(success=False, error=f"Swap API: {r.status_code}")
                swap_data = r.json()

            tx_bytes = base58.b58decode(swap_data["swapTransaction"])
            tx = VersionedTransaction.from_bytes(tx_bytes)
            signed = VersionedTransaction(tx.message, [wallet.keypair])

            async with AsyncClient(self._rpc_url) as rpc:
                result = await rpc.send_transaction(signed)
                sig = str(result.value)
                await asyncio.sleep(2)
                conf = await rpc.confirm_transaction(Signature.from_string(sig), Confirmed)
                if conf.value[0].err:
                    return SwapResult(success=False, signature=sig, error="TX failed")
                return SwapResult(success=True, signature=sig)
        except Exception as e:
            logger.error(f"Swap error: {e}")
            return SwapResult(success=False, error=str(e))

class TradingBot:
    def __init__(self):
        if not config.telegram.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN required")

        self.db = Database(config.database.path)
        self.rate_limiter = RateLimiter()
        self.wallet: Optional[SecureWallet] = None
        self.jupiter: Optional[JupiterClient] = None

        if config.wallet.private_key:
            self.wallet = SecureWallet(config.wallet.private_key, config.solana.rpc_url)
            self.jupiter = JupiterClient(config.solana.rpc_url)

        self.app = Application.builder().token(config.telegram.bot_token).build()
        self._register_handlers()
        logger.info("TradingBot initialized")

    def _register_handlers(self):
        cmds = [
            ("start", self.cmd_start), ("help", self.cmd_help), ("menu", self.cmd_menu),
            ("wallet", self.cmd_wallet), ("balance", self.cmd_balance),
            ("buy", self.cmd_buy), ("sell", self.cmd_sell),
            ("positions", self.cmd_positions), ("history", self.cmd_history), ("settings", self.cmd_settings)
        ]
        for cmd, handler in cmds:
            self.app.add_handler(CommandHandler(cmd, handler))
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
        self.app.add_error_handler(self.error_handler)

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.error(f"Error: {context.error}")
        if update and update.effective_message:
            await update.effective_message.reply_text("An error occurred.")

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        wallet_status = "Connected" if self.wallet and self.wallet.public_key else "Not configured"
        await update.message.reply_text(
            f"<b>Solana Trading Bot</b>\n\nWelcome {user.first_name}!\n\n"
            f"Wallet: {wallet_status}\nNetwork: {config.solana.network}\n\n"
            f"Commands: /menu /buy /sell /balance /help",
            parse_mode="HTML"
        )

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "<b>Commands</b>\n\n"
            "/buy &lt;token&gt; &lt;sol&gt; - Buy tokens\n"
            "/sell &lt;token&gt; &lt;%&gt; - Sell tokens\n"
            "/balance - SOL balance\n"
            "/wallet - Wallet info\n"
            "/positions - Open positions\n"
            "/menu - Main menu",
            parse_mode="HTML"
        )

    async def cmd_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [
            [InlineKeyboardButton("Buy", callback_data="m_buy"), InlineKeyboardButton("Sell", callback_data="m_sell")],
            [InlineKeyboardButton("Balance", callback_data="m_bal"), InlineKeyboardButton("Positions", callback_data="m_pos")],
            [InlineKeyboardButton("Wallet", callback_data="m_wallet"), InlineKeyboardButton("Settings", callback_data="m_set")]
        ]
        await update.message.reply_text("<b>Menu</b>", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")

    async def cmd_wallet(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.wallet or not self.wallet.public_key:
            await update.message.reply_text("Wallet not configured.")
            return
        bal = await self.wallet.get_balance()
        await update.message.reply_text(
            f"<b>Wallet</b>\n\n<code>{self.wallet.public_key}</code>\n\nBalance: {bal:.4f} SOL",
            parse_mode="HTML"
        )

    async def cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.wallet:
            await update.message.reply_text("Wallet not configured.")
            return
        bal = await self.wallet.get_balance()
        await update.message.reply_text(f"<b>Balance:</b> {bal:.4f} SOL", parse_mode="HTML")

    async def cmd_buy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.rate_limiter.is_allowed(update.effective_user.id):
            await update.message.reply_text("Rate limited. Wait a moment.")
            return
        if not self.wallet or not self.jupiter:
            await update.message.reply_text("Wallet not configured.")
            return
        if not context.args or len(context.args) < 2:
            await update.message.reply_text("Usage: /buy &lt;token&gt; &lt;sol_amount&gt;\nExample: /buy BONK 0.1", parse_mode="HTML")
            return

        token_q, amt_str = context.args[0], context.args[1]
        try:
            amount = float(amt_str)
            if amount <= 0 or amount > config.trading.max_position_size_sol:
                await update.message.reply_text(f"Amount must be 0-{config.trading.max_position_size_sol} SOL")
                return
        except Exception:
            await update.message.reply_text("Invalid amount")
            return

        bal = await self.wallet.get_balance()
        if bal < amount + 0.01:
            await update.message.reply_text(f"Insufficient balance: {bal:.4f} SOL")
            return

        token = await self.jupiter.get_token(token_q)
        if not token:
            await update.message.reply_text(f"Token not found: {token_q}")
            return

        msg = await update.message.reply_text(f"Getting quote for {amount} SOL -> {token.symbol}...")
        quote = await self.jupiter.get_quote(SOL_MINT, token.address, int(amount * LAMPORTS_PER_SOL), config.trading.max_slippage_bps)
        if not quote:
            await msg.edit_text("Failed to get quote")
            return

        out_amt = quote.out_amount / (10 ** token.decimals)
        keyboard = [[
            InlineKeyboardButton("Confirm", callback_data=f"buy_{token.address}_{amount}"),
            InlineKeyboardButton("Cancel", callback_data="cancel")
        ]]
        await msg.edit_text(
            f"<b>Buy Quote</b>\n\nSpend: {amount} SOL\nGet: {out_amt:,.2f} {token.symbol}\n"
            f"Impact: {quote.price_impact:.2f}%\nRoute: {quote.route}",
            reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML"
        )

    async def cmd_sell(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.rate_limiter.is_allowed(update.effective_user.id):
            await update.message.reply_text("Rate limited.")
            return
        if not self.wallet or not self.jupiter:
            await update.message.reply_text("Wallet not configured.")
            return
        if not context.args or len(context.args) < 2:
            await update.message.reply_text("Usage: /sell &lt;token&gt; &lt;percentage&gt;\nExample: /sell BONK 100", parse_mode="HTML")
            return

        token_q, pct_str = context.args[0], context.args[1]
        try:
            pct = float(pct_str)
            if pct <= 0 or pct > 100:
                await update.message.reply_text("Percentage must be 1-100")
                return
        except Exception:
            await update.message.reply_text("Invalid percentage")
            return

        token = await self.jupiter.get_token(token_q)
        if not token:
            await update.message.reply_text(f"Token not found: {token_q}")
            return

        bal, dec = await self.wallet.get_token_balance(token.address)
        if bal <= 0:
            await update.message.reply_text(f"No {token.symbol} to sell")
            return

        sell_amt = bal * (pct / 100)
        sell_raw = int(sell_amt * (10 ** dec))

        msg = await update.message.reply_text(f"Getting quote to sell {sell_amt:,.2f} {token.symbol}...")
        quote = await self.jupiter.get_quote(token.address, SOL_MINT, sell_raw, config.trading.max_slippage_bps)
        if not quote:
            await msg.edit_text("Failed to get quote")
            return

        out_sol = quote.out_amount / LAMPORTS_PER_SOL
        keyboard = [[
            InlineKeyboardButton("Confirm", callback_data=f"sell_{token.address}_{sell_raw}"),
            InlineKeyboardButton("Cancel", callback_data="cancel")
        ]]
        await msg.edit_text(
            f"<b>Sell Quote</b>\n\nSell: {sell_amt:,.2f} {token.symbol}\nGet: {out_sol:.4f} SOL\n"
            f"Impact: {quote.price_impact:.2f}%",
            reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML"
        )

    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        positions = self.db.get_open_positions()
        if not positions:
            await update.message.reply_text("No open positions.")
            return
        text = "<b>Positions</b>\n\n"
        for p in positions:
            text += f"{p.token_symbol}: {p.amount:,.2f}\n"
        await update.message.reply_text(text, parse_mode="HTML")

    async def cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        trades = self.db.get_recent_trades(10)
        if not trades:
            await update.message.reply_text("No history.")
            return
        text = "<b>History</b>\n\n"
        for t in trades:
            text += f"{t.side} {t.token_symbol}: {t.sol_amount:.4f} SOL\n"
        await update.message.reply_text(text, parse_mode="HTML")

    async def cmd_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            f"<b>Settings</b>\n\nMax Position: {config.trading.max_position_size_sol} SOL\n"
            f"Slippage: {config.trading.max_slippage_bps/100}%\n"
            f"Stop Loss: {config.trading.stop_loss_percent}%\n"
            f"Take Profit: {config.trading.take_profit_percent}%\n"
            f"Network: {config.solana.network}",
            parse_mode="HTML"
        )

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        data = query.data

        if data == "cancel":
            await query.edit_message_text("Cancelled.")
            return

        if data.startswith("buy_"):
            parts = data.split("_")
            if len(parts) >= 3:
                token_mint, amount = parts[1], float(parts[2])
                await query.edit_message_text("Executing buy...")

                quote = await self.jupiter.get_quote(SOL_MINT, token_mint, int(amount * LAMPORTS_PER_SOL), config.trading.max_slippage_bps)
                if not quote:
                    await query.edit_message_text("Quote expired. Try again.")
                    return

                result = await self.jupiter.execute_swap(quote, self.wallet)
                if result.success:
                    token = await self.jupiter.get_token(token_mint)
                    sym = token.symbol if token else "tokens"
                    await query.edit_message_text(
                        f"<b>Buy Success!</b>\n\nSpent: {amount} SOL\nReceived: {sym}\n\n"
                        f"<a href=\"https://solscan.io/tx/{result.signature}\">View TX</a>",
                        parse_mode="HTML", disable_web_page_preview=True
                    )
                    self.db.record_trade(token_mint, sym, "BUY", amount, quote.out_amount, result.signature)
                else:
                    await query.edit_message_text(f"Buy failed: {result.error}")

        elif data.startswith("sell_"):
            parts = data.split("_")
            if len(parts) >= 3:
                token_mint, amount_raw = parts[1], int(parts[2])
                await query.edit_message_text("Executing sell...")

                quote = await self.jupiter.get_quote(token_mint, SOL_MINT, amount_raw, config.trading.max_slippage_bps)
                if not quote:
                    await query.edit_message_text("Quote expired. Try again.")
                    return

                result = await self.jupiter.execute_swap(quote, self.wallet)
                if result.success:
                    token = await self.jupiter.get_token(token_mint)
                    sym = token.symbol if token else "tokens"
                    sol_received = quote.out_amount / LAMPORTS_PER_SOL
                    await query.edit_message_text(
                        f"<b>Sell Success!</b>\n\nSold: {sym}\nReceived: {sol_received:.4f} SOL\n\n"
                        f"<a href=\"https://solscan.io/tx/{result.signature}\">View TX</a>",
                        parse_mode="HTML", disable_web_page_preview=True
                    )
                    self.db.record_trade(token_mint, sym, "SELL", sol_received, amount_raw, result.signature)
                else:
                    await query.edit_message_text(f"Sell failed: {result.error}")

        elif data == "m_buy":
            await query.edit_message_text("Usage: /buy &lt;token&gt; &lt;sol&gt;\nExample: /buy BONK 0.1", parse_mode="HTML")
        elif data == "m_sell":
            await query.edit_message_text("Usage: /sell &lt;token&gt; &lt;%&gt;\nExample: /sell BONK 100", parse_mode="HTML")
        elif data == "m_bal":
            if self.wallet:
                bal = await self.wallet.get_balance()
                await query.edit_message_text(f"<b>Balance:</b> {bal:.4f} SOL", parse_mode="HTML")
            else:
                await query.edit_message_text("Wallet not configured.")
        elif data == "m_pos":
            pos = self.db.get_open_positions()
            if not pos:
                await query.edit_message_text("No positions.")
            else:
                text = "<b>Positions</b>\n\n" + "\n".join([f"{p.token_symbol}: {p.amount:,.2f}" for p in pos])
                await query.edit_message_text(text, parse_mode="HTML")
        elif data == "m_wallet":
            if self.wallet:
                bal = await self.wallet.get_balance()
                await query.edit_message_text(f"<b>Wallet</b>\n\n<code>{self.wallet.public_key}</code>\n\n{bal:.4f} SOL", parse_mode="HTML")
            else:
                await query.edit_message_text("Wallet not configured.")
        elif data == "m_set":
            await query.edit_message_text(f"<b>Settings</b>\n\nMax: {config.trading.max_position_size_sol} SOL\nSlippage: {config.trading.max_slippage_bps/100}%", parse_mode="HTML")

    def run(self):
        logger.info("Starting bot...")
        self.app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
