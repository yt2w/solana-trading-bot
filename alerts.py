
import asyncio
import aiohttp
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
from abc import ABC, abstractmethod
import aiofiles
from pathlib import Path

logger = logging.getLogger(__name__)

class AlertType(Enum):
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    PRICE_CHANGE_PCT = "price_change_pct"
    VOLUME_SPIKE = "volume_spike"
    VOLUME_ABOVE = "volume_above"
    WHALE_MOVEMENT = "whale_movement"
    LARGE_TRANSFER = "large_transfer"
    TOKEN_LISTING = "token_listing"
    LP_CHANGE = "lp_change"
    LP_ADDED = "lp_added"
    LP_REMOVED = "lp_removed"
    SAFETY_ALERT = "safety_alert"
    RUG_WARNING = "rug_warning"
    HONEYPOT_DETECTED = "honeypot_detected"
    POSITION_STOP_LOSS = "position_stop_loss"
    POSITION_TAKE_PROFIT = "position_take_profit"
    POSITION_LIQUIDATION = "position_liquidation"
    DCA_EXECUTED = "dca_executed"
    DCA_COMPLETED = "dca_completed"
    COPY_TRADE = "copy_trade"
    MARKET_CAP_ABOVE = "market_cap_above"
    MARKET_CAP_BELOW = "market_cap_below"
    HOLDER_COUNT_CHANGE = "holder_count_change"
    CUSTOM = "custom"

class NotificationChannel(Enum):
    TELEGRAM = "telegram"
    WEBHOOK = "webhook"
    DISCORD = "discord"
    EMAIL = "email"
    IN_APP = "in_app"

class AlertRepeat(Enum):
    ONCE = "once"
    ALWAYS = "always"
    DAILY = "daily"
    HOURLY = "hourly"

class AlertPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

class Comparison(Enum):
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    EQUAL = "eq"
    GREATER_EQUAL = "gte"
    LESS_EQUAL = "lte"
    CHANGE_UP = "change_up"
    CHANGE_DOWN = "change_down"
    CHANGE_ANY = "change_any"

@dataclass
class AlertConfig:
    alert_id: str
    user_id: str
    alert_type: AlertType
    token_mint: Optional[str] = None
    token_symbol: Optional[str] = None
    threshold_value: float = 0.0
    comparison: Comparison = Comparison.GREATER_THAN
    notification_channels: List[NotificationChannel] = field(default_factory=lambda: [NotificationChannel.TELEGRAM])
    repeat: AlertRepeat = AlertRepeat.ONCE
    cooldown_seconds: int = 300
    priority: AlertPriority = AlertPriority.MEDIUM
    active: bool = True
    name: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.alert_type, str):
            self.alert_type = AlertType(self.alert_type)
        if isinstance(self.comparison, str):
            self.comparison = Comparison(self.comparison)
        if isinstance(self.repeat, str):
            self.repeat = AlertRepeat(self.repeat)
        if isinstance(self.priority, (int, str)):
            self.priority = AlertPriority(int(self.priority)) if isinstance(self.priority, str) else AlertPriority(self.priority)
        if self.notification_channels and isinstance(self.notification_channels[0], str):
            self.notification_channels = [NotificationChannel(c) for c in self.notification_channels]
    
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def is_on_cooldown(self) -> bool:
        if self.last_triggered is None:
            return False
        cooldown_end = self.last_triggered + timedelta(seconds=self.cooldown_seconds)
        return datetime.utcnow() < cooldown_end
    
    def can_trigger(self) -> bool:
        return self.active and not self.is_expired() and not self.is_on_cooldown()
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['alert_type'] = self.alert_type.value
        data['comparison'] = self.comparison.value
        data['repeat'] = self.repeat.value
        data['priority'] = self.priority.value
        data['notification_channels'] = [c.value for c in self.notification_channels]
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['expires_at'] = self.expires_at.isoformat() if self.expires_at else None
        data['last_triggered'] = self.last_triggered.isoformat() if self.last_triggered else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertConfig':
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if 'expires_at' in data and data['expires_at'] and isinstance(data['expires_at'], str):
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        if 'last_triggered' in data and data['last_triggered'] and isinstance(data['last_triggered'], str):
            data['last_triggered'] = datetime.fromisoformat(data['last_triggered'])
        return cls(**data)

@dataclass
class TriggeredAlert:
    trigger_id: str
    alert_id: str
    user_id: str
    alert_type: AlertType
    token_mint: Optional[str]
    triggered_at: datetime
    trigger_value: float
    threshold_value: float
    message: str
    notification_sent: bool = False
    notification_channels_sent: List[NotificationChannel] = field(default_factory=list)
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['alert_type'] = self.alert_type.value
        data['triggered_at'] = self.triggered_at.isoformat()
        data['notification_channels_sent'] = [c.value for c in self.notification_channels_sent]
        return data

@dataclass 
class DoNotDisturbConfig:
    enabled: bool = False
    start_hour: int = 22
    end_hour: int = 8
    timezone: str = "UTC"
    allow_critical: bool = True

@dataclass
class RateLimitConfig:
    max_per_minute: int = 10
    max_per_hour: int = 60
    max_per_day: int = 500
    burst_limit: int = 5

class NotificationProvider(ABC):
    
    @abstractmethod
    async def send(self, user_id: str, message: str, **kwargs) -> bool:
        pass
    
    @abstractmethod
    def format_message(self, alert: TriggeredAlert, detailed: bool = False) -> str:
        pass

class TelegramNotificationProvider(NotificationProvider):
    
    def __init__(self, bot_token: str, default_chat_id: Optional[str] = None):
        self.bot_token = bot_token
        self.default_chat_id = default_chat_id
        self.api_base = f"https://api.telegram.org/bot{bot_token}"
    
    async def send(self, user_id: str, message: str, chat_id: Optional[str] = None, 
                   parse_mode: str = "HTML", **kwargs) -> bool:
        target_chat = chat_id or self.default_chat_id or user_id
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "chat_id": target_chat,
                    "text": message,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": kwargs.get("disable_preview", True)
                }
                
                async with session.post(
                    f"{self.api_base}/sendMessage",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        logger.debug(f"Telegram notification sent to {target_chat}")
                        return True
                    else:
                        error = await resp.text()
                        logger.error(f"Telegram send failed: {error}")
                        return False
        except Exception as e:
            logger.error(f"Telegram notification error: {e}")
            return False
    
    def format_message(self, alert: TriggeredAlert, detailed: bool = False) -> str:
        emoji_map = {
            AlertType.PRICE_ABOVE: "[UP]",
            AlertType.PRICE_BELOW: "[DOWN]",
            AlertType.PRICE_CHANGE_PCT: "[CHANGE]",
            AlertType.VOLUME_SPIKE: "[VOLUME]",
            AlertType.WHALE_MOVEMENT: "[WHALE]",
            AlertType.SAFETY_ALERT: "[WARN]",
            AlertType.RUG_WARNING: "[ALERT]",
            AlertType.POSITION_STOP_LOSS: "[SL]",
            AlertType.POSITION_TAKE_PROFIT: "[TP]",
            AlertType.DCA_EXECUTED: "[DCA]",
            AlertType.COPY_TRADE: "[COPY]",
            AlertType.LP_ADDED: "[LP+]",
            AlertType.LP_REMOVED: "[LP-]",
        }
        
        emoji = emoji_map.get(alert.alert_type, "[ALERT]")
        
        if detailed:
            token_display = "N/A"
            if alert.token_mint:
                token_display = f"{alert.token_mint[:8]}...{alert.token_mint[-6:]}"
            msg = f"{emoji} <b>ALERT: {alert.alert_type.value.upper()}</b>\n\n"
            msg += f"Token: {token_display}\n"
            msg += f"Trigger Value: {alert.trigger_value:,.6f}\n"
            msg += f"Threshold: {alert.threshold_value:,.6f}\n"
            msg += f"Time: {alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            msg += f"\n{alert.message}"
        else:
            msg = f"{emoji} {alert.message}"
        
        return msg

class WebhookNotificationProvider(NotificationProvider):
    
    def __init__(self, default_url: Optional[str] = None, headers: Optional[Dict] = None):
        self.default_url = default_url
        self.headers = headers or {"Content-Type": "application/json"}
    
    async def send(self, user_id: str, message: str, webhook_url: Optional[str] = None, **kwargs) -> bool:
        url = webhook_url or self.default_url
        if not url:
            logger.error("No webhook URL provided")
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "user_id": user_id,
                    "message": message,
                    "timestamp": datetime.utcnow().isoformat(),
                    **kwargs
                }
                
                async with session.post(
                    url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status in (200, 201, 202, 204):
                        logger.debug(f"Webhook sent to {url}")
                        return True
                    else:
                        error = await resp.text()
                        logger.error(f"Webhook failed ({resp.status}): {error}")
                        return False
        except Exception as e:
            logger.error(f"Webhook error: {e}")
            return False
    
    def format_message(self, alert: TriggeredAlert, detailed: bool = False) -> str:
        return json.dumps(alert.to_dict())

class DiscordNotificationProvider(NotificationProvider):
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
    
    async def send(self, user_id: str, message: str, webhook_url: Optional[str] = None, **kwargs) -> bool:
        url = webhook_url or self.webhook_url
        if not url:
            logger.error("No Discord webhook URL")
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "content": message,
                    "username": kwargs.get("username", "Trading Bot"),
                }
                
                if kwargs.get("embed"):
                    payload["embeds"] = [kwargs["embed"]]
                
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    return resp.status in (200, 204)
        except Exception as e:
            logger.error(f"Discord error: {e}")
            return False
    
    def format_message(self, alert: TriggeredAlert, detailed: bool = False) -> str:
        return f"**{alert.alert_type.value.upper()}**\n{alert.message}"

class AlertChecker:
    
    @staticmethod
    def check_price_alert(alert: AlertConfig, current_price: float, 
                          previous_price: Optional[float] = None) -> Optional[TriggeredAlert]:
        if not alert.can_trigger():
            return None
        
        triggered = False
        trigger_value = current_price
        message = ""
        
        if alert.alert_type == AlertType.PRICE_ABOVE:
            if current_price > alert.threshold_value:
                triggered = True
                message = f"Price ${current_price:,.6f} is above ${alert.threshold_value:,.6f}"
        
        elif alert.alert_type == AlertType.PRICE_BELOW:
            if current_price < alert.threshold_value:
                triggered = True
                message = f"Price ${current_price:,.6f} is below ${alert.threshold_value:,.6f}"
        
        elif alert.alert_type == AlertType.PRICE_CHANGE_PCT and previous_price:
            pct_change = ((current_price - previous_price) / previous_price) * 100
            trigger_value = pct_change
            
            if alert.comparison == Comparison.CHANGE_UP and pct_change >= alert.threshold_value:
                triggered = True
                message = f"Price increased {pct_change:+.2f}% (threshold: {alert.threshold_value}%)"
            elif alert.comparison == Comparison.CHANGE_DOWN and pct_change <= -alert.threshold_value:
                triggered = True
                message = f"Price decreased {pct_change:+.2f}% (threshold: -{alert.threshold_value}%)"
            elif alert.comparison == Comparison.CHANGE_ANY and abs(pct_change) >= alert.threshold_value:
                triggered = True
                message = f"Price changed {pct_change:+.2f}% (threshold: +/-{alert.threshold_value}%)"
        
        if triggered:
            return TriggeredAlert(
                trigger_id=str(uuid.uuid4()),
                alert_id=alert.alert_id,
                user_id=alert.user_id,
                alert_type=alert.alert_type,
                token_mint=alert.token_mint,
                triggered_at=datetime.utcnow(),
                trigger_value=trigger_value,
                threshold_value=alert.threshold_value,
                message=message
            )
        
        return None
    
    @staticmethod
    def check_volume_alert(alert: AlertConfig, current_volume: float, 
                           avg_volume: float) -> Optional[TriggeredAlert]:
        if not alert.can_trigger():
            return None
        
        triggered = False
        message = ""
        
        if alert.alert_type == AlertType.VOLUME_SPIKE:
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            if volume_ratio >= alert.threshold_value:
                triggered = True
                message = f"Volume spike: {volume_ratio:.1f}x average (threshold: {alert.threshold_value}x)"
        
        elif alert.alert_type == AlertType.VOLUME_ABOVE:
            if current_volume > alert.threshold_value:
                triggered = True
                message = f"Volume ${current_volume:,.0f} exceeds ${alert.threshold_value:,.0f}"
        
        if triggered:
            return TriggeredAlert(
                trigger_id=str(uuid.uuid4()),
                alert_id=alert.alert_id,
                user_id=alert.user_id,
                alert_type=alert.alert_type,
                token_mint=alert.token_mint,
                triggered_at=datetime.utcnow(),
                trigger_value=current_volume,
                threshold_value=alert.threshold_value,
                message=message
            )
        
        return None
    
    @staticmethod
    def check_whale_alert(alert: AlertConfig, transfer_amount: float, 
                          wallet_address: str) -> Optional[TriggeredAlert]:
        if not alert.can_trigger():
            return None
        
        if transfer_amount >= alert.threshold_value:
            return TriggeredAlert(
                trigger_id=str(uuid.uuid4()),
                alert_id=alert.alert_id,
                user_id=alert.user_id,
                alert_type=alert.alert_type,
                token_mint=alert.token_mint,
                triggered_at=datetime.utcnow(),
                trigger_value=transfer_amount,
                threshold_value=alert.threshold_value,
                message=f"Whale movement: {transfer_amount:,.2f} tokens from {wallet_address[:8]}...",
                extra_data={"wallet": wallet_address}
            )
        
        return None
    
    @staticmethod
    def check_market_cap_alert(alert: AlertConfig, current_mcap: float) -> Optional[TriggeredAlert]:
        if not alert.can_trigger():
            return None
        
        triggered = False
        message = ""
        
        if alert.alert_type == AlertType.MARKET_CAP_ABOVE and current_mcap > alert.threshold_value:
            triggered = True
            message = f"Market cap ${current_mcap:,.0f} exceeds ${alert.threshold_value:,.0f}"
        
        elif alert.alert_type == AlertType.MARKET_CAP_BELOW and current_mcap < alert.threshold_value:
            triggered = True
            message = f"Market cap ${current_mcap:,.0f} below ${alert.threshold_value:,.0f}"
        
        if triggered:
            return TriggeredAlert(
                trigger_id=str(uuid.uuid4()),
                alert_id=alert.alert_id,
                user_id=alert.user_id,
                alert_type=alert.alert_type,
                token_mint=alert.token_mint,
                triggered_at=datetime.utcnow(),
                trigger_value=current_mcap,
                threshold_value=alert.threshold_value,
                message=message
            )
        
        return None

class AlertManager:
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        rate_limit: Optional[RateLimitConfig] = None
    ):
        self.storage_path = storage_path or Path("./data/alerts")
        self.rate_limit = rate_limit or RateLimitConfig()
        
        self.alerts: Dict[str, AlertConfig] = {}
        self.alerts_by_user: Dict[str, Set[str]] = defaultdict(set)
        self.alerts_by_token: Dict[str, Set[str]] = defaultdict(set)
        
        self.triggered_history: List[TriggeredAlert] = []
        self.max_history = 10000
        
        self.providers: Dict[NotificationChannel, NotificationProvider] = {}
        
        self.user_dnd: Dict[str, DoNotDisturbConfig] = {}
        self.user_channels: Dict[str, Dict[str, str]] = {}
        
        self._notification_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._rate_limit_reset: Dict[str, datetime] = {}
        
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._price_cache: Dict[str, float] = {}
        self._previous_prices: Dict[str, float] = {}
        
        self._on_alert_triggered: List[Callable] = []
        self._price_fetcher: Optional[Callable] = None
    
    async def initialize(self):
        self.storage_path.mkdir(parents=True, exist_ok=True)
        await self._load_alerts()
        logger.info(f"AlertManager initialized with {len(self.alerts)} alerts")
    
    def register_provider(self, channel: NotificationChannel, provider: NotificationProvider):
        self.providers[channel] = provider
        logger.info(f"Registered {channel.value} notification provider")
    
    def set_telegram_provider(self, bot_token: str, default_chat_id: Optional[str] = None):
        self.register_provider(
            NotificationChannel.TELEGRAM,
            TelegramNotificationProvider(bot_token, default_chat_id)
        )
    
    def set_webhook_provider(self, default_url: Optional[str] = None):
        self.register_provider(
            NotificationChannel.WEBHOOK,
            WebhookNotificationProvider(default_url)
        )
    
    def set_price_fetcher(self, fetcher: Callable):
        self._price_fetcher = fetcher

    async def create_alert(
        self,
        user_id: str,
        alert_type: AlertType,
        threshold_value: float,
        token_mint: Optional[str] = None,
        comparison: Comparison = Comparison.GREATER_THAN,
        channels: Optional[List[NotificationChannel]] = None,
        repeat: AlertRepeat = AlertRepeat.ONCE,
        cooldown_seconds: int = 300,
        priority: AlertPriority = AlertPriority.MEDIUM,
        name: Optional[str] = None,
        expires_in_hours: Optional[int] = None,
        **extra_data
    ) -> AlertConfig:
        alert_id = str(uuid.uuid4())
        
        expires_at = None
        if expires_in_hours:
            expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
        
        alert = AlertConfig(
            alert_id=alert_id,
            user_id=user_id,
            alert_type=alert_type,
            token_mint=token_mint,
            threshold_value=threshold_value,
            comparison=comparison,
            notification_channels=channels or [NotificationChannel.TELEGRAM],
            repeat=repeat,
            cooldown_seconds=cooldown_seconds,
            priority=priority,
            name=name or f"{alert_type.value}_{threshold_value}",
            expires_at=expires_at,
            extra_data=extra_data
        )
        
        self.alerts[alert_id] = alert
        self.alerts_by_user[user_id].add(alert_id)
        
        if token_mint:
            self.alerts_by_token[token_mint].add(alert_id)
        
        await self._save_alerts()
        
        logger.info(f"Created alert {alert_id} for user {user_id}: {alert_type.value}")
        return alert
    
    async def delete_alert(self, alert_id: str) -> bool:
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        
        del self.alerts[alert_id]
        self.alerts_by_user[alert.user_id].discard(alert_id)
        
        if alert.token_mint:
            self.alerts_by_token[alert.token_mint].discard(alert_id)
        
        await self._save_alerts()
        
        logger.info(f"Deleted alert {alert_id}")
        return True
    
    async def toggle_alert(self, alert_id: str) -> Optional[bool]:
        if alert_id not in self.alerts:
            return None
        
        self.alerts[alert_id].active = not self.alerts[alert_id].active
        self.alerts[alert_id].updated_at = datetime.utcnow()
        
        await self._save_alerts()
        
        return self.alerts[alert_id].active
    
    async def update_alert(self, alert_id: str, **updates) -> Optional[AlertConfig]:
        if alert_id not in self.alerts:
            return None
        
        alert = self.alerts[alert_id]
        
        for key, value in updates.items():
            if hasattr(alert, key):
                setattr(alert, key, value)
        
        alert.updated_at = datetime.utcnow()
        await self._save_alerts()
        
        return alert
    
    def get_alert(self, alert_id: str) -> Optional[AlertConfig]:
        return self.alerts.get(alert_id)
    
    def list_alerts(
        self,
        user_id: Optional[str] = None,
        token_mint: Optional[str] = None,
        alert_type: Optional[AlertType] = None,
        active_only: bool = False
    ) -> List[AlertConfig]:
        if user_id:
            alert_ids = self.alerts_by_user.get(user_id, set())
        elif token_mint:
            alert_ids = self.alerts_by_token.get(token_mint, set())
        else:
            alert_ids = set(self.alerts.keys())
        
        alerts = [self.alerts[aid] for aid in alert_ids if aid in self.alerts]
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        if active_only:
            alerts = [a for a in alerts if a.active and not a.is_expired()]
        
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)
    
    def get_triggered_alerts(
        self,
        user_id: Optional[str] = None,
        limit: int = 100,
        since: Optional[datetime] = None
    ) -> List[TriggeredAlert]:
        history = self.triggered_history
        
        if user_id:
            history = [t for t in history if t.user_id == user_id]
        
        if since:
            history = [t for t in history if t.triggered_at >= since]
        
        return sorted(history, key=lambda t: t.triggered_at, reverse=True)[:limit]

    async def create_price_above_alert(
        self,
        user_id: str,
        token_mint: str,
        price: float,
        **kwargs
    ) -> AlertConfig:
        return await self.create_alert(
            user_id=user_id,
            alert_type=AlertType.PRICE_ABOVE,
            token_mint=token_mint,
            threshold_value=price,
            comparison=Comparison.GREATER_THAN,
            **kwargs
        )
    
    async def create_price_below_alert(
        self,
        user_id: str,
        token_mint: str,
        price: float,
        **kwargs
    ) -> AlertConfig:
        return await self.create_alert(
            user_id=user_id,
            alert_type=AlertType.PRICE_BELOW,
            token_mint=token_mint,
            threshold_value=price,
            comparison=Comparison.LESS_THAN,
            **kwargs
        )
    
    async def create_percent_change_alert(
        self,
        user_id: str,
        token_mint: str,
        percent: float,
        direction: str = "any",
        **kwargs
    ) -> AlertConfig:
        comparison_map = {
            "up": Comparison.CHANGE_UP,
            "down": Comparison.CHANGE_DOWN,
            "any": Comparison.CHANGE_ANY
        }
        
        return await self.create_alert(
            user_id=user_id,
            alert_type=AlertType.PRICE_CHANGE_PCT,
            token_mint=token_mint,
            threshold_value=abs(percent),
            comparison=comparison_map.get(direction, Comparison.CHANGE_ANY),
            **kwargs
        )
    
    async def create_stop_loss_alert(
        self,
        user_id: str,
        token_mint: str,
        stop_price: float,
        position_id: Optional[str] = None,
        **kwargs
    ) -> AlertConfig:
        return await self.create_alert(
            user_id=user_id,
            alert_type=AlertType.POSITION_STOP_LOSS,
            token_mint=token_mint,
            threshold_value=stop_price,
            comparison=Comparison.LESS_EQUAL,
            priority=AlertPriority.HIGH,
            position_id=position_id,
            **kwargs
        )
    
    async def create_take_profit_alert(
        self,
        user_id: str,
        token_mint: str,
        target_price: float,
        position_id: Optional[str] = None,
        **kwargs
    ) -> AlertConfig:
        return await self.create_alert(
            user_id=user_id,
            alert_type=AlertType.POSITION_TAKE_PROFIT,
            token_mint=token_mint,
            threshold_value=target_price,
            comparison=Comparison.GREATER_EQUAL,
            priority=AlertPriority.HIGH,
            position_id=position_id,
            **kwargs
        )
    
    async def create_whale_alert(
        self,
        user_id: str,
        token_mint: str,
        min_amount: float,
        **kwargs
    ) -> AlertConfig:
        return await self.create_alert(
            user_id=user_id,
            alert_type=AlertType.WHALE_MOVEMENT,
            token_mint=token_mint,
            threshold_value=min_amount,
            repeat=AlertRepeat.ALWAYS,
            **kwargs
        )

    async def check_price_alerts(self, token_mint: str, current_price: float) -> List[TriggeredAlert]:
        triggered = []
        
        previous_price = self._previous_prices.get(token_mint)
        self._previous_prices[token_mint] = self._price_cache.get(token_mint, current_price)
        self._price_cache[token_mint] = current_price
        
        alert_ids = self.alerts_by_token.get(token_mint, set())
        
        for alert_id in list(alert_ids):
            alert = self.alerts.get(alert_id)
            if not alert or not alert.active:
                continue
            
            if alert.alert_type in (AlertType.PRICE_ABOVE, AlertType.PRICE_BELOW, AlertType.PRICE_CHANGE_PCT):
                result = AlertChecker.check_price_alert(alert, current_price, previous_price)
                
                if result:
                    triggered.append(result)
                    await self._handle_triggered_alert(alert, result)
        
        return triggered
    
    async def check_all_alerts(self, market_data: Dict[str, Dict[str, float]]) -> List[TriggeredAlert]:
        all_triggered = []
        
        for token_mint, data in market_data.items():
            if "price" in data:
                triggered = await self.check_price_alerts(token_mint, data["price"])
                all_triggered.extend(triggered)
        
        return all_triggered
    
    async def trigger_alert_manually(
        self,
        alert_type: AlertType,
        user_id: str,
        message: str,
        token_mint: Optional[str] = None,
        trigger_value: float = 0,
        **extra_data
    ) -> TriggeredAlert:
        triggered = TriggeredAlert(
            trigger_id=str(uuid.uuid4()),
            alert_id="manual",
            user_id=user_id,
            alert_type=alert_type,
            token_mint=token_mint,
            triggered_at=datetime.utcnow(),
            trigger_value=trigger_value,
            threshold_value=0,
            message=message,
            extra_data=extra_data
        )
        
        channels = [NotificationChannel.TELEGRAM]
        
        await self._send_notifications(user_id, triggered, channels)
        self._add_to_history(triggered)
        
        return triggered
    
    async def _handle_triggered_alert(self, alert: AlertConfig, triggered: TriggeredAlert):
        alert.last_triggered = datetime.utcnow()
        alert.trigger_count += 1
        
        if alert.repeat == AlertRepeat.ONCE:
            alert.active = False
        
        await self._send_notifications(alert.user_id, triggered, alert.notification_channels)
        
        self._add_to_history(triggered)
        
        for callback in self._on_alert_triggered:
            try:
                await callback(triggered)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        await self._save_alerts()
    
    def _add_to_history(self, triggered: TriggeredAlert):
        self.triggered_history.append(triggered)
        
        if len(self.triggered_history) > self.max_history:
            self.triggered_history = self.triggered_history[-self.max_history:]

    async def _send_notifications(
        self,
        user_id: str,
        triggered: TriggeredAlert,
        channels: List[NotificationChannel]
    ):
        if not self._can_notify_user(user_id, triggered.alert_type):
            logger.debug(f"User {user_id} is in DND mode")
            return
        
        if not self._check_rate_limit(user_id):
            logger.warning(f"Rate limit exceeded for user {user_id}")
            return
        
        for channel in channels:
            provider = self.providers.get(channel)
            if not provider:
                logger.warning(f"No provider for channel {channel.value}")
                continue
            
            try:
                message = provider.format_message(triggered, detailed=True)
                
                channel_config = self.user_channels.get(user_id, {}).get(channel.value, {})
                
                success = await provider.send(user_id, message, **channel_config)
                
                if success:
                    triggered.notification_sent = True
                    triggered.notification_channels_sent.append(channel)
                    self._increment_notification_count(user_id)
                    
            except Exception as e:
                logger.error(f"Failed to send {channel.value} notification: {e}")
    
    def _can_notify_user(self, user_id: str, alert_type: AlertType) -> bool:
        dnd = self.user_dnd.get(user_id)
        if not dnd or not dnd.enabled:
            return True
        
        if dnd.allow_critical and alert_type in (
            AlertType.RUG_WARNING,
            AlertType.SAFETY_ALERT,
            AlertType.HONEYPOT_DETECTED,
            AlertType.POSITION_STOP_LOSS
        ):
            return True
        
        now = datetime.utcnow()
        current_hour = now.hour
        
        if dnd.start_hour > dnd.end_hour:
            if current_hour >= dnd.start_hour or current_hour < dnd.end_hour:
                return False
        else:
            if dnd.start_hour <= current_hour < dnd.end_hour:
                return False
        
        return True
    
    def _check_rate_limit(self, user_id: str) -> bool:
        now = datetime.utcnow()
        counts = self._notification_counts[user_id]
        
        reset_time = self._rate_limit_reset.get(user_id)
        if not reset_time or now > reset_time:
            counts.clear()
            self._rate_limit_reset[user_id] = now + timedelta(hours=1)
        
        if counts.get('hour', 0) >= self.rate_limit.max_per_hour:
            return False
        
        return True
    
    def _increment_notification_count(self, user_id: str):
        self._notification_counts[user_id]['hour'] += 1
        self._notification_counts[user_id]['day'] += 1
    
    def set_user_dnd(self, user_id: str, config: DoNotDisturbConfig):
        self.user_dnd[user_id] = config
    
    def set_user_channel_config(self, user_id: str, channel: NotificationChannel, config: Dict[str, Any]):
        if user_id not in self.user_channels:
            self.user_channels[user_id] = {}
        self.user_channels[user_id][channel.value] = config
    
    def set_user_telegram_chat(self, user_id: str, chat_id: str):
        self.set_user_channel_config(user_id, NotificationChannel.TELEGRAM, {"chat_id": chat_id})
    
    def set_user_webhook(self, user_id: str, webhook_url: str):
        self.set_user_channel_config(user_id, NotificationChannel.WEBHOOK, {"webhook_url": webhook_url})
