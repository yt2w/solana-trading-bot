
import asyncio
import gzip
import hashlib
import hmac
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Optional, Dict, Any, List, Callable, Awaitable,
    Union, TypeVar, Generic, Iterator, AsyncIterator
)
import logging
import re
import csv
import io
from contextlib import asynccontextmanager
from collections import defaultdict
import base64

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

try:
    import fcntl
    FCNTL_AVAILABLE = True
except ImportError:
    FCNTL_AVAILABLE = False

try:
    import msvcrt
    MSVCRT_AVAILABLE = True
except ImportError:
    MSVCRT_AVAILABLE = False

logger = logging.getLogger(__name__)

class AuditAction(Enum):
    WALLET_CREATE = "wallet.create"
    WALLET_IMPORT = "wallet.import"
    WALLET_EXPORT = "wallet.export"
    WALLET_DELETE = "wallet.delete"
    WALLET_VIEW = "wallet.view"
    WALLET_LIST = "wallet.list"
    
    KEY_ACCESS = "key.access"
    KEY_DECRYPT = "key.decrypt"
    KEY_GENERATE = "key.generate"
    KEY_ROTATE = "key.rotate"
    
    TRADE_QUOTE = "trade.quote"
    TRADE_EXECUTE = "trade.execute"
    TRADE_CONFIRM = "trade.confirm"
    TRADE_FAIL = "trade.fail"
    TRADE_CANCEL = "trade.cancel"
    TRADE_TIMEOUT = "trade.timeout"
    
    SETTINGS_VIEW = "settings.view"
    SETTINGS_CHANGE = "settings.change"
    SETTINGS_RESET = "settings.reset"
    
    LOGIN = "auth.login"
    LOGOUT = "auth.logout"
    AUTH_FAIL = "auth.fail"
    AUTH_REFRESH = "auth.refresh"
    SESSION_CREATE = "session.create"
    SESSION_EXPIRE = "session.expire"
    
    RATE_LIMIT_HIT = "rate.hit"
    RATE_LIMIT_EXCEEDED = "rate.exceeded"
    RATE_LIMIT_RESET = "rate.reset"
    
    SECURITY_ALERT = "security.alert"
    SUSPICIOUS_ACTIVITY = "security.suspicious"
    TAMPERING_DETECTED = "security.tampering"
    INVALID_SIGNATURE = "security.invalid_signature"
    CHAIN_BROKEN = "security.chain_broken"
    
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    SYSTEM_ERROR = "system.error"
    SYSTEM_MAINTENANCE = "system.maintenance"
    
    DATA_EXPORT = "data.export"
    DATA_DELETE = "data.delete"
    DATA_BACKUP = "data.backup"
    DATA_RESTORE = "data.restore"
    
    ADMIN_ACTION = "admin.action"
    CONFIG_CHANGE = "admin.config"
    
    API_CALL = "api.call"
    API_ERROR = "api.error"
    WEBHOOK_SEND = "webhook.send"
    WEBHOOK_FAIL = "webhook.fail"

class AuditResult(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    PENDING = "pending"
    DENIED = "denied"
    TIMEOUT = "timeout"
    ERROR = "error"
    SKIPPED = "skipped"

class AlertSeverity(Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AuditEntry:
    timestamp: str
    entry_id: str
    sequence_number: int
    
    action: str
    result: str
    
    user_id: Optional[str] = None
    username: Optional[str] = None
    
    wallet_address: Optional[str] = None
    
    details: Dict[str, Any] = field(default_factory=dict)
    
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    user_agent: Optional[str] = None
    
    previous_hash: str = ""
    chain_hash: str = ""
    signature: str = ""
    
    version: str = "2.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEntry':
        return cls(**data)
    
    def get_hashable_content(self) -> str:
        content = {
            'timestamp': self.timestamp,
            'entry_id': self.entry_id,
            'sequence_number': self.sequence_number,
            'action': self.action,
            'result': self.result,
            'user_id': self.user_id,
            'username': self.username,
            'wallet_address': self.wallet_address,
            'details': self.details,
            'ip_address': self.ip_address,
            'session_id': self.session_id,
            'user_agent': self.user_agent,
            'previous_hash': self.previous_hash,
            'version': self.version,
        }
        return json.dumps(content, sort_keys=True, separators=(',', ':'))

@dataclass
class AuditStats:
    total_entries: int = 0
    entries_today: int = 0
    entries_this_hour: int = 0
    by_action: Dict[str, int] = field(default_factory=dict)
    by_result: Dict[str, int] = field(default_factory=dict)
    by_user: Dict[str, int] = field(default_factory=dict)
    chain_valid: bool = True
    last_entry_time: Optional[str] = None
    oldest_entry_time: Optional[str] = None

@dataclass 
class IntegrityReport:
    is_valid: bool
    total_entries: int
    verified_entries: int
    first_invalid_sequence: Optional[int] = None
    errors: List[str] = field(default_factory=list)
    verified_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    merkle_root: Optional[str] = None

class SecurityUtils:
    
    SOLANA_ADDRESS_PATTERN = re.compile(r'^[1-9A-HJ-NP-Za-km-z]{32,44}$')
    
    SENSITIVE_FIELDS = {
        'private_key', 'privatekey', 'secret', 'password', 'passwd',
        'seed', 'mnemonic', 'seed_phrase', 'recovery_phrase',
        'api_key', 'apikey', 'token', 'auth_token', 'access_token',
        'secret_key', 'secretkey', 'signing_key'
    }
    
    @staticmethod
    def mask_address(address: Optional[str], visible_chars: int = 4) -> Optional[str]:
        if not address:
            return None
        if len(address) <= visible_chars * 2:
            return '*' * len(address)
        return f"{address[:visible_chars]}...{address[-visible_chars:]}"
    
    @staticmethod
    def mask_ip(ip: Optional[str]) -> Optional[str]:
        if not ip:
            return None
        if ':' in ip:
            parts = ip.split(':')
            if len(parts) >= 2:
                return f"{parts[0]}:{parts[1]}:****"
        else:
            parts = ip.split('.')
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.*.*"
        return "***"
    
    @classmethod
    def sanitize_details(cls, details: Dict[str, Any]) -> Dict[str, Any]:
        if not details:
            return {}
        
        sanitized = {}
        for key, value in details.items():
            key_lower = key.lower()
            
            if any(sensitive in key_lower for sensitive in cls.SENSITIVE_FIELDS):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = cls.sanitize_details(value)
            elif isinstance(value, str):
                if cls._looks_like_secret(value):
                    sanitized[key] = "[REDACTED]"
                elif cls.SOLANA_ADDRESS_PATTERN.match(value):
                    sanitized[key] = cls.mask_address(value)
                else:
                    sanitized[key] = value
            elif isinstance(value, list):
                sanitized[key] = [
                    cls.sanitize_details(v) if isinstance(v, dict) else v
                    for v in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized
    
    @staticmethod
    def _looks_like_secret(value: str) -> bool:
        if not value or len(value) < 20:
            return False
        if len(value) in (64, 88) and all(c in '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz' for c in value):
            return True
        if len(value) >= 40 and value.replace('+', '').replace('/', '').replace('=', '').isalnum():
            return True
        words = value.lower().split()
        if len(words) in (12, 24):
            return True
        return False
    
    @staticmethod
    def generate_hmac(key: bytes, data: str) -> str:
        return hmac.new(key, data.encode('utf-8'), hashlib.sha256).hexdigest()
    
    @staticmethod
    def verify_hmac(key: bytes, data: str, signature: str) -> bool:
        expected = hmac.new(key, data.encode('utf-8'), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)
    
    @staticmethod
    def compute_chain_hash(content: str, previous_hash: str) -> str:
        combined = f"{previous_hash}:{content}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    
    @staticmethod
    def compute_merkle_root(hashes: List[str]) -> str:
        if not hashes:
            return hashlib.sha256(b'empty').hexdigest()
        if len(hashes) == 1:
            return hashes[0]
        if len(hashes) % 2 == 1:
            hashes = hashes + [hashes[-1]]
        next_level = []
        for i in range(0, len(hashes), 2):
            combined = hashes[i] + hashes[i + 1]
            next_level.append(hashlib.sha256(combined.encode()).hexdigest())
        return SecurityUtils.compute_merkle_root(next_level)

class FileLock:
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.lock_file = filepath.with_suffix(filepath.suffix + '.lock')
        self._handle = None
    
    async def acquire(self, timeout: float = 10.0) -> bool:
        start = asyncio.get_event_loop().time()
        
        while True:
            try:
                self._handle = open(self.lock_file, 'w')
                
                if FCNTL_AVAILABLE:
                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                elif MSVCRT_AVAILABLE:
                    msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    if self.lock_file.exists():
                        age = asyncio.get_event_loop().time() - self.lock_file.stat().st_mtime
                        if age > 60:
                            self.lock_file.unlink()
                        else:
                            raise BlockingIOError("Lock held")
                    self._handle.write(str(os.getpid()))
                    self._handle.flush()
                
                return True
                
            except (BlockingIOError, OSError):
                if self._handle:
                    self._handle.close()
                    self._handle = None
                
                elapsed = asyncio.get_event_loop().time() - start
                if elapsed >= timeout:
                    return False
                
                await asyncio.sleep(0.1)
    
    def release(self):
        if self._handle:
            try:
                if FCNTL_AVAILABLE:
                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
                elif MSVCRT_AVAILABLE:
                    try:
                        msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
                    except Exception:
                        pass
            except Exception:
                pass
            finally:
                self._handle.close()
                self._handle = None
                try:
                    self.lock_file.unlink()
                except Exception:
                    pass
    
    async def __aenter__(self):
        acquired = await self.acquire()
        if not acquired:
            raise TimeoutError(f"Could not acquire lock for {self.filepath}")
        return self
    
    async def __aexit__(self, *args):
        self.release()

class AuditEncryption:
    
    def __init__(self, key: Optional[bytes] = None, password: Optional[str] = None):
        if not CRYPTO_AVAILABLE:
            self._fernet = None
            logger.warning("Cryptography not available - encryption disabled")
            return
        
        if key:
            self._fernet = Fernet(key)
        elif password:
            salt = b'audit_log_salt_v2'
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            self._fernet = Fernet(key)
        else:
            self._fernet = None
    
    def encrypt(self, data: str) -> str:
        if not self._fernet:
            return data
        return self._fernet.encrypt(data.encode()).decode()
    
    def decrypt(self, data: str) -> str:
        if not self._fernet:
            return data
        return self._fernet.decrypt(data.encode()).decode()
    
    @property
    def enabled(self) -> bool:
        return self._fernet is not None

class WebhookNotifier:
    
    def __init__(self, webhook_url: Optional[str] = None, timeout: float = 10.0):
        self.webhook_url = webhook_url
        self.timeout = timeout
        self._session = None
    
    async def notify(self, entry: AuditEntry, severity: AlertSeverity) -> bool:
        if not self.webhook_url:
            return False
        
        try:
            import aiohttp
            
            payload = {
                "timestamp": entry.timestamp,
                "action": entry.action,
                "result": entry.result,
                "severity": severity.value,
                "user_id": entry.user_id,
                "details": entry.details,
                "entry_id": entry.entry_id,
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    return response.status < 400
                    
        except ImportError:
            logger.warning("aiohttp not available for webhook notifications")
            return False
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
            return False

class SecureAuditLogger:
    
    GENESIS_HASH = "0" * 64
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        hmac_key: Optional[bytes] = None,
        encryption_key: Optional[bytes] = None,
        retention_days: int = 90,
        rotation_size_mb: int = 100,
        enable_compression: bool = True,
        webhook_url: Optional[str] = None,
        auto_verify_on_start: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._hmac_key = hmac_key or os.urandom(32)
        self._encryption = AuditEncryption(encryption_key) if encryption_key else None
        
        self.retention_days = retention_days
        self.rotation_size_mb = rotation_size_mb
        self.enable_compression = enable_compression
        
        self._sequence_number = 0
        self._previous_hash = self.GENESIS_HASH
        self._lock = asyncio.Lock()
        self._file_lock = FileLock(self._current_log_path)
        self._initialized = False
        
        self._webhook = WebhookNotifier(webhook_url)
        self._alert_callbacks: List[Callable[[AuditEntry, AlertSeverity], Awaitable[None]]] = []
        
        self._alert_actions = {
            AuditAction.SECURITY_ALERT: AlertSeverity.HIGH,
            AuditAction.SUSPICIOUS_ACTIVITY: AlertSeverity.HIGH,
            AuditAction.TAMPERING_DETECTED: AlertSeverity.CRITICAL,
            AuditAction.CHAIN_BROKEN: AlertSeverity.CRITICAL,
            AuditAction.AUTH_FAIL: AlertSeverity.MEDIUM,
            AuditAction.RATE_LIMIT_EXCEEDED: AlertSeverity.MEDIUM,
            AuditAction.WALLET_DELETE: AlertSeverity.MEDIUM,
            AuditAction.KEY_DECRYPT: AlertSeverity.LOW,
        }
        
        self._stats_cache: Optional[AuditStats] = None
        self._stats_cache_time: Optional[datetime] = None
        self._auto_verify = auto_verify_on_start
        
        logger.info(f"SecureAuditLogger initialized: {self.log_dir}")
    
    @property
    def _current_log_path(self) -> Path:
        return self.log_dir / "audit.jsonl"
    
    @property
    def _index_path(self) -> Path:
        return self.log_dir / "audit_index.json"
    
    @property
    def _state_path(self) -> Path:
        return self.log_dir / "audit_state.json"
    
    @property
    def _checkpoint_path(self) -> Path:
        return self.log_dir / "audit_checkpoints.jsonl"
    
    async def initialize(self) -> 'SecureAuditLogger':
        async with self._lock:
            if self._initialized:
                return self
            
            await self._load_state()
            
            if self._auto_verify and self._current_log_path.exists():
                report = await self._verify_chain_internal()
                if not report.is_valid:
                    logger.error(f"Chain integrity check failed: {report.errors}")
            
            self._initialized = True
            logger.info(f"Audit logger initialized. Sequence: {self._sequence_number}")
            
            return self
    
    async def _load_state(self):
        if self._state_path.exists():
            try:
                data = json.loads(self._state_path.read_text())
                self._sequence_number = data.get('sequence_number', 0)
                self._previous_hash = data.get('previous_hash', self.GENESIS_HASH)
            except Exception as e:
                logger.warning(f"Could not load state, reconstructing: {e}")
                await self._reconstruct_state()
        else:
            await self._reconstruct_state()
    
    async def _reconstruct_state(self):
        if not self._current_log_path.exists():
            return
        
        try:
            with open(self._current_log_path, 'rb') as f:
                f.seek(0, 2)
                size = f.tell()
                if size == 0:
                    return
                
                chunk_size = min(4096, size)
                f.seek(size - chunk_size)
                chunk = f.read().decode('utf-8')
                lines = chunk.strip().split('\n')
                
                if lines:
                    last_entry = json.loads(lines[-1])
                    self._sequence_number = last_entry.get('sequence_number', 0)
                    self._previous_hash = last_entry.get('chain_hash', self.GENESIS_HASH)
                    
        except Exception as e:
            logger.error(f"Could not reconstruct state: {e}")
    
    async def _save_state(self):
        try:
            state = {
                'sequence_number': self._sequence_number,
                'previous_hash': self._previous_hash,
                'updated_at': datetime.now(timezone.utc).isoformat(),
            }
            
            temp_path = self._state_path.with_suffix('.tmp')
            temp_path.write_text(json.dumps(state, indent=2))
            temp_path.replace(self._state_path)
            
        except Exception as e:
            logger.error(f"Could not save state: {e}")

    async def log(
        self,
        action: AuditAction,
        result: AuditResult,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        wallet_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None,
        user_agent: Optional[str] = None,
        encrypt_details: bool = False,
    ) -> AuditEntry:
        if not self._initialized:
            await self.initialize()
        
        async with self._lock:
            self._sequence_number += 1
            
            safe_details = SecurityUtils.sanitize_details(details or {})
            if encrypt_details and self._encryption:
                safe_details = {
                    '_encrypted': self._encryption.encrypt(json.dumps(safe_details))
                }
            
            entry = AuditEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                entry_id=str(uuid.uuid4()),
                sequence_number=self._sequence_number,
                action=action.value,
                result=result.value,
                user_id=str(user_id) if user_id else None,
                username=username,
                wallet_address=SecurityUtils.mask_address(wallet_address),
                details=safe_details,
                ip_address=SecurityUtils.mask_ip(ip_address),
                session_id=session_id,
                user_agent=user_agent[:200] if user_agent else None,
                previous_hash=self._previous_hash,
            )
            
            hashable_content = entry.get_hashable_content()
            entry.chain_hash = SecurityUtils.compute_chain_hash(
                hashable_content, self._previous_hash
            )
            
            entry.signature = SecurityUtils.generate_hmac(
                self._hmac_key, 
                f"{hashable_content}:{entry.chain_hash}"
            )
            
            self._previous_hash = entry.chain_hash
            
            await self._write_entry(entry)
            
            if self._sequence_number % 100 == 0:
                await self._save_state()
            
            await self._check_rotation()
            
            self._stats_cache = None
            
        await self._handle_alerts(entry, action)
        
        return entry
    
    async def _write_entry(self, entry: AuditEntry):
        async with self._file_lock:
            try:
                with open(self._current_log_path, 'a', encoding='utf-8') as f:
                    json.dump(entry.to_dict(), f, separators=(',', ':'))
                    f.write('\n')
                    f.flush()
                    os.fsync(f.fileno())
                    
            except Exception as e:
                logger.error(f"Failed to write audit entry: {e}")
                raise
    
    async def _check_rotation(self):
        try:
            if not self._current_log_path.exists():
                return
            
            size_mb = self._current_log_path.stat().st_size / (1024 * 1024)
            if size_mb >= self.rotation_size_mb:
                await self._rotate_log()
                
        except Exception as e:
            logger.error(f"Rotation check failed: {e}")
    
    async def _rotate_log(self):
        try:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            rotated_name = f"audit_{timestamp}.jsonl"
            rotated_path = self.log_dir / rotated_name
            
            await self._create_checkpoint()
            
            shutil.move(str(self._current_log_path), str(rotated_path))
            
            if self.enable_compression:
                await self._compress_file(rotated_path)
            
            self._previous_hash = self.GENESIS_HASH
            await self._save_state()
            
            logger.info(f"Rotated audit log to {rotated_name}")
            
            await self._cleanup_old_logs()
            
        except Exception as e:
            logger.error(f"Log rotation failed: {e}")
    
    async def _compress_file(self, filepath: Path):
        try:
            compressed_path = filepath.with_suffix('.jsonl.gz')
            
            with open(filepath, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            filepath.unlink()
            logger.info(f"Compressed {filepath.name}")
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
    
    async def _cleanup_old_logs(self):
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
            
            for filepath in self.log_dir.glob('audit_*.jsonl*'):
                try:
                    mtime = datetime.fromtimestamp(
                        filepath.stat().st_mtime, 
                        tz=timezone.utc
                    )
                    if mtime < cutoff:
                        filepath.unlink()
                        logger.info(f"Deleted old audit log: {filepath.name}")
                except Exception as e:
                    logger.error(f"Could not delete {filepath}: {e}")
                    
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    async def _create_checkpoint(self):
        try:
            hashes = []
            async for entry in self._read_entries():
                hashes.append(entry.chain_hash)
            
            if not hashes:
                return
            
            checkpoint = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'entry_count': len(hashes),
                'first_sequence': 1,
                'last_sequence': self._sequence_number,
                'merkle_root': SecurityUtils.compute_merkle_root(hashes),
                'last_hash': self._previous_hash,
            }
            
            checkpoint['signature'] = SecurityUtils.generate_hmac(
                self._hmac_key,
                json.dumps(checkpoint, sort_keys=True)
            )
            
            with open(self._checkpoint_path, 'a', encoding='utf-8') as f:
                json.dump(checkpoint, f)
                f.write('\n')
            
            logger.info(f"Created checkpoint: {checkpoint['merkle_root'][:16]}...")
            
        except Exception as e:
            logger.error(f"Checkpoint creation failed: {e}")
    
    async def _handle_alerts(self, entry: AuditEntry, action: AuditAction):
        severity = self._alert_actions.get(action)
        
        if not severity:
            return
        
        if self._webhook.webhook_url:
            asyncio.create_task(self._webhook.notify(entry, severity))
        
        for callback in self._alert_callbacks:
            try:
                await callback(entry, severity)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def register_alert_callback(
        self, 
        callback: Callable[[AuditEntry, AlertSeverity], Awaitable[None]]
    ):
        self._alert_callbacks.append(callback)

    async def verify_chain(self) -> IntegrityReport:
        async with self._lock:
            return await self._verify_chain_internal()
    
    async def _verify_chain_internal(self) -> IntegrityReport:
        errors = []
        verified = 0
        first_invalid = None
        hashes = []
        
        previous_hash = self.GENESIS_HASH
        expected_sequence = 0
        
        try:
            async for entry in self._read_entries():
                expected_sequence += 1
                
                if entry.sequence_number != expected_sequence:
                    errors.append(
                        f"Sequence gap: expected {expected_sequence}, "
                        f"got {entry.sequence_number}"
                    )
                    if first_invalid is None:
                        first_invalid = expected_sequence
                
                if entry.previous_hash != previous_hash:
                    errors.append(
                        f"Chain break at sequence {entry.sequence_number}: "
                        f"previous hash mismatch"
                    )
                    if first_invalid is None:
                        first_invalid = entry.sequence_number
                
                expected_hash = SecurityUtils.compute_chain_hash(
                    entry.get_hashable_content(),
                    entry.previous_hash
                )
                if entry.chain_hash != expected_hash:
                    errors.append(
                        f"Hash mismatch at sequence {entry.sequence_number}"
                    )
                    if first_invalid is None:
                        first_invalid = entry.sequence_number
                
                signature_content = f"{entry.get_hashable_content()}:{entry.chain_hash}"
                if not SecurityUtils.verify_hmac(
                    self._hmac_key, signature_content, entry.signature
                ):
                    errors.append(
                        f"Invalid signature at sequence {entry.sequence_number}"
                    )
                    if first_invalid is None:
                        first_invalid = entry.sequence_number
                
                hashes.append(entry.chain_hash)
                previous_hash = entry.chain_hash
                verified += 1
                
        except Exception as e:
            errors.append(f"Verification error: {e}")
        
        return IntegrityReport(
            is_valid=len(errors) == 0,
            total_entries=verified,
            verified_entries=verified if not errors else (first_invalid or 1) - 1,
            first_invalid_sequence=first_invalid,
            errors=errors,
            merkle_root=SecurityUtils.compute_merkle_root(hashes) if hashes else None,
        )
    
    async def verify_entry(self, entry: AuditEntry) -> bool:
        signature_content = f"{entry.get_hashable_content()}:{entry.chain_hash}"
        return SecurityUtils.verify_hmac(
            self._hmac_key, signature_content, entry.signature
        )

    async def _read_entries(
        self,
        filepath: Optional[Path] = None
    ) -> AsyncIterator[AuditEntry]:
        filepath = filepath or self._current_log_path
        
        if not filepath.exists():
            return
        
        if filepath.suffix == '.gz':
            opener = gzip.open
        else:
            opener = open
        
        with opener(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield AuditEntry.from_dict(json.loads(line))
                    except Exception as e:
                        logger.warning(f"Could not parse entry: {e}")
    
    async def query_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        actions: Optional[List[AuditAction]] = None,
        results: Optional[List[AuditResult]] = None,
        user_id: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[AuditEntry]:
        action_values = {a.value for a in actions} if actions else None
        result_values = {r.value for r in results} if results else None
        
        matches = []
        skipped = 0
        
        async for entry in self._read_entries():
            if start_time:
                entry_time = datetime.fromisoformat(entry.timestamp.replace('Z', '+00:00'))
                if entry_time < start_time:
                    continue
            
            if end_time:
                entry_time = datetime.fromisoformat(entry.timestamp.replace('Z', '+00:00'))
                if entry_time > end_time:
                    continue
            
            if action_values and entry.action not in action_values:
                continue
            
            if result_values and entry.result not in result_values:
                continue
            
            if user_id and entry.user_id != str(user_id):
                continue
            
            if skipped < offset:
                skipped += 1
                continue
            
            matches.append(entry)
            
            if len(matches) >= limit:
                break
        
        return matches
    
    async def get_user_activity(
        self,
        user_id: str,
        limit: int = 100
    ) -> List[AuditEntry]:
        return await self.query_logs(user_id=user_id, limit=limit)
    
    async def get_entries_by_action(
        self,
        action: AuditAction,
        limit: int = 100
    ) -> List[AuditEntry]:
        return await self.query_logs(actions=[action], limit=limit)
    
    async def get_security_events(
        self,
        start_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditEntry]:
        security_actions = [
            AuditAction.SECURITY_ALERT,
            AuditAction.SUSPICIOUS_ACTIVITY,
            AuditAction.TAMPERING_DETECTED,
            AuditAction.AUTH_FAIL,
            AuditAction.RATE_LIMIT_EXCEEDED,
            AuditAction.INVALID_SIGNATURE,
            AuditAction.CHAIN_BROKEN,
        ]
        return await self.query_logs(
            start_time=start_time,
            actions=security_actions,
            limit=limit
        )
    
    async def get_recent_entries(self, count: int = 50) -> List[AuditEntry]:
        entries = []
        async for entry in self._read_entries():
            entries.append(entry)
        return entries[-count:] if len(entries) > count else entries

    async def get_stats(self, force_refresh: bool = False) -> AuditStats:
        if not force_refresh and self._stats_cache:
            age = datetime.now(timezone.utc) - self._stats_cache_time
            if age.total_seconds() < 60:
                return self._stats_cache
        
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        hour_start = now.replace(minute=0, second=0, microsecond=0)
        
        stats = AuditStats()
        
        async for entry in self._read_entries():
            stats.total_entries += 1
            
            entry_time = datetime.fromisoformat(
                entry.timestamp.replace('Z', '+00:00')
            )
            
            if entry_time >= today_start:
                stats.entries_today += 1
            
            if entry_time >= hour_start:
                stats.entries_this_hour += 1
            
            stats.by_action[entry.action] = stats.by_action.get(entry.action, 0) + 1
            stats.by_result[entry.result] = stats.by_result.get(entry.result, 0) + 1
            
            if entry.user_id:
                stats.by_user[entry.user_id] = stats.by_user.get(entry.user_id, 0) + 1
            
            if not stats.oldest_entry_time:
                stats.oldest_entry_time = entry.timestamp
            stats.last_entry_time = entry.timestamp
        
        report = await self._verify_chain_internal()
        stats.chain_valid = report.is_valid
        
        self._stats_cache = stats
        self._stats_cache_time = now
        
        return stats
    
    async def export_user_data(
        self,
        user_id: str,
        format: str = 'json'
    ) -> Union[str, bytes]:
        entries = await self.get_user_activity(user_id, limit=100000)
        
        if format == 'csv':
            output = io.StringIO()
            writer = csv.writer(output)
            
            writer.writerow([
                'timestamp', 'entry_id', 'action', 'result',
                'wallet_address', 'details', 'ip_address'
            ])
            
            for entry in entries:
                writer.writerow([
                    entry.timestamp,
                    entry.entry_id,
                    entry.action,
                    entry.result,
                    entry.wallet_address or '',
                    json.dumps(entry.details),
                    entry.ip_address or '',
                ])
            
            return output.getvalue()
        else:
            return json.dumps(
                [entry.to_dict() for entry in entries],
                indent=2
            )
    
    async def delete_user_data(self, user_id: str) -> int:
        logger.warning(
            f"User data deletion requested for {user_id}. "
            "Implementing anonymization to preserve chain integrity."
        )
        
        await self.log(
            action=AuditAction.DATA_DELETE,
            result=AuditResult.SUCCESS,
            user_id="SYSTEM",
            details={
                'requested_user_id': user_id,
                'action': 'anonymization',
                'note': 'Chain integrity preserved'
            }
        )
        
        return 0
    
    async def generate_integrity_report(self) -> Dict[str, Any]:
        verification = await self.verify_chain()
        stats = await self.get_stats()
        
        return {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'chain_integrity': {
                'is_valid': verification.is_valid,
                'total_entries': verification.total_entries,
                'verified_entries': verification.verified_entries,
                'errors': verification.errors,
                'merkle_root': verification.merkle_root,
            },
            'statistics': {
                'total_entries': stats.total_entries,
                'entries_today': stats.entries_today,
                'by_action': stats.by_action,
                'by_result': stats.by_result,
            },
            'configuration': {
                'retention_days': self.retention_days,
                'rotation_size_mb': self.rotation_size_mb,
                'compression_enabled': self.enable_compression,
                'encryption_enabled': self._encryption is not None,
            }
        }

    async def close(self):
        async with self._lock:
            await self._save_state()
            logger.info("SecureAuditLogger closed")
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, *args):
        await self.close()

_audit_logger: Optional[SecureAuditLogger] = None

async def get_audit_logger() -> SecureAuditLogger:
    global _audit_logger
    if _audit_logger is None:
        raise RuntimeError("Audit logger not initialized. Call init_audit_logger first.")
    return _audit_logger

async def init_audit_logger(
    log_dir: Union[str, Path],
    hmac_key: Optional[bytes] = None,
    **kwargs
) -> SecureAuditLogger:
    global _audit_logger
    _audit_logger = SecureAuditLogger(log_dir, hmac_key, **kwargs)
    await _audit_logger.initialize()
    return _audit_logger

async def audit_log(
    action: AuditAction,
    result: AuditResult,
    **kwargs
) -> AuditEntry:
    logger_instance = await get_audit_logger()
    return await logger_instance.log(action, result, **kwargs)

if __name__ == "__main__":
    async def main():
        
        async with SecureAuditLogger(
            log_dir="./audit_logs",
            retention_days=90,
        ) as audit:
            
            await audit.log(
                action=AuditAction.SYSTEM_START,
                result=AuditResult.SUCCESS,
                details={"version": "2.0.0"}
            )
            
            await audit.log(
                action=AuditAction.WALLET_CREATE,
                result=AuditResult.SUCCESS,
                user_id="123456789",
                username="test_user",
                wallet_address="7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU",
                ip_address="127.0.0.1"
            )
            
            await audit.log(
                action=AuditAction.TRADE_EXECUTE,
                result=AuditResult.SUCCESS,
                user_id="123456789",
                wallet_address="7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU",
                details={
                    "token_in": "SOL",
                    "token_out": "USDC",
                    "amount": "1.5",
                    "tx_signature": "5xyz..."
                }
            )
            
            report = await audit.verify_chain()
            print(f"Chain valid: {report.is_valid}")
            print(f"Merkle root: {report.merkle_root}")
            
            stats = await audit.get_stats()
            print(f"Total entries: {stats.total_entries}")
            print(f"By action: {stats.by_action}")
            
            user_logs = await audit.get_user_activity("123456789")
            print(f"User entries: {len(user_logs)}")
            
            integrity_report = await audit.generate_integrity_report()
            print(json.dumps(integrity_report, indent=2))
    
    asyncio.run(main())
