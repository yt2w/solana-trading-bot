
import re
import json
import base58
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Union, Callable, TypeVar
from urllib.parse import urlparse
import unicodedata

from .exceptions import (
    ValidationError,
    InvalidAddressError,
    InvalidAmountError,
    InvalidParameterError,
    SecurityError,
)

T = TypeVar('T')

SOLANA_ADDRESS_LENGTH = 32
SOLANA_SIGNATURE_LENGTH = 64

MIN_SOL_AMOUNT = Decimal("0.000000001")
MAX_SOL_AMOUNT = Decimal("1000000000")
LAMPORTS_PER_SOL = 1_000_000_000
MAX_LAMPORTS = 2**64 - 1

MIN_SLIPPAGE_BPS = 0
MAX_SLIPPAGE_BPS = 10000
DEFAULT_SLIPPAGE_BPS = 100
MIN_PRIORITY_FEE = 0
MAX_PRIORITY_FEE = 10_000_000_000

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128
MAX_WALLET_NAME_LENGTH = 64
MAX_STRING_LENGTH = 10000

DANGEROUS_CHARS = set('<>&"\'\\`${}[]|;')
PATH_TRAVERSAL_PATTERNS = ['..', '~', '%2e%2e', '%252e']

@dataclass
class ValidationResult:
    
    is_valid: bool
    value: Any = None
    error: Optional[str] = None
    field_name: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def success(cls, value: Any, field_name: Optional[str] = None) -> 'ValidationResult':
        return cls(is_valid=True, value=value, field_name=field_name)
    
    @classmethod
    def failure(cls, error: str, field_name: Optional[str] = None,
                details: Optional[Dict[str, Any]] = None) -> 'ValidationResult':
        return cls(is_valid=False, error=error, field_name=field_name, details=details or {})
    
    def raise_if_invalid(self) -> Any:
        if not self.is_valid:
            raise ValidationError(self.error or "Validation failed",
                                  field=self.field_name, details=self.details)
        return self.value

@dataclass
class BatchValidationResult:
    
    results: List[ValidationResult] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        return all(r.is_valid for r in self.results)
    
    @property
    def errors(self) -> List[str]:
        return [r.error for r in self.results if not r.is_valid and r.error]
    
    @property
    def failed_fields(self) -> List[str]:
        return [r.field_name for r in self.results if not r.is_valid and r.field_name]
    
    def add(self, result: ValidationResult) -> 'BatchValidationResult':
        self.results.append(result)
        return self
    
    def raise_if_invalid(self) -> Dict[str, Any]:
        if not self.is_valid:
            raise ValidationError(
                f"Validation failed for fields: {', '.join(self.failed_fields)}",
                details={"errors": self.errors, "fields": self.failed_fields})
        return {r.field_name: r.value for r in self.results if r.field_name}

def validate_solana_address(address: Any, field_name: str = "address") -> str:
    if not isinstance(address, str):
        raise InvalidAddressError(
            f"Address must be a string, got {type(address).__name__}",
            address=str(address)[:50],
            field=field_name
        )
    
    address = address.strip()
    
    if not address:
        raise InvalidAddressError("Address cannot be empty", address="", field=field_name)
    
    if len(address) < 32 or len(address) > 44:
        raise InvalidAddressError(
            f"Invalid address length: {len(address)} characters",
            address=address, field=field_name
        )
    
    try:
        decoded = base58.b58decode(address)
    except Exception as e:
        raise InvalidAddressError(
            f"Invalid base58 encoding: {str(e)}",
            address=address, field=field_name
        )
    
    if len(decoded) != SOLANA_ADDRESS_LENGTH:
        raise InvalidAddressError(
            f"Decoded address has wrong length: {len(decoded)} bytes (expected {SOLANA_ADDRESS_LENGTH})",
            address=address, field=field_name
        )
    
    try:
        from solders.pubkey import Pubkey
        Pubkey.from_string(address)
    except ImportError:
        pass
    except Exception as e:
        raise InvalidAddressError(
            f"Cryptographic validation failed: {str(e)}",
            address=address, field=field_name
        )
    
    return address

def validate_token_mint(address: Any, field_name: str = "token_mint") -> str:
    return validate_solana_address(address, field_name)

def validate_transaction_signature(signature: Any, field_name: str = "signature") -> str:
    if not isinstance(signature, str):
        raise InvalidAddressError(
            f"Signature must be a string, got {type(signature).__name__}",
            address=str(signature)[:50], field=field_name
        )
    
    signature = signature.strip()
    
    if not signature:
        raise InvalidAddressError("Signature cannot be empty", address="", field=field_name)
    
    if len(signature) < 80 or len(signature) > 90:
        raise InvalidAddressError(
            f"Invalid signature length: {len(signature)} characters",
            address=signature, field=field_name
        )
    
    try:
        decoded = base58.b58decode(signature)
    except Exception as e:
        raise InvalidAddressError(
            f"Invalid base58 encoding in signature: {str(e)}",
            address=signature, field=field_name
        )
    
    if len(decoded) != SOLANA_SIGNATURE_LENGTH:
        raise InvalidAddressError(
            f"Decoded signature has wrong length: {len(decoded)} bytes (expected {SOLANA_SIGNATURE_LENGTH})",
            address=signature, field=field_name
        )
    
    return signature

def validate_solana_address_safe(address: Any, field_name: str = "address") -> ValidationResult:
    try:
        validated = validate_solana_address(address, field_name)
        return ValidationResult.success(validated, field_name)
    except InvalidAddressError as e:
        return ValidationResult.failure(str(e), field_name, {"address": str(address)[:50]})

def validate_sol_amount(
    amount: Any,
    field_name: str = "amount",
    min_amount: Optional[Decimal] = None,
    max_amount: Optional[Decimal] = None,
    allow_zero: bool = False
) -> Decimal:
    min_amount = min_amount if min_amount is not None else MIN_SOL_AMOUNT
    max_amount = max_amount if max_amount is not None else MAX_SOL_AMOUNT
    
    try:
        if isinstance(amount, Decimal):
            decimal_amount = amount
        elif isinstance(amount, str):
            cleaned = amount.strip().replace(",", "")
            if not cleaned:
                raise InvalidAmountError("Amount cannot be empty", amount=amount, field=field_name)
            decimal_amount = Decimal(cleaned)
        elif isinstance(amount, (int, float)):
            decimal_amount = Decimal(str(amount))
        else:
            raise InvalidAmountError(
                f"Amount must be a number, got {type(amount).__name__}",
                amount=str(amount), field=field_name
            )
    except InvalidOperation as e:
        raise InvalidAmountError(f"Invalid number format: {str(e)}", amount=str(amount), field=field_name)
    
    if decimal_amount.is_nan():
        raise InvalidAmountError("Amount cannot be NaN", amount=str(amount), field=field_name)
    
    if decimal_amount.is_infinite():
        raise InvalidAmountError("Amount cannot be infinite", amount=str(amount), field=field_name)
    
    if decimal_amount < 0:
        raise InvalidAmountError("Amount cannot be negative", amount=str(decimal_amount),
                                  field=field_name, min_value=0)
    
    if decimal_amount == 0 and not allow_zero:
        raise InvalidAmountError("Amount cannot be zero", amount="0",
                                  field=field_name, min_value=str(min_amount))
    
    if decimal_amount < min_amount and not (allow_zero and decimal_amount == 0):
        raise InvalidAmountError(
            f"Amount {decimal_amount} is below minimum {min_amount}",
            amount=str(decimal_amount), field=field_name, min_value=str(min_amount)
        )
    
    if decimal_amount > max_amount:
        raise InvalidAmountError(
            f"Amount {decimal_amount} exceeds maximum {max_amount}",
            amount=str(decimal_amount), field=field_name, max_value=str(max_amount)
        )
    
    return decimal_amount.quantize(Decimal("0.000000001"), rounding=ROUND_DOWN)

def validate_lamports(
    lamports: Any,
    field_name: str = "lamports",
    min_lamports: int = 0,
    max_lamports: int = MAX_LAMPORTS,
    allow_zero: bool = True
) -> int:
    try:
        if isinstance(lamports, str):
            lamports = lamports.strip().replace(",", "")
            if not lamports:
                raise InvalidAmountError("Lamports cannot be empty", amount="", field=field_name)
            lamports_int = int(lamports)
        elif isinstance(lamports, float):
            if lamports != int(lamports):
                raise InvalidAmountError("Lamports must be a whole number",
                                          amount=str(lamports), field=field_name)
            lamports_int = int(lamports)
        elif isinstance(lamports, int):
            lamports_int = lamports
        else:
            raise InvalidAmountError(
                f"Lamports must be an integer, got {type(lamports).__name__}",
                amount=str(lamports), field=field_name
            )
    except ValueError as e:
        raise InvalidAmountError(f"Invalid lamports format: {str(e)}",
                                  amount=str(lamports), field=field_name)
    
    if lamports_int < 0:
        raise InvalidAmountError("Lamports cannot be negative",
                                  amount=str(lamports_int), field=field_name, min_value=0)
    
    if lamports_int == 0 and not allow_zero:
        raise InvalidAmountError("Lamports cannot be zero", amount="0",
                                  field=field_name, min_value=str(min_lamports))
    
    if lamports_int < min_lamports:
        raise InvalidAmountError(
            f"Lamports {lamports_int} is below minimum {min_lamports}",
            amount=str(lamports_int), field=field_name, min_value=str(min_lamports)
        )
    
    if lamports_int > max_lamports:
        raise InvalidAmountError(
            f"Lamports {lamports_int} exceeds maximum {max_lamports}",
            amount=str(lamports_int), field=field_name, max_value=str(max_lamports)
        )
    
    return lamports_int

def validate_token_amount(
    amount: Any,
    decimals: int,
    field_name: str = "token_amount",
    min_amount: Optional[Decimal] = None,
    max_amount: Optional[Decimal] = None,
    allow_zero: bool = False
) -> Decimal:
    if not isinstance(decimals, int) or decimals < 0 or decimals > 18:
        raise InvalidParameterError(
            f"Token decimals must be between 0 and 18, got {decimals}",
            param_name="decimals", param_value=str(decimals)
        )
    
    if min_amount is None:
        min_amount = Decimal(10) ** (-decimals) if decimals > 0 else Decimal("1")
    if max_amount is None:
        max_amount = Decimal(10) ** 18
    
    try:
        if isinstance(amount, Decimal):
            decimal_amount = amount
        elif isinstance(amount, str):
            cleaned = amount.strip().replace(",", "")
            if not cleaned:
                raise InvalidAmountError("Token amount cannot be empty", amount="", field=field_name)
            decimal_amount = Decimal(cleaned)
        elif isinstance(amount, (int, float)):
            decimal_amount = Decimal(str(amount))
        else:
            raise InvalidAmountError(
                f"Token amount must be a number, got {type(amount).__name__}",
                amount=str(amount), field=field_name
            )
    except InvalidOperation as e:
        raise InvalidAmountError(f"Invalid token amount format: {str(e)}",
                                  amount=str(amount), field=field_name)
    
    if decimal_amount.is_nan() or decimal_amount.is_infinite():
        raise InvalidAmountError("Token amount must be a finite number",
                                  amount=str(amount), field=field_name)
    
    if decimal_amount < 0:
        raise InvalidAmountError("Token amount cannot be negative",
                                  amount=str(decimal_amount), field=field_name)
    
    if decimal_amount == 0 and not allow_zero:
        raise InvalidAmountError("Token amount cannot be zero", amount="0", field=field_name)
    
    if decimal_amount < min_amount and not (allow_zero and decimal_amount == 0):
        raise InvalidAmountError(
            f"Token amount {decimal_amount} is below minimum {min_amount}",
            amount=str(decimal_amount), field=field_name, min_value=str(min_amount)
        )
    
    if decimal_amount > max_amount:
        raise InvalidAmountError(
            f"Token amount {decimal_amount} exceeds maximum {max_amount}",
            amount=str(decimal_amount), field=field_name, max_value=str(max_amount)
        )
    
    quantizer = Decimal(10) ** (-decimals)
    return decimal_amount.quantize(quantizer, rounding=ROUND_DOWN)

def sol_to_lamports(sol_amount: Decimal) -> int:
    return int(sol_amount * LAMPORTS_PER_SOL)

def lamports_to_sol(lamports: int) -> Decimal:
    return Decimal(lamports) / LAMPORTS_PER_SOL

def validate_slippage(
    slippage_bps: Any,
    field_name: str = "slippage_bps",
    min_bps: int = MIN_SLIPPAGE_BPS,
    max_bps: int = MAX_SLIPPAGE_BPS
) -> int:
    try:
        if isinstance(slippage_bps, str):
            slippage_bps = slippage_bps.strip()
            if slippage_bps.endswith('%'):
                slippage_bps = float(slippage_bps[:-1]) * 100
            slippage_int = int(float(slippage_bps))
        elif isinstance(slippage_bps, float):
            slippage_int = int(slippage_bps)
        elif isinstance(slippage_bps, int):
            slippage_int = slippage_bps
        else:
            raise InvalidParameterError(
                f"Slippage must be a number, got {type(slippage_bps).__name__}",
                param_name=field_name, param_value=str(slippage_bps)
            )
    except ValueError as e:
        raise InvalidParameterError(f"Invalid slippage format: {str(e)}",
                                     param_name=field_name, param_value=str(slippage_bps))
    
    if slippage_int < min_bps:
        raise InvalidParameterError(
            f"Slippage {slippage_int} bps is below minimum {min_bps} bps",
            param_name=field_name, param_value=str(slippage_int), min_value=min_bps
        )
    
    if slippage_int > max_bps:
        raise InvalidParameterError(
            f"Slippage {slippage_int} bps exceeds maximum {max_bps} bps ({max_bps/100}%)",
            param_name=field_name, param_value=str(slippage_int), max_value=max_bps
        )
    
    return slippage_int

def validate_priority_fee(
    fee_lamports: Any,
    field_name: str = "priority_fee",
    min_fee: int = MIN_PRIORITY_FEE,
    max_fee: int = MAX_PRIORITY_FEE
) -> int:
    try:
        validated = validate_lamports(
            fee_lamports, field_name=field_name,
            min_lamports=min_fee, max_lamports=max_fee, allow_zero=True
        )
    except InvalidAmountError as e:
        raise InvalidParameterError(str(e), param_name=field_name, param_value=str(fee_lamports))
    
    return validated

def validate_trade_params(params: Any, field_name: str = "trade_params") -> Dict[str, Any]:
    if not isinstance(params, dict):
        raise ValidationError(
            f"Trade parameters must be a dictionary, got {type(params).__name__}",
            field=field_name
        )
    
    validated = {}
    batch = BatchValidationResult()
    
    if "token_mint" in params:
        try:
            validated["token_mint"] = validate_token_mint(params["token_mint"], "token_mint")
            batch.add(ValidationResult.success(validated["token_mint"], "token_mint"))
        except InvalidAddressError as e:
            batch.add(ValidationResult.failure(str(e), "token_mint"))
    else:
        batch.add(ValidationResult.failure("token_mint is required", "token_mint"))
    
    if "amount" in params:
        try:
            validated["amount"] = validate_sol_amount(params["amount"], "amount")
            batch.add(ValidationResult.success(validated["amount"], "amount"))
        except InvalidAmountError as e:
            batch.add(ValidationResult.failure(str(e), "amount"))
    else:
        batch.add(ValidationResult.failure("amount is required", "amount"))
    
    if "slippage_bps" in params:
        try:
            validated["slippage_bps"] = validate_slippage(params["slippage_bps"], "slippage_bps")
            batch.add(ValidationResult.success(validated["slippage_bps"], "slippage_bps"))
        except InvalidParameterError as e:
            batch.add(ValidationResult.failure(str(e), "slippage_bps"))
    else:
        validated["slippage_bps"] = DEFAULT_SLIPPAGE_BPS
        batch.add(ValidationResult.success(DEFAULT_SLIPPAGE_BPS, "slippage_bps"))
    
    if "priority_fee" in params:
        try:
            validated["priority_fee"] = validate_priority_fee(params["priority_fee"], "priority_fee")
            batch.add(ValidationResult.success(validated["priority_fee"], "priority_fee"))
        except InvalidParameterError as e:
            batch.add(ValidationResult.failure(str(e), "priority_fee"))
    else:
        validated["priority_fee"] = 0
        batch.add(ValidationResult.success(0, "priority_fee"))
    
    if not batch.is_valid:
        raise ValidationError(
            f"Trade parameter validation failed: {', '.join(batch.errors)}",
            field=field_name,
            details={"errors": batch.errors, "fields": batch.failed_fields}
        )
    
    return validated

def validate_telegram_id(telegram_id: Any, field_name: str = "telegram_id") -> int:
    try:
        if isinstance(telegram_id, str):
            telegram_id = telegram_id.strip()
            if not telegram_id:
                raise ValidationError("Telegram ID cannot be empty", field=field_name)
            id_int = int(telegram_id)
        elif isinstance(telegram_id, int):
            id_int = telegram_id
        elif isinstance(telegram_id, float):
            if telegram_id != int(telegram_id):
                raise ValidationError("Telegram ID must be a whole number", field=field_name)
            id_int = int(telegram_id)
        else:
            raise ValidationError(
                f"Telegram ID must be a number, got {type(telegram_id).__name__}",
                field=field_name
            )
    except ValueError as e:
        raise ValidationError(f"Invalid Telegram ID format: {str(e)}", field=field_name)
    
    if id_int <= 0:
        raise ValidationError("Telegram ID must be a positive integer", field=field_name)
    
    if id_int > 2**63:
        raise ValidationError("Telegram ID exceeds maximum value", field=field_name)
    
    return id_int

def validate_wallet_name(
    name: Any,
    field_name: str = "wallet_name",
    max_length: int = MAX_WALLET_NAME_LENGTH
) -> str:
    if not isinstance(name, str):
        raise SecurityError(
            f"Wallet name must be a string, got {type(name).__name__}",
            security_context=field_name
        )
    
    name = name.strip()
    if not name:
        raise SecurityError("Wallet name cannot be empty", security_context=field_name)
    
    if len(name) > max_length:
        raise SecurityError(
            f"Wallet name exceeds maximum length of {max_length} characters",
            security_context=field_name
        )
    
    name_lower = name.lower()
    for pattern in PATH_TRAVERSAL_PATTERNS:
        if pattern in name_lower:
            raise SecurityError(
                f"Wallet name contains forbidden pattern: {pattern}",
                security_context=field_name,
                details={"forbidden_pattern": pattern}
            )
    
    if '/' in name or '\\' in name:
        raise SecurityError(
            "Wallet name cannot contain path separators",
            security_context=field_name
        )
    
    if not re.match(r'^[a-zA-Z0-9_\- ]+$', name):
        raise SecurityError(
            "Wallet name can only contain letters, numbers, spaces, hyphens, and underscores",
            security_context=field_name
        )
    
    return name

def validate_password_strength(
    password: Any,
    field_name: str = "password",
    min_length: int = MIN_PASSWORD_LENGTH,
    max_length: int = MAX_PASSWORD_LENGTH,
    require_uppercase: bool = True,
    require_lowercase: bool = True,
    require_digit: bool = True,
    require_special: bool = False
) -> str:
    if not isinstance(password, str):
        raise SecurityError(
            f"Password must be a string, got {type(password).__name__}",
            security_context=field_name
        )
    
    if len(password) < min_length:
        raise SecurityError(
            f"Password must be at least {min_length} characters long",
            security_context=field_name,
            details={"min_length": min_length, "actual_length": len(password)}
        )
    
    if len(password) > max_length:
        raise SecurityError(
            f"Password cannot exceed {max_length} characters",
            security_context=field_name,
            details={"max_length": max_length}
        )
    
    errors = []
    
    if require_uppercase and not re.search(r'[A-Z]', password):
        errors.append("at least one uppercase letter")
    
    if require_lowercase and not re.search(r'[a-z]', password):
        errors.append("at least one lowercase letter")
    
    if require_digit and not re.search(r'\d', password):
        errors.append("at least one digit")
    
    if require_special and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        errors.append("at least one special character")
    
    if errors:
        raise SecurityError(
            f"Password must contain {', '.join(errors)}",
            security_context=field_name,
            details={"missing_requirements": errors}
        )
    
    return password

def sanitize_string(
    s: Any,
    field_name: str = "string",
    max_length: int = MAX_STRING_LENGTH,
    allow_newlines: bool = True,
    remove_control_chars: bool = True
) -> str:
    if not isinstance(s, str):
        raise ValidationError(f"Expected string, got {type(s).__name__}", field=field_name)
    
    if len(s) > max_length:
        raise ValidationError(f"String exceeds maximum length of {max_length}", field=field_name)
    
    s = unicodedata.normalize('NFC', s)
    
    if remove_control_chars:
        allowed_control = set()
        if allow_newlines:
            allowed_control.update({'\n', '\r', '\t'})
        
        result = []
        for char in s:
            if unicodedata.category(char) == 'Cc':
                if char in allowed_control:
                    result.append(char)
            else:
                result.append(char)
        s = ''.join(result)
    
    s = ''.join(c for c in s if c not in DANGEROUS_CHARS)
    
    return s

def validate_json_safe(
    data: Any,
    field_name: str = "data",
    max_depth: int = 10,
    max_size: int = 1_000_000
) -> Any:
    def check_depth(obj: Any, current_depth: int = 0) -> None:
        if current_depth > max_depth:
            raise ValidationError(
                f"Data exceeds maximum nesting depth of {max_depth}",
                field=field_name
            )
        
        if isinstance(obj, dict):
            for value in obj.values():
                check_depth(value, current_depth + 1)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                check_depth(item, current_depth + 1)
    
    check_depth(data)
    
    try:
        serialized = json.dumps(data, default=str)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValidationError(f"Data is not JSON serializable: {str(e)}", field=field_name)
    
    if len(serialized) > max_size:
        raise ValidationError(
            f"Serialized data exceeds maximum size of {max_size} bytes",
            field=field_name
        )
    
    return data

def validate_url(
    url: Any,
    field_name: str = "url",
    allowed_schemes: Optional[List[str]] = None,
    require_tls: bool = False
) -> str:
    if allowed_schemes is None:
        allowed_schemes = ['http', 'https', 'ws', 'wss']
    
    if not isinstance(url, str):
        raise ValidationError(f"URL must be a string, got {type(url).__name__}", field=field_name)
    
    url = url.strip()
    if not url:
        raise ValidationError("URL cannot be empty", field=field_name)
    
    try:
        parsed = urlparse(url)
    except Exception as e:
        raise ValidationError(f"Invalid URL format: {str(e)}", field=field_name)
    
    if not parsed.scheme:
        raise ValidationError(
            "URL must include a scheme (http, https, ws, wss)",
            field=field_name
        )
    
    if parsed.scheme.lower() not in allowed_schemes:
        raise ValidationError(
            f"URL scheme must be one of: {', '.join(allowed_schemes)}",
            field=field_name
        )
    
    if require_tls and parsed.scheme.lower() not in ['https', 'wss']:
        raise ValidationError("URL must use TLS (https or wss)", field=field_name)
    
    if not parsed.netloc:
        raise ValidationError("URL must include a host", field=field_name)
    
    dangerous_in_url = ['<', '>', '"', "'", '{', '}', '|', '\\', '^', '`']
    for char in dangerous_in_url:
        if char in url:
            raise ValidationError(f"URL contains forbidden character: {char}", field=field_name)
    
    return url

def validate_batch(
    validations: List[tuple],
    raise_on_first_error: bool = False
) -> BatchValidationResult:
    batch = BatchValidationResult()
    
    for validator, value, field_name in validations:
        try:
            validated_value = validator(value)
            batch.add(ValidationResult.success(validated_value, field_name))
        except Exception as e:
            result = ValidationResult.failure(str(e), field_name)
            batch.add(result)
            
            if raise_on_first_error:
                result.raise_if_invalid()
    
    return batch

def create_validator(
    validate_func: Callable[[Any], T],
    error_class: type = ValidationError
) -> Callable[[Any, str], T]:
    def validator(value: Any, field_name: str = "value") -> T:
        try:
            return validate_func(value)
        except error_class:
            raise
        except Exception as e:
            raise error_class(f"Validation failed: {str(e)}", field=field_name)
    
    return validator

def is_valid_solana_address(address: Any) -> bool:
    try:
        validate_solana_address(address)
        return True
    except Exception:
        return False

def is_valid_sol_amount(amount: Any) -> bool:
    try:
        validate_sol_amount(amount)
        return True
    except Exception:
        return False

def is_valid_url(url: Any) -> bool:
    try:
        validate_url(url)
        return True
    except Exception:
        return False

def is_valid_telegram_id(telegram_id: Any) -> bool:
    try:
        validate_telegram_id(telegram_id)
        return True
    except Exception:
        return False

def is_valid_wallet_name(name: Any) -> bool:
    try:
        validate_wallet_name(name)
        return True
    except Exception:
        return False

__all__ = [
    'ValidationResult',
    'BatchValidationResult',
    'validate_solana_address',
    'validate_token_mint',
    'validate_transaction_signature',
    'validate_solana_address_safe',
    'validate_sol_amount',
    'validate_lamports',
    'validate_token_amount',
    'sol_to_lamports',
    'lamports_to_sol',
    'validate_slippage',
    'validate_priority_fee',
    'validate_trade_params',
    'validate_telegram_id',
    'validate_wallet_name',
    'validate_password_strength',
    'sanitize_string',
    'validate_json_safe',
    'validate_url',
    'validate_batch',
    'create_validator',
    'is_valid_solana_address',
    'is_valid_sol_amount',
    'is_valid_url',
    'is_valid_telegram_id',
    'is_valid_wallet_name',
    'LAMPORTS_PER_SOL',
    'MIN_SOL_AMOUNT',
    'MAX_SOL_AMOUNT',
    'DEFAULT_SLIPPAGE_BPS',
]
