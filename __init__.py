
__version__ = "1.0.0"
__author__ = "Trading Bot Team"

from . import exceptions
from . import validators
from . import rate_limiter
from . import retry
from . import audit_secure
from . import transaction
from . import jupiter_async
from . import wallet_async
from . import risk_manager
from . import token_scanner
from . import dca_engine
from . import copy_trading
from . import alerts
from . import analytics

__all__ = [
    "exceptions",
    "validators", 
    "rate_limiter",
    "retry",
    "audit_secure",
    "transaction",
    "jupiter_async",
    "wallet_async",
    "risk_manager",
    "token_scanner",
    "dca_engine",
    "copy_trading",
    "alerts",
    "analytics",
]

def get_version():
    return __version__
