"""Core infrastructure module.

Foundation services required by all domains:
- Configuration schemas (Valves, EncryptedStr)
- Error handling classes
- Error message formatting
- Session logging
- Circuit breaker resilience
- Pure utility functions
"""

from .circuit_breaker import CircuitBreaker
from .config import EncryptedStr, UserValves, Valves
from .error_formatter import ErrorFormatter
from .errors import OpenRouterAPIError, StatusMessages
from .logging_system import SessionLogger
from .utils import (
    _coerce_bool,
    _pretty_json,
    _render_error_template,
    _safe_json_loads,
)

__all__ = [
    "CircuitBreaker",
    "EncryptedStr",
    "ErrorFormatter",
    "OpenRouterAPIError",
    "SessionLogger",
    "StatusMessages",
    "UserValves",
    "Valves",
    "_coerce_bool",
    "_pretty_json",
    "_render_error_template",
    "_safe_json_loads",
]
