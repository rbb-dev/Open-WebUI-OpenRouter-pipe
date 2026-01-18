"""Core infrastructure module.

Foundation services required by all domains:
- Configuration schemas (Valves, EncryptedStr)
- Error handling classes
- Error message formatting
- Session logging
- Circuit breaker resilience
- Pure utility functions
"""

from .config import Valves, UserValves, EncryptedStr, LOGGER
from .errors import OpenRouterAPIError, StatusMessages
from .error_formatter import ErrorFormatter
from .logging_system import SessionLogger
from .circuit_breaker import CircuitBreaker
from .utils import (
    _coerce_bool,
    _safe_json_loads,
    _render_error_template,
    _pretty_json,
)

__all__ = [
    "Valves",
    "UserValves",
    "EncryptedStr",
    "LOGGER",
    "OpenRouterAPIError",
    "StatusMessages",
    "ErrorFormatter",
    "SessionLogger",
    "CircuitBreaker",
    "_coerce_bool",
    "_safe_json_loads",
    "_render_error_template",
    "_pretty_json",
]
