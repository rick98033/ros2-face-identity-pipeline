"""Canonical error codes for boundary failures.

From docs/discussions/08_boundary_contracts.md - provides uniform error
semantics across all system boundaries (MCP, Runtime Daemon, ROS Gateway).

Usage:
    from thor_telemetry import ErrorCode, is_retryable, normalize_error_code

    code = ErrorCode.TIMEOUT
    if is_retryable(code):
        # retry logic

    # Normalize unknown string codes
    code = normalize_error_code("SOME_UNKNOWN_CODE", logger=my_logger)
    # Returns ErrorCode.INTERNAL, logs warning
"""

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .logging import StructuredLogger


class ErrorCode(str, Enum):
    """Canonical error codes for all boundary failures.

    Retry decisions are made ONLY on the retryable flag derived from these codes,
    not on the code value itself. Use is_retryable() to check.

    Retryable codes (transient failures):
        UNAVAILABLE - Dependency not reachable / down
        TIMEOUT - Dependency did not respond in time
        OVERLOADED - Dependency explicitly overloaded (e.g., 429, 503)

    Non-retryable codes (permanent failures):
        AUTH - Authentication or authorization failure
        BAD_REQUEST - Invalid input / schema violation
        NOT_FOUND - Referenced entity does not exist
        CONFLICT - State conflict / double execution
        PARSE_ERROR - Malformed response from dependency
        INTERNAL - Unexpected internal error (catch-all)
    """

    # Retryable
    UNAVAILABLE = "UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    OVERLOADED = "OVERLOADED"

    # Non-retryable
    AUTH = "AUTH"
    BAD_REQUEST = "BAD_REQUEST"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    PARSE_ERROR = "PARSE_ERROR"
    INTERNAL = "INTERNAL"


RETRYABLE_CODES: frozenset[ErrorCode] = frozenset(
    {
        ErrorCode.UNAVAILABLE,
        ErrorCode.TIMEOUT,
        ErrorCode.OVERLOADED,
    }
)


def is_retryable(code: "ErrorCode | str") -> bool:
    """Check if an error code is retryable.

    Args:
        code: ErrorCode enum or string value

    Returns:
        True if the code represents a transient failure that may succeed on retry.
        Returns False for unknown string codes.
    """
    if isinstance(code, str):
        try:
            code = ErrorCode(code)
        except ValueError:
            return False
    return code in RETRYABLE_CODES


def normalize_error_code(
    code: str, logger: "StructuredLogger | None" = None
) -> ErrorCode:
    """Normalize a string to ErrorCode, logging unknown codes.

    Args:
        code: String error code to normalize
        logger: Optional StructuredLogger for warning on unknown codes

    Returns:
        The corresponding ErrorCode, or ErrorCode.INTERNAL for unknown codes.
    """
    try:
        return ErrorCode(code)
    except ValueError:
        if logger:
            logger.warning("unknown_error_code", code=code, normalized_to="INTERNAL")
        return ErrorCode.INTERNAL
