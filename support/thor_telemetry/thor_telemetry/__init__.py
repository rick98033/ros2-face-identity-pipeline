"""Thor telemetry - shared trace context and structured logging.

This package provides canonical identifiers for end-to-end request correlation
and structured JSONL logging for all Thor components.

Trace Context:
    from thor_telemetry import TraceContext, set_trace_context, get_trace_context

    ctx = TraceContext(session_id="sess_xxx", turn_id="turn_xxx", trace_id="turn_xxx")
    token = set_trace_context(ctx)
    # ... do work ...
    reset_trace_context(token)

Structured Logging:
    from thor_telemetry import get_logger, TraceContext

    logger = get_logger("orchestrator")
    logger.info("event_name", trace_ctx=ctx, key="value")
"""

from .trace import (
    TraceContext,
    set_trace_context,
    get_trace_context,
    clear_trace_context,
    reset_trace_context,
    trace_context_or_none,
    generate_session_id,
    generate_turn_id,
    inject_trace_to_header,
    extract_trace_from_header,
    trace_headers,
    extract_trace_from_http_headers,
    missing_trace_fields,
    trace_log_fields,
    is_valid_turn_id,
    is_valid_session_id,
)

from .logging import (
    get_logger,
    StructuredLogger,
    LOG_SCHEMA_VERSION,
)

from .errors import (
    ErrorCode,
    RETRYABLE_CODES,
    is_retryable,
    normalize_error_code,
)

from .freshness import (
    FreshValue,
    StaleStateError,
)

from .retention import (
    RetentionDeclaration,
)

from .boundary import (
    BoundaryType,
    ErrorBehavior,
    RetryPolicy,
    BoundaryContract,
)

from .reconciliation import (
    ReconciliationTracker,
    ReconciliationState,
    ReconciliationResult,
    ReconciliationHealth,
)

__all__ = [
    # Trace context
    "TraceContext",
    "set_trace_context",
    "get_trace_context",
    "clear_trace_context",
    "reset_trace_context",
    "trace_context_or_none",
    "generate_session_id",
    "generate_turn_id",
    "inject_trace_to_header",
    "extract_trace_from_header",
    "trace_headers",
    "extract_trace_from_http_headers",
    "missing_trace_fields",
    "trace_log_fields",
    "is_valid_turn_id",
    "is_valid_session_id",
    # Structured logging
    "get_logger",
    "StructuredLogger",
    "LOG_SCHEMA_VERSION",
    # Error codes
    "ErrorCode",
    "RETRYABLE_CODES",
    "is_retryable",
    "normalize_error_code",
    # Freshness-bounded state
    "FreshValue",
    "StaleStateError",
    # Retention declarations
    "RetentionDeclaration",
    # Boundary contracts
    "BoundaryType",
    "ErrorBehavior",
    "RetryPolicy",
    "BoundaryContract",
    # Cross-boundary reconciliation (CP-005)
    "ReconciliationTracker",
    "ReconciliationState",
    "ReconciliationResult",
    "ReconciliationHealth",
]
