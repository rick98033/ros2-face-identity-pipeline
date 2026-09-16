"""Cross-boundary contract declaration types (SI-11.1).

From standards/conventions/cross-boundary-contract-declaration.md —
provides declarative contract types for all cross-boundary interactions.

Every cross-boundary call site must declare timeout, error behavior,
retry semantics, and error codes. BoundaryContract is a frozen dataclass
that codifies this declaration with construction-time validation.

Usage:
    from thor_telemetry import (
        BoundaryContract, BoundaryType, ErrorBehavior, RetryPolicy, ErrorCode,
    )

    LLM_CONTRACT = BoundaryContract(
        boundary_name="orchestrator_vllm",
        boundary_type=BoundaryType.HTTP,
        timeout_sec=5.0,
        error_behavior=ErrorBehavior.RETRY_THEN_FAIL,
        retry_policy=RetryPolicy(max_retries=1, backoff_base_sec=0.15, jitter=True),
        error_codes=frozenset({ErrorCode.TIMEOUT, ErrorCode.UNAVAILABLE}),
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .errors import ErrorCode


class BoundaryType(str, Enum):
    """Taxonomy of cross-boundary interaction types."""
    GPU_INFERENCE = "gpu_inference"
    DATABASE = "database"
    RPC = "rpc"
    FILE_IO = "file_io"
    HTTP = "http"
    WEBSOCKET = "websocket"
    IPC = "ipc"


class ErrorBehavior(str, Enum):
    """What happens when a boundary call fails."""
    FAIL = "fail"
    DEGRADE = "degrade"
    RETRY_THEN_FAIL = "retry_then_fail"
    RETRY_THEN_DEGRADE = "retry_then_degrade"


_RETRY_BEHAVIORS = frozenset({
    ErrorBehavior.RETRY_THEN_FAIL,
    ErrorBehavior.RETRY_THEN_DEGRADE,
})

_NON_RETRY_BEHAVIORS = frozenset({
    ErrorBehavior.FAIL,
    ErrorBehavior.DEGRADE,
})


@dataclass(frozen=True)
class RetryPolicy:
    """Retry semantics for a boundary call.

    Args:
        max_retries: Maximum retry attempts (not counting initial attempt).
            Must be >= 1.
        backoff_base_sec: Base interval for exponential backoff. Must be > 0.
        jitter: Whether to add random jitter to backoff intervals.
    """
    max_retries: int
    backoff_base_sec: float
    jitter: bool = True

    def __post_init__(self) -> None:
        if self.max_retries < 1:
            raise ValueError(
                f"max_retries must be >= 1, got {self.max_retries}"
            )
        if self.backoff_base_sec <= 0:
            raise ValueError(
                f"backoff_base_sec must be > 0, got {self.backoff_base_sec}"
            )


@dataclass(frozen=True)
class BoundaryContract:
    """Declarative contract for a cross-boundary interaction (SI-11.1).

    Immutable after construction. Validated at construction time —
    ValueError raised for invalid declarations.

    Args:
        boundary_name: Identifies the call site (e.g., "orchestrator_vllm").
        boundary_type: Taxonomy category from BoundaryType enum.
        timeout_sec: Maximum time before the call is considered timed out.
        error_behavior: What happens on failure.
        retry_policy: Retry semantics, or None for single-attempt.
        error_codes: ErrorCode values this boundary can produce.
    """
    boundary_name: str
    boundary_type: BoundaryType
    timeout_sec: float
    error_behavior: ErrorBehavior
    retry_policy: Optional[RetryPolicy]
    error_codes: frozenset[ErrorCode]

    def __post_init__(self) -> None:
        if self.timeout_sec <= 0:
            raise ValueError(
                f"timeout_sec must be > 0, got {self.timeout_sec}"
            )
        if not self.error_codes:
            raise ValueError("error_codes must be non-empty")
        if self.error_behavior in _RETRY_BEHAVIORS and self.retry_policy is None:
            raise ValueError(
                f"error_behavior={self.error_behavior.value} requires "
                f"retry_policy to be set"
            )
        if self.error_behavior in _NON_RETRY_BEHAVIORS and self.retry_policy is not None:
            raise ValueError(
                f"error_behavior={self.error_behavior.value} must not have "
                f"retry_policy (no retry on non-retryable behavior)"
            )

    def log_fields(self) -> dict:
        """Declaration fields for structured logging.

        Returns a dict with the contract's static fields, suitable for
        inclusion in boundary call log events alongside runtime fields
        (ok, error_code, timing_ms, trace_id).
        """
        fields: dict = {
            "boundary_name": self.boundary_name,
            "boundary_type": self.boundary_type.value,
            "timeout_sec": self.timeout_sec,
            "error_behavior": self.error_behavior.value,
        }
        if self.retry_policy is not None:
            fields["retry_max"] = self.retry_policy.max_retries
            fields["retry_backoff_sec"] = self.retry_policy.backoff_base_sec
            fields["retry_jitter"] = self.retry_policy.jitter
        return fields
