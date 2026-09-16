"""Structured JSONL logging with rotation for Thor pipeline.

This module provides a shared logging infrastructure that writes structured
JSON lines to disk with automatic rotation and thread-safe writes.

Usage:
    from thor_telemetry import get_logger

    logger = get_logger("orchestrator")
    logger.info("llm_request_start",
        trace_ctx=trace_ctx,
        model="nemotron",
        context_items=16
    )

Log files are written to ~/.thor/logs/ with the following layout:
    - pipeline.jsonl (orchestrator, mic_asr, tts)
    - mcp.jsonl (mcp_client, mcp_server)
    - tools.jsonl (tool_runner)
    - safety.jsonl (dialogue_monitor)
    - misc.jsonl (unknown components)
    - daemons/runtime_daemon.jsonl
    - daemons/ros_gateway.jsonl

Features:
    - Automatic base fields: log_schema_version, ts_wall, level, component, event, pid, process
    - Safe JSON serialization (never crashes on bad input)
    - File rotation at 50MB with 10 backups
    - Thread-safe writes within a process
    - Trace context injection when provided

Note: Does not coordinate across processes. Line interleaving is possible
when multiple processes write to the same stream. Use pid/process fields
to filter/join reliably.
"""

import json
import os
import threading
import time
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .errors import ErrorCode
    from .trace import TraceContext

# Log schema version for future-proofing
LOG_SCHEMA_VERSION = 1

# Rotation settings
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_BACKUP_COUNT = 10

# Base log directory
LOG_DIR = Path(os.path.expanduser("~/.thor/logs"))

# Component to stream mapping
COMPONENT_STREAMS = {
    "mic_asr_node": "pipeline",
    "orchestrator": "pipeline",
    "tts_node": "pipeline",
    "mcp_client": "mcp",
    "mcp_server": "mcp",
    "tool_runner": "tools",
    "runtime_daemon": "daemons/runtime_daemon",
    "ros_gateway": "daemons/ros_gateway",
    "dialogue_monitor": "safety",
    "person_tracker": "tracking",
    "iou_tracker": "tracking",
    "state_manager": "pipeline",
    "time_event_store": "pipeline",
    "memory_processor": "pipeline",
    "face_detection_node": "perception",
    "person_detector_node": "perception",
    "auraface_id_node": "perception",
    "perception_state_server": "perception",
    "face_state_server": "perception",
    "mjpeg_server": "perception",
    "peoplenet_trt": "perception",
    "enrollment_store": "perception",
    "alignment": "perception",
}
DEFAULT_STREAM = "misc"

# Global registry of loggers (one per stream)
_loggers: dict[str, "StructuredLogger"] = {}
_loggers_lock = threading.Lock()


def _convert_known_types(value: Any) -> tuple[Any, bool]:
    """Convert known non-serializable types. Returns (converted, was_converted)."""
    if isinstance(value, Enum):
        return value.value, True
    if isinstance(value, Exception):
        return {"type": type(value).__name__, "msg": str(value)[:200]}, True
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>", True
    if hasattr(value, "__dataclass_fields__"):
        try:
            return asdict(value), True
        except Exception:
            return str(value)[:200], True
    return value, False


def _safe_dumps(record: dict) -> str:
    """Serialize record to JSON. Never raises."""
    # First pass: convert known problematic types
    converted_record = {}
    for k, v in record.items():
        converted, _ = _convert_known_types(v)
        converted_record[k] = converted

    # Try optimistic serialization
    try:
        return json.dumps(converted_record)
    except (TypeError, ValueError):
        # Fallback: stringify problematic fields one by one
        errors = []
        safe_record = {}
        for k, v in converted_record.items():
            try:
                json.dumps({k: v})
                safe_record[k] = v
            except (TypeError, ValueError):
                safe_record[k] = str(v)[:200]
                errors.append(f"field {k} not serializable")
        if errors:
            safe_record["_logging_errors"] = errors
        return json.dumps(safe_record)


class StructuredLogger:
    """Thread-safe JSONL logger with rotation.

    Each logger instance is associated with a stream (log file) and a component.
    Multiple components may share the same stream.
    """

    def __init__(self, component: str, stream: str):
        """Initialize logger for a component.

        Args:
            component: Component name (e.g., "orchestrator")
            stream: Stream name (e.g., "pipeline") - determines file path
        """
        self._component = component
        self._stream = stream
        self._process = component  # Use component as process name
        self._pid = os.getpid()

        # Determine file path
        self._log_path = LOG_DIR / f"{stream}.jsonl"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread safety
        self._lock = threading.Lock()
        self._file: Optional[Any] = None

    def _get_file(self):
        """Get file handle, opening if needed."""
        if self._file is None or self._file.closed:
            self._file = open(self._log_path, "a", buffering=1)  # Line buffered
        return self._file

    def _rotate_if_needed(self):
        """Rotate log file if it exceeds size limit."""
        try:
            if not self._log_path.exists():
                return

            size = self._log_path.stat().st_size
            if size < MAX_FILE_SIZE_BYTES:
                return

            # Close current file
            if self._file and not self._file.closed:
                self._file.close()
                self._file = None

            # Delete oldest backup if at limit
            oldest = self._log_path.with_suffix(f".jsonl.{MAX_BACKUP_COUNT}")
            if oldest.exists():
                oldest.unlink()

            # Shift existing backups (9 -> 10, 8 -> 9, etc.)
            for i in range(MAX_BACKUP_COUNT - 1, 0, -1):
                src = self._log_path.with_suffix(f".jsonl.{i}")
                dst = self._log_path.with_suffix(f".jsonl.{i + 1}")
                if src.exists():
                    src.rename(dst)

            # Rotate current file to .1
            backup = self._log_path.with_suffix(".jsonl.1")
            self._log_path.rename(backup)

        except Exception:
            # Never fail on rotation errors - just skip rotation
            pass

    def _write_record(self, record: dict):
        """Write a record to the log file."""
        with self._lock:
            self._rotate_if_needed()
            try:
                f = self._get_file()
                line = _safe_dumps(record)
                f.write(line + "\n")
                f.flush()
            except Exception:
                # Never crash on logging errors
                pass

    def _log(
        self,
        level: str,
        event: str,
        trace_ctx: Optional["TraceContext"] = None,
        msg: Optional[str] = None,
        **kwargs,
    ):
        """Internal logging method."""
        record = {
            "log_schema_version": LOG_SCHEMA_VERSION,
            "ts_wall": time.time(),
            "level": level,
            "component": self._component,
            "event": event,
            "pid": self._pid,
            "process": self._process,
        }

        # Add trace context if provided
        if trace_ctx is not None:
            record["session_id"] = trace_ctx.session_id
            record["turn_id"] = trace_ctx.turn_id
            record["trace_id"] = trace_ctx.trace_id

        # Add optional message
        if msg is not None:
            record["msg"] = msg

        # Add extra fields, filtering out None values
        for k, v in kwargs.items():
            if v is not None:
                record[k] = v

        self._write_record(record)

    def debug(
        self,
        event: str,
        trace_ctx: Optional["TraceContext"] = None,
        msg: Optional[str] = None,
        **kwargs,
    ):
        """Log a DEBUG level event."""
        self._log("DEBUG", event, trace_ctx, msg, **kwargs)

    def info(
        self,
        event: str,
        trace_ctx: Optional["TraceContext"] = None,
        msg: Optional[str] = None,
        **kwargs,
    ):
        """Log an INFO level event."""
        self._log("INFO", event, trace_ctx, msg, **kwargs)

    def warning(
        self,
        event: str,
        trace_ctx: Optional["TraceContext"] = None,
        msg: Optional[str] = None,
        **kwargs,
    ):
        """Log a WARNING level event."""
        self._log("WARNING", event, trace_ctx, msg, **kwargs)

    def error(
        self,
        event: str,
        trace_ctx: Optional["TraceContext"] = None,
        msg: Optional[str] = None,
        **kwargs,
    ):
        """Log an ERROR level event."""
        self._log("ERROR", event, trace_ctx, msg, **kwargs)

    def emit_failure(
        self,
        operation: str,
        error_code: "ErrorCode",
        trace_ctx: Optional["TraceContext"] = None,
        *,
        error_detail: Optional[str] = None,
        trigger: Optional[str] = None,
        prior_state: Optional[str] = None,
        new_state: Optional[str] = None,
        **kwargs,
    ):
        """Emit a structured failure event (SI-2.3, SI-12.2).

        Mandatory fields (component, ts_wall, level, pid, process) are
        auto-populated.  ``operation`` and ``error_code`` are required
        positional arguments so callers cannot omit them.

        Args:
            operation: Operation that failed (e.g., "safety_assessment").
            error_code: Canonical ErrorCode from thor_telemetry.errors.
            trace_ctx: Optional TraceContext for session/trace correlation.
            error_detail: Human-readable detail (truncated to 200 chars, no PII).
            trigger: What caused the fault (e.g., "timeout", "exception").
            prior_state: State before the fault transition.
            new_state: State after the fault transition.
            **kwargs: Additional structured fields.
        """
        extra: dict[str, Any] = {
            "operation": operation,
            "error_code": error_code,
        }
        if error_detail is not None:
            extra["error_detail"] = error_detail[:200]
        if trigger is not None:
            extra["trigger"] = trigger
        if prior_state is not None:
            extra["prior_state"] = prior_state
        if new_state is not None:
            extra["new_state"] = new_state
        extra.update(kwargs)
        self._log("ERROR", "failure", trace_ctx, **extra)

    def decision_record(
        self,
        invariant: str,
        decision: str,
        inputs: Optional[dict] = None,
        trace_ctx: Optional["TraceContext"] = None,
        **kwargs,
    ):
        """Emit a structured decision record (SI-12.3, SI-12.4, SI-13.2).

        Captures the invariant being served, the input evidence, and the
        outcome for safety-critical decisions.  Distinct from ``info()``
        (events) and ``emit_failure()`` (faults).

        Args:
            invariant: The SI-NNN or principle being served (e.g., "SI-12.3").
            decision: The outcome chosen (e.g., "escalate", "authorized").
            inputs: Evidence that produced the decision (no PII). Must be
                JSON-serializable.  None is treated as empty dict.
            trace_ctx: Optional TraceContext for session/trace correlation.
            **kwargs: Additional structured fields.
        """
        extra: dict[str, Any] = {
            "invariant": invariant,
            "decision": decision,
            "inputs": inputs if inputs is not None else {},
        }
        extra.update(kwargs)
        self._log("INFO", "decision_record", trace_ctx, **extra)

    def close(self):
        """Close the log file."""
        with self._lock:
            if self._file and not self._file.closed:
                self._file.close()
                self._file = None


def get_logger(component: str) -> StructuredLogger:
    """Get or create a structured logger for a component.

    Args:
        component: Component name (e.g., "orchestrator", "mic_asr_node")

    Returns:
        StructuredLogger instance for the component's stream
    """
    stream = COMPONENT_STREAMS.get(component, DEFAULT_STREAM)

    with _loggers_lock:
        # Use component as key so each component gets its own logger
        # but they may share the same underlying stream/file
        key = f"{component}:{stream}"
        if key not in _loggers:
            _loggers[key] = StructuredLogger(component, stream)
        return _loggers[key]
