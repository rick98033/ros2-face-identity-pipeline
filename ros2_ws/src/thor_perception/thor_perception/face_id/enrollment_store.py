"""Face enrollment storage and matching.

This module provides persistent storage for enrolled face embeddings and
cosine similarity matching for face identification.

Storage format:
    ~/.thor/identity/face_enrollment.json

    {
        "_meta": {
            "embedding_dim": 512,
            "model": "fal/AuraFace-v1",
            "created_at": "2025-12-20T..."
        },
        "user_alice": {
            "embeddings": [[...], [...], ...],
            "enrolled_at": "2025-12-20T...",
            "sample_count": 15,
            "model_version": "fal/AuraFace-v1"
        }
    }

Matching strategy:
    Uses max similarity across all enrolled samples per user, which is more
    robust than centroid matching for handling domain gap (different lighting,
    angles, etc.).
"""

import json
import logging
import time as _time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

from thor_telemetry import get_logger as _get_structured_logger, ErrorCode
from thor_telemetry import BoundaryContract, BoundaryType, ErrorBehavior
from thor_telemetry import RetentionDeclaration
_structured_logger = _get_structured_logger("enrollment_store")

# --- CP-001 Retention Declarations (SI-12.4, SI-13.2, AD-046) ---
RETENTION_DECLARATIONS = (
    RetentionDeclaration(
        data_category="face_embeddings",
        owner="enrollment_store",
        retention_period="indefinite_until_explicit_delete",
        access_control="component-internal",
        deletion_mechanism="explicit_api",
        legal_basis="user_consent_at_enrollment",
    ),
    RetentionDeclaration(
        data_category="enrollment_metadata",
        owner="enrollment_store",
        retention_period="indefinite_until_explicit_delete",
        access_control="component-internal",
        deletion_mechanism="explicit_api",
    ),
)

# Default enrollment file path
ENROLLMENT_PATH = Path.home() / ".thor" / "identity" / "face_enrollment.json"

# Expected embedding dimension for AuraFace (fal/AuraFace-v1 outputs 212-dim)
EXPECTED_EMBEDDING_DIM = 212

# Maximum samples per user to cap storage and matching cost
MAX_SAMPLES_PER_USER = 50

# --- CP-010 Boundary Contracts (SI-11.1) ---
_file_io_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="enrollment_io")

ENROLLMENT_LOAD_CONTRACT = BoundaryContract(
    boundary_name="enrollment_store_load",
    boundary_type=BoundaryType.FILE_IO,
    timeout_sec=5.0,
    error_behavior=ErrorBehavior.DEGRADE,
    retry_policy=None,
    error_codes=frozenset({
        ErrorCode.TIMEOUT, ErrorCode.UNAVAILABLE, ErrorCode.PARSE_ERROR,
    }),
)

ENROLLMENT_SAVE_CONTRACT = BoundaryContract(
    boundary_name="enrollment_store_save",
    boundary_type=BoundaryType.FILE_IO,
    timeout_sec=5.0,
    error_behavior=ErrorBehavior.FAIL,
    retry_policy=None,
    error_codes=frozenset({
        ErrorCode.TIMEOUT, ErrorCode.UNAVAILABLE, ErrorCode.INTERNAL,
    }),
)


@dataclass
class EnrolledUser:
    """Enrolled user with face embeddings."""

    user_id: str
    centroid: np.ndarray        # L2-normalized mean of embeddings (for fallback)
    embeddings: list[np.ndarray]  # Individual L2-normalized embeddings
    sample_count: int
    enrolled_at: str
    model_version: str


class FaceEnrollmentStore:
    """Face enrollment storage and matching.

    Loads enrolled face embeddings from JSON storage and provides
    cosine similarity matching against incoming embeddings.

    Attributes:
        path: Path to enrollment JSON file
        users: Dict mapping user_id to EnrolledUser
    """

    def __init__(self, path: Path = ENROLLMENT_PATH):
        """Initialize enrollment store.

        Args:
            path: Path to enrollment JSON file. Created on first enrollment
                  if it doesn't exist.
        """
        self.path = path
        self.users: dict[str, EnrolledUser] = {}
        self._load()

    def _load(self):
        """Load enrollment from JSON with validation."""
        if not self.path.exists():
            logger.info(f"No enrollment file at {self.path}")
            return

        t0 = _time.monotonic()
        try:
            future = _file_io_pool.submit(self._load_file_sync)
            data = future.result(timeout=ENROLLMENT_LOAD_CONTRACT.timeout_sec)
        except TimeoutError:
            timing_ms = round((_time.monotonic() - t0) * 1000, 1)
            logger.error("Enrollment load timed out")
            _structured_logger.emit_failure(
                operation=ENROLLMENT_LOAD_CONTRACT.boundary_name,
                error_code=ErrorCode.TIMEOUT,
                error_detail=f"File read timed out after {ENROLLMENT_LOAD_CONTRACT.timeout_sec}s",
                trigger="TimeoutError",
                timing_ms=timing_ms,
                **{k: v for k, v in ENROLLMENT_LOAD_CONTRACT.log_fields().items()
                   if k != "boundary_name"},
            )
            return
        except Exception as e:
            timing_ms = round((_time.monotonic() - t0) * 1000, 1)
            error_code = ErrorCode.PARSE_ERROR if isinstance(e, json.JSONDecodeError) else ErrorCode.UNAVAILABLE
            logger.error(f"Failed to load enrollment: {e}")
            _structured_logger.emit_failure(
                operation=ENROLLMENT_LOAD_CONTRACT.boundary_name,
                error_code=error_code,
                error_detail=str(e)[:200],
                trigger=type(e).__name__,
                timing_ms=timing_ms,
                **{k: v for k, v in ENROLLMENT_LOAD_CONTRACT.log_fields().items()
                   if k != "boundary_name"},
            )
            return

        if data is None:
            return

        loaded_count = 0
        skipped_count = 0

        for user_id, info in data.items():
            # Skip metadata entries
            if user_id.startswith("_"):
                continue

            # Skip entries without embeddings
            if "embeddings" not in info or not info["embeddings"]:
                logger.warning(f"Skipping {user_id}: no embeddings")
                skipped_count += 1
                continue

            embeddings = [np.array(e, dtype=np.float32) for e in info["embeddings"]]

            # Validate embedding dimension
            if embeddings[0].shape[0] != EXPECTED_EMBEDDING_DIM:
                logger.warning(
                    f"Skipping {user_id}: wrong dim {embeddings[0].shape[0]} "
                    f"(expected {EXPECTED_EMBEDDING_DIM})"
                )
                skipped_count += 1
                continue

            # Cap samples per user
            if len(embeddings) > MAX_SAMPLES_PER_USER:
                logger.info(
                    f"Capping {user_id} from {len(embeddings)} to {MAX_SAMPLES_PER_USER} samples"
                )
                embeddings = embeddings[:MAX_SAMPLES_PER_USER]

            # Compute centroid (for fallback/debug)
            centroid = np.mean(embeddings, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

            self.users[user_id] = EnrolledUser(
                user_id=user_id,
                centroid=centroid,
                embeddings=embeddings,
                sample_count=len(embeddings),
                enrolled_at=info.get("enrolled_at", ""),
                model_version=info.get("model_version", ""),
            )
            loaded_count += 1

        logger.info(f"Loaded {loaded_count} users, skipped {skipped_count}")

    def _load_file_sync(self):
        """Read and parse enrollment JSON (runs in executor thread)."""
        with open(self.path) as f:
            return json.load(f)

    def match(
        self,
        embedding: np.ndarray,
        top_k: int = 3
    ) -> list[tuple[str, float]]:
        """Match embedding against enrolled users using max-to-samples.

        Uses max similarity across all samples per user, which is more robust
        than centroid-only matching for domain gap cases (different lighting,
        angles, etc.).

        Args:
            embedding: L2-normalized query embedding, shape (512,)
            top_k: Number of top matches to return

        Returns:
            List of (user_id, score) tuples, sorted by score descending.
            Returns [("UNKNOWN", 0.0)] if no enrolled users or all skipped.
        """
        if not self.users:
            return [("UNKNOWN", 0.0)]

        if embedding is None:
            return [("UNKNOWN", 0.0)]

        scores = []
        for user_id, user in self.users.items():
            # Skip users with no embeddings (shouldn't happen, but guard)
            if not user.embeddings:
                continue

            # Max similarity across all enrolled samples
            max_score = max(
                float(np.dot(embedding, e))
                for e in user.embeddings
            )
            scores.append((user_id, max_score))

        if not scores:
            return [("UNKNOWN", 0.0)]

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def save(self):
        """Save enrollment to JSON."""
        # Ensure directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "_meta": {
                "embedding_dim": EXPECTED_EMBEDDING_DIM,
                "model": "fal/AuraFace-v1",
                "created_at": datetime.now().isoformat(),
            }
        }

        for user_id, user in self.users.items():
            data[user_id] = {
                "embeddings": [e.tolist() for e in user.embeddings],
                "enrolled_at": user.enrolled_at,
                "sample_count": user.sample_count,
                "model_version": user.model_version,
            }

        t0 = _time.monotonic()
        try:
            future = _file_io_pool.submit(self._save_file_sync, data)
            future.result(timeout=ENROLLMENT_SAVE_CONTRACT.timeout_sec)
        except TimeoutError:
            timing_ms = round((_time.monotonic() - t0) * 1000, 1)
            _structured_logger.emit_failure(
                operation=ENROLLMENT_SAVE_CONTRACT.boundary_name,
                error_code=ErrorCode.TIMEOUT,
                error_detail=f"File write timed out after {ENROLLMENT_SAVE_CONTRACT.timeout_sec}s",
                trigger="TimeoutError",
                timing_ms=timing_ms,
                **{k: v for k, v in ENROLLMENT_SAVE_CONTRACT.log_fields().items()
                   if k != "boundary_name"},
            )
            raise
        except Exception as e:
            timing_ms = round((_time.monotonic() - t0) * 1000, 1)
            _structured_logger.emit_failure(
                operation=ENROLLMENT_SAVE_CONTRACT.boundary_name,
                error_code=ErrorCode.UNAVAILABLE,
                error_detail=str(e)[:200],
                trigger=type(e).__name__,
                timing_ms=timing_ms,
                **{k: v for k, v in ENROLLMENT_SAVE_CONTRACT.log_fields().items()
                   if k != "boundary_name"},
            )
            raise
        timing_ms = round((_time.monotonic() - t0) * 1000, 1)
        _structured_logger.info(
            "boundary_call",
            **ENROLLMENT_SAVE_CONTRACT.log_fields(),
            ok=True,
            timing_ms=timing_ms,
        )

        logger.info(f"Saved {len(self.users)} users to {self.path}")

    def _save_file_sync(self, data: dict):
        """Write enrollment JSON to disk (runs in executor thread)."""
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def add_user(
        self,
        user_id: str,
        embeddings: list[np.ndarray],
        model_version: str = "fal/AuraFace-v1",
        append: bool = False
    ):
        """Add or update a user's enrollment.

        Args:
            user_id: Unique user identifier
            embeddings: List of L2-normalized embeddings
            model_version: Model version string
            append: If True, append to existing embeddings; otherwise replace
        """
        if not embeddings:
            logger.warning(f"No embeddings provided for {user_id}")
            return

        # Validate dimensions
        for i, e in enumerate(embeddings):
            if e.shape[0] != EXPECTED_EMBEDDING_DIM:
                raise ValueError(
                    f"Embedding {i} has wrong dim {e.shape[0]} "
                    f"(expected {EXPECTED_EMBEDDING_DIM})"
                )

        if append and user_id in self.users:
            existing = self.users[user_id].embeddings
            combined = existing + embeddings
            if len(combined) > MAX_SAMPLES_PER_USER:
                combined = combined[:MAX_SAMPLES_PER_USER]
            embeddings = combined
            enrolled_at = self.users[user_id].enrolled_at
        else:
            enrolled_at = datetime.now().isoformat()

        # Cap samples
        if len(embeddings) > MAX_SAMPLES_PER_USER:
            embeddings = embeddings[:MAX_SAMPLES_PER_USER]

        # Compute centroid
        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        self.users[user_id] = EnrolledUser(
            user_id=user_id,
            centroid=centroid,
            embeddings=embeddings,
            sample_count=len(embeddings),
            enrolled_at=enrolled_at,
            model_version=model_version,
        )

        logger.info(f"Added/updated {user_id} with {len(embeddings)} samples")

    def remove_user(self, user_id: str) -> bool:
        """Remove a user from enrollment.

        Args:
            user_id: User identifier to remove

        Returns:
            True if user was removed, False if not found
        """
        if user_id in self.users:
            del self.users[user_id]
            logger.info(f"Removed {user_id}")
            return True
        return False

    def list_users(self) -> list[str]:
        """List all enrolled user IDs."""
        return list(self.users.keys())

    def get_user_info(self, user_id: str) -> Optional[dict]:
        """Get info about an enrolled user.

        Returns:
            Dict with user info, or None if not found
        """
        if user_id not in self.users:
            return None

        user = self.users[user_id]
        return {
            "user_id": user.user_id,
            "sample_count": user.sample_count,
            "enrolled_at": user.enrolled_at,
            "model_version": user.model_version,
        }
