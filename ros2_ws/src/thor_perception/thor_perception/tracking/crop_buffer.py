"""Per-track face crop buffer for on-demand AuraFace inference.

Writes to /tmp/thor_face_crops (shared volume with thor_pipeline container).

Bounded resource with explicit cleanup:
- Max 500 total crops on disk
- Max 3 crops per track
- Delete-on-evict when track retires or crop ages out
- Periodic cleanup timer (every 10s)
- JPEG quality 85, max size 160x160 (aligned to AuraFace input)
"""

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from thor_telemetry import (
    get_logger as _get_structured_logger,
    BoundaryContract, BoundaryType, ErrorBehavior, ErrorCode,
)

_structured_logger = _get_structured_logger("crop_buffer")

# Shared volume mounted into both thor_perception and thor_pipeline containers
CROP_DIR = Path("/tmp/thor_face_crops")
MAX_TOTAL_CROPS = 500
MAX_CROPS_PER_TRACK = 3
MAX_AGE_SEC = 30.0
JPEG_QUALITY = 85
CROP_SIZE = (160, 160)  # AuraFace input size

# --- CP-010 Boundary Contracts (SI-11.1) ---
_file_io_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="crop_io")

CROP_WRITE_CONTRACT = BoundaryContract(
    boundary_name="crop_buffer_write",
    boundary_type=BoundaryType.FILE_IO,
    timeout_sec=1.0,
    error_behavior=ErrorBehavior.DEGRADE,
    retry_policy=None,
    error_codes=frozenset({ErrorCode.TIMEOUT, ErrorCode.UNAVAILABLE}),
)

CROP_CLEANUP_CONTRACT = BoundaryContract(
    boundary_name="crop_buffer_cleanup",
    boundary_type=BoundaryType.FILE_IO,
    timeout_sec=5.0,
    error_behavior=ErrorBehavior.DEGRADE,
    retry_policy=None,
    error_codes=frozenset({ErrorCode.TIMEOUT, ErrorCode.UNAVAILABLE}),
)


@dataclass
class CropEntry:
    """Metadata for a saved crop."""

    track_id: int
    crop_path: str
    timestamp: float
    quality_flags: list[str]
    bbox: tuple[int, int, int, int]  # x, y, w, h
    area: int


class FaceCropBuffer:
    """Per-track ring buffer of recent face crops.

    Crops are written to shared volume /tmp/thor_face_crops for cross-container access.
    IdentityEvidenceNode (in thor_pipeline container) reads crops via file paths.

    Thread-safe: all methods are protected by a lock.
    """

    def __init__(self, crop_dir: Path = CROP_DIR, logger=None):
        """Initialize crop buffer.

        Args:
            crop_dir: Directory to store crops (default: /tmp/thor_face_crops)
            logger: Optional ROS logger for debug messages
        """
        self._store: dict[int, deque[CropEntry]] = {}
        self._lock = Lock()
        self._total_crops = 0
        self._crop_dir = crop_dir
        self._logger = logger
        self._crop_dir.mkdir(parents=True, exist_ok=True)

    def add_crop(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        quality_flags: list[str],
    ) -> Optional[str]:
        """Save crop to shared volume and add to buffer.

        Args:
            track_id: Face track ID
            frame: Full frame (BGR)
            bbox: Bounding box (x, y, w, h)
            quality_flags: Quality flags from face detection

        Returns:
            Path to saved crop, or None if save failed
        """
        x, y, w, h = bbox

        # Clamp to frame bounds
        x = max(0, x)
        y = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)

        if x2 <= x or y2 <= y:
            return None

        crop = frame[y:y2, x:x2]
        if crop.size == 0:
            return None

        # Resize to AuraFace input size
        crop = cv2.resize(crop, CROP_SIZE, interpolation=cv2.INTER_LINEAR)

        ts = time.time()
        filename = f"{track_id}_{int(ts * 1000)}.jpg"
        crop_path = str(self._crop_dir / filename)

        t0 = time.monotonic()
        try:
            future = _file_io_pool.submit(cv2.imwrite, crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            future.result(timeout=CROP_WRITE_CONTRACT.timeout_sec)
        except TimeoutError:
            timing_ms = round((time.monotonic() - t0) * 1000, 1)
            if self._logger:
                self._logger.warning(f"Crop write timed out: {crop_path}")
            _structured_logger.emit_failure(
                operation=CROP_WRITE_CONTRACT.boundary_name,
                error_code=ErrorCode.TIMEOUT,
                error_detail=f"cv2.imwrite timed out after {CROP_WRITE_CONTRACT.timeout_sec}s",
                trigger="TimeoutError",
                timing_ms=timing_ms,
                **{k: v for k, v in CROP_WRITE_CONTRACT.log_fields().items()
                   if k != "boundary_name"},
            )
            return None
        except Exception as e:
            timing_ms = round((time.monotonic() - t0) * 1000, 1)
            if self._logger:
                self._logger.warning(f"Failed to save crop: {e}")
            _structured_logger.emit_failure(
                operation=CROP_WRITE_CONTRACT.boundary_name,
                error_code=ErrorCode.UNAVAILABLE,
                error_detail=str(e)[:200],
                trigger=type(e).__name__,
                timing_ms=timing_ms,
                **{k: v for k, v in CROP_WRITE_CONTRACT.log_fields().items()
                   if k != "boundary_name"},
            )
            return None

        entry = CropEntry(
            track_id=track_id,
            crop_path=crop_path,
            timestamp=ts,
            quality_flags=quality_flags,
            bbox=bbox,
            area=w * h,
        )

        with self._lock:
            if track_id not in self._store:
                self._store[track_id] = deque(maxlen=MAX_CROPS_PER_TRACK)

            # Evict oldest for this track if at capacity
            if len(self._store[track_id]) >= MAX_CROPS_PER_TRACK:
                evicted = self._store[track_id].popleft()
                self._delete_crop(evicted.crop_path)
                self._total_crops -= 1

            self._store[track_id].append(entry)
            self._total_crops += 1

            # Global eviction if over total limit
            if self._total_crops > MAX_TOTAL_CROPS:
                self._evict_oldest_global()

        return crop_path

    def get_best_crop(self, track_id: int) -> Optional[str]:
        """Get most recent crop path for a track.

        Args:
            track_id: Face track ID

        Returns:
            Path to crop file, or None if not found
        """
        with self._lock:
            if track_id not in self._store:
                return None
            crops = self._store[track_id]
            if not crops:
                return None
            # Return most recent (rightmost in deque)
            return crops[-1].crop_path

    def retire_track(self, track_id: int) -> None:
        """Remove all crops for a retired track.

        Called when a face track is lost.

        Args:
            track_id: Face track ID to retire
        """
        with self._lock:
            if track_id not in self._store:
                return
            for entry in self._store[track_id]:
                self._delete_crop(entry.crop_path)
                self._total_crops -= 1
            del self._store[track_id]

    def cleanup_stale(self) -> int:
        """Remove crops older than MAX_AGE_SEC.

        Returns:
            Number of crops deleted
        """
        now = time.time()
        deleted = 0

        with self._lock:
            for track_id in list(self._store.keys()):
                new_deque = deque(maxlen=MAX_CROPS_PER_TRACK)
                for entry in self._store[track_id]:
                    if now - entry.timestamp > MAX_AGE_SEC:
                        self._delete_crop(entry.crop_path)
                        self._total_crops -= 1
                        deleted += 1
                    else:
                        new_deque.append(entry)
                if new_deque:
                    self._store[track_id] = new_deque
                else:
                    del self._store[track_id]

        return deleted

    def cleanup_orphan_files(self) -> int:
        """Remove crop files not tracked in buffer.

        Handles files left behind from crashes or restarts.

        Returns:
            Number of orphan files deleted
        """
        tracked_paths: set[str] = set()

        with self._lock:
            for crops in self._store.values():
                for entry in crops:
                    tracked_paths.add(entry.crop_path)

        t0 = time.monotonic()
        try:
            future = _file_io_pool.submit(self._cleanup_orphans_sync, tracked_paths)
            deleted = future.result(timeout=CROP_CLEANUP_CONTRACT.timeout_sec)
        except TimeoutError:
            timing_ms = round((time.monotonic() - t0) * 1000, 1)
            _structured_logger.emit_failure(
                operation=CROP_CLEANUP_CONTRACT.boundary_name,
                error_code=ErrorCode.TIMEOUT,
                error_detail=f"Orphan cleanup timed out after {CROP_CLEANUP_CONTRACT.timeout_sec}s",
                trigger="TimeoutError",
                timing_ms=timing_ms,
                **{k: v for k, v in CROP_CLEANUP_CONTRACT.log_fields().items()
                   if k != "boundary_name"},
            )
            return 0
        except Exception:
            return 0

        return deleted

    def get_stats(self) -> dict:
        """Get buffer statistics.

        Returns:
            Dict with total_crops, num_tracks, disk_bytes
        """
        try:
            future = _file_io_pool.submit(self._compute_disk_bytes_sync)
            disk_bytes = future.result(timeout=CROP_CLEANUP_CONTRACT.timeout_sec)
        except (TimeoutError, Exception):
            disk_bytes = 0

        with self._lock:
            return {
                "total_crops": self._total_crops,
                "num_tracks": len(self._store),
                "disk_bytes": disk_bytes,
            }

    def _cleanup_orphans_sync(self, tracked_paths: set[str]) -> int:
        """Remove orphan files synchronously (runs in executor thread)."""
        deleted = 0
        try:
            for path in self._crop_dir.glob("*.jpg"):
                if str(path) not in tracked_paths:
                    try:
                        path.unlink()
                        deleted += 1
                    except OSError:
                        pass
        except Exception:
            pass
        return deleted

    def _compute_disk_bytes_sync(self) -> int:
        """Compute total disk usage synchronously (runs in executor thread)."""
        disk_bytes = 0
        try:
            for path in self._crop_dir.glob("*.jpg"):
                try:
                    disk_bytes += path.stat().st_size
                except OSError:
                    pass
        except Exception:
            pass
        return disk_bytes

    def _evict_oldest_global(self) -> None:
        """Evict oldest crop across all tracks (called with lock held)."""
        oldest_entry: Optional[CropEntry] = None
        oldest_track_id: Optional[int] = None

        for track_id, crops in self._store.items():
            if crops and (oldest_entry is None or crops[0].timestamp < oldest_entry.timestamp):
                oldest_entry = crops[0]
                oldest_track_id = track_id

        if oldest_entry and oldest_track_id is not None:
            self._store[oldest_track_id].popleft()
            self._delete_crop(oldest_entry.crop_path)
            self._total_crops -= 1

            # Clean up empty track
            if not self._store[oldest_track_id]:
                del self._store[oldest_track_id]

    def _delete_crop(self, path: str) -> None:
        """Delete crop file (called with or without lock)."""
        try:
            os.unlink(path)
        except OSError:
            pass  # File already deleted or inaccessible
