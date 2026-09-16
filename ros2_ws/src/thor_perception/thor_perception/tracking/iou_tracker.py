"""Lightweight IoU-based face tracker with motion prediction.

This tracker provides frame-to-frame association for face detections using:
- IoU (Intersection over Union) matching with predicted positions
- Constant-velocity motion model
- Hard gates for IoU, distance, and size ratio
- Greedy assignment with weighted cost function

Designed to run in the same process as inference for minimal latency.
"""

from dataclasses import dataclass, field
from typing import Optional
import math
import numpy as np

from thor_telemetry import get_logger as _get_structured_logger, ErrorCode
_structured_logger = _get_structured_logger("iou_tracker")


@dataclass
class TrackedFace:
    """State for a tracked face across frames."""

    track_id: int
    # Bounding box in center+size format (cx, cy, w, h)
    cx: float
    cy: float
    w: float
    h: float
    # Landmarks (10 values: 5 points x 2 coords)
    landmarks: np.ndarray
    confidence: float
    first_seen: float  # monotonic timestamp
    last_seen: float
    detection_count: int = 1
    missed_frames: int = 0
    # Motion prediction (constant-velocity model)
    vx: float = 0.0  # pixels/sec
    vy: float = 0.0
    # Stability tracking
    consecutive_hits: int = 1
    consecutive_misses: int = 0
    stability_score: float = 0.0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Return bbox as (cx, cy, w, h) tuple."""
        return (self.cx, self.cy, self.w, self.h)

    @property
    def face_size(self) -> float:
        """Return min(width, height) as face size metric."""
        return min(self.w, self.h)

    @property
    def is_confirmed(self) -> bool:
        """Track is confirmed if it has enough detections."""
        # This is set externally based on min_hits parameter
        return self.detection_count >= 3  # default min_hits


@dataclass
class MatchDebug:
    """Debug info for a single detection-to-track match."""

    track_id: int
    det_idx: int
    cost: float
    iou: float
    dist: float


@dataclass
class RawDetection:
    """Raw face detection from YuNet (before tracking)."""

    cx: float
    cy: float
    w: float
    h: float
    confidence: float
    landmarks: np.ndarray  # (10,) array: [left_eye_x, left_eye_y, ...]


def compute_iou(box1: tuple[float, float, float, float],
                box2: tuple[float, float, float, float]) -> float:
    """Compute IoU between two boxes in center+size format (cx, cy, w, h)."""
    # Convert to corners
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    # Union
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def compute_size_ratio(size1: float, size2: float) -> float:
    """Compute size ratio as min/max (0-1 range)."""
    if size1 <= 0 or size2 <= 0:
        return 0.0
    return min(size1, size2) / max(size1, size2)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to [min_val, max_val] range."""
    return max(min_val, min(max_val, value))


def update_stability(track: TrackedFace, matched: bool) -> float:
    """Update track's stability score based on hit/miss pattern.

    Stability represents confidence that this is the same physical face over time.
    - Increases with consecutive hits
    - Decreases with consecutive misses
    - Capped by age (young tracks can't be super stable)
    """
    if matched:
        track.consecutive_hits += 1
        track.consecutive_misses = 0
    else:
        track.consecutive_misses += 1
        track.consecutive_hits = 0

    # Base score from consecutive hits (ramp up over 10 frames)
    base = min(1.0, track.consecutive_hits / 10)
    # Age-based ceiling (young tracks can't be super stable)
    age_cap = min(1.0, track.detection_count / 30)
    # Penalty for misses
    miss_penalty = 0.1 * track.consecutive_misses

    track.stability_score = max(0.0, min(base, age_cap) - miss_penalty)
    return track.stability_score


class IoUTracker:
    """Frame-to-frame face tracker using IoU matching with motion prediction.

    Features:
    - Constant-velocity motion model for position prediction
    - Weighted cost function combining IoU and distance
    - Hard gates for IoU, distance, and size ratio
    - Separate retirement thresholds for confirmed vs unconfirmed tracks
    - Debug logging support via DEBUG_TRACKING=1

    Usage:
        tracker = IoUTracker()
        tracks, matches = tracker.update(detections, timestamp)
    """

    def __init__(
        self,
        iou_thresh: float = 0.3,
        dist_thresh: float = 0.15,
        size_ratio_thresh: float = 0.65,
        max_missed: int = 10,
        max_missed_confirmed: int = 15,
        min_hits: int = 3,
        cost_weight_iou: float = 0.6,
        cost_weight_dist: float = 0.4,
        vmax: float = 320.0,
        frame_width: int = 640,
        frame_height: int = 480,
        logger=None,
    ):
        """Initialize tracker with parameters.

        Args:
            iou_thresh: Hard gate - min IoU for association (with exception)
            dist_thresh: Hard gate - max normalized center distance
            size_ratio_thresh: Hard gate - min(a,b)/max(a,b) for face size
            max_missed: Frames before unconfirmed track retirement (~0.67s @ 15 FPS)
            max_missed_confirmed: Frames before confirmed track retirement (~1s)
            min_hits: Detections before track is confirmed
            cost_weight_iou: Weight for (1-IoU) in cost function
            cost_weight_dist: Weight for normalized distance
            vmax: Max velocity in pixels/sec (clamps velocity updates)
            frame_width: Frame width for distance normalization
            frame_height: Frame height for distance normalization
            logger: Optional StructuredLogger for transition records (CP-011)
        """
        self.iou_thresh = iou_thresh
        self.dist_thresh = dist_thresh
        self.size_ratio_thresh = size_ratio_thresh
        self.max_missed = max_missed
        self.max_missed_confirmed = max_missed_confirmed
        self.min_hits = min_hits
        self.cost_weight_iou = cost_weight_iou
        self.cost_weight_dist = cost_weight_dist
        self.vmax = vmax
        self.frame_diagonal = math.sqrt(frame_width**2 + frame_height**2)
        self._logger = logger

        self.tracks: dict[int, TrackedFace] = {}
        self.next_id = 1
        self.last_timestamp: Optional[float] = None

    def update(
        self,
        detections: list[RawDetection],
        timestamp: float,
        debug: bool = False,
    ) -> tuple[list[TrackedFace], list[MatchDebug]]:
        """Associate detections to tracks and return updated tracks.

        Args:
            detections: List of raw face detections from current frame
            timestamp: Monotonic timestamp (time.monotonic())
            debug: If True, populate match debug info

        Returns:
            (tracks, matches): List of active tracks and debug match info
        """
        matches: list[MatchDebug] = []

        # Compute dt and handle timestamp discontinuities
        dt = 0.0
        if self.last_timestamp is not None:
            raw_dt = timestamp - self.last_timestamp
            if raw_dt > 2.0 or raw_dt < -0.1:
                # Timestamp discontinuity - reset tracker state
                # CP-011: emit transition record before mass clear
                if self._logger and self.tracks:
                    self._logger.info(
                        "state_transition",
                        entity_type="face_track",
                        entity_id="bulk",
                        old_state="mixed",
                        new_state="deleted",
                        reason="timestamp_discontinuity",
                        previous_context={"track_count": len(self.tracks)},
                    )
                self.tracks.clear()
                _structured_logger.emit_failure(
                    operation="update_timestamp",
                    error_code=ErrorCode.INTERNAL,
                    error_detail=f"Timestamp discontinuity (raw_dt={raw_dt:.3f}s), tracks cleared",
                    trigger="timestamp_discontinuity",
                )
                self.last_timestamp = timestamp
                return [], []
            # Guard against weird timestamps
            if raw_dt <= 0 or raw_dt > 0.5:
                # Camera hiccup, RTSP jitter - disable velocity update
                dt = 0.0
            else:
                dt = raw_dt
        self.last_timestamp = timestamp

        # Step 1: Predict track positions using constant-velocity model
        predicted_positions: dict[int, tuple[float, float]] = {}
        for track_id, track in self.tracks.items():
            if dt > 0:
                pred_cx = track.cx + track.vx * dt
                pred_cy = track.cy + track.vy * dt
            else:
                pred_cx = track.cx
                pred_cy = track.cy
            predicted_positions[track_id] = (pred_cx, pred_cy)

        # Step 2: Compute cost matrix with hard gates
        # Cost entry: (cost, det_idx, track_id, iou, dist)
        cost_entries: list[tuple[float, int, int, float, float]] = []

        for det_idx, det in enumerate(detections):
            det_box = (det.cx, det.cy, det.w, det.h)
            det_size = min(det.w, det.h)

            for track_id, track in self.tracks.items():
                pred_cx, pred_cy = predicted_positions[track_id]
                # Predicted box uses current track size (w, h don't change much)
                pred_box = (pred_cx, pred_cy, track.w, track.h)

                # Compute metrics
                iou = compute_iou(det_box, pred_box)
                dist = math.sqrt((det.cx - pred_cx) ** 2 + (det.cy - pred_cy) ** 2)
                norm_dist = dist / self.frame_diagonal
                size_ratio = compute_size_ratio(det_size, track.face_size)

                # Apply hard gates
                # Gate 1: Distance threshold
                if norm_dist > self.dist_thresh:
                    continue

                # Gate 2: Size ratio threshold
                if size_ratio < self.size_ratio_thresh:
                    continue

                # Gate 3: IoU threshold with exception
                # Exception: if very close AND similar size, allow slightly low IoU
                # This handles detector bbox jitter without ID churn
                iou_exception = norm_dist < 0.05 and size_ratio > 0.8
                if iou < self.iou_thresh and not iou_exception:
                    continue

                # Compute weighted cost
                cost = self.cost_weight_iou * (1 - iou) + self.cost_weight_dist * norm_dist
                cost_entries.append((cost, det_idx, track_id, iou, norm_dist))

        # Step 3: Greedy assignment by ascending cost
        cost_entries.sort(key=lambda x: x[0])  # Sort by cost ascending

        matched_dets: set[int] = set()
        matched_tracks: set[int] = set()

        for cost, det_idx, track_id, iou, dist in cost_entries:
            if det_idx in matched_dets or track_id in matched_tracks:
                continue

            # Match found
            matched_dets.add(det_idx)
            matched_tracks.add(track_id)

            if debug:
                matches.append(MatchDebug(
                    track_id=track_id,
                    det_idx=det_idx,
                    cost=cost,
                    iou=iou,
                    dist=dist,
                ))

        # Step 4: Update matched tracks
        for det_idx in matched_dets:
            det = detections[det_idx]
            # Find which track this detection matched
            track_id = None
            for cost, d_idx, t_id, _, _ in cost_entries:
                if d_idx == det_idx and t_id in matched_tracks and d_idx in matched_dets:
                    # Check this is the actual match (not just a candidate)
                    for m_cost, m_det, m_track, _, _ in cost_entries:
                        if m_det == det_idx and m_track not in matched_tracks:
                            continue
                        if m_det == det_idx:
                            track_id = m_track
                            break
                    break

            # Re-find the track_id properly
            track_id = None
            for m in matches if debug else []:
                if m.det_idx == det_idx:
                    track_id = m.track_id
                    break

            # If debug is off, we need another way
            if track_id is None:
                for cost, d_idx, t_id, _, _ in cost_entries:
                    if d_idx == det_idx and t_id in matched_tracks:
                        track_id = t_id
                        break

            if track_id is None:
                continue

            track = self.tracks[track_id]

            # Update velocity with clamp (only if dt > 0)
            if dt > 0:
                new_vx = (det.cx - track.cx) / dt
                new_vy = (det.cy - track.cy) / dt
                track.vx = clamp(new_vx, -self.vmax, self.vmax)
                track.vy = clamp(new_vy, -self.vmax, self.vmax)

            # Update position and other state
            track.cx = det.cx
            track.cy = det.cy
            track.w = det.w
            track.h = det.h
            track.landmarks = det.landmarks.copy()
            track.confidence = det.confidence
            track.last_seen = timestamp
            track.detection_count += 1
            track.missed_frames = 0

            # Update stability
            update_stability(track, matched=True)

        # Step 5: Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx in matched_dets:
                continue

            new_track = TrackedFace(
                track_id=self.next_id,
                cx=det.cx,
                cy=det.cy,
                w=det.w,
                h=det.h,
                landmarks=det.landmarks.copy(),
                confidence=det.confidence,
                first_seen=timestamp,
                last_seen=timestamp,
            )
            self.tracks[self.next_id] = new_track
            self.next_id += 1

        # Step 6: Update unmatched tracks and age out stale ones
        tracks_to_remove: list[int] = []

        for track_id, track in self.tracks.items():
            if track_id in matched_tracks:
                continue

            # Track was not matched this frame
            track.missed_frames += 1
            update_stability(track, matched=False)

            # Check retirement threshold
            is_confirmed = track.detection_count >= self.min_hits
            max_miss = self.max_missed_confirmed if is_confirmed else self.max_missed

            if track.missed_frames > max_miss:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            # CP-011: emit transition record before destruction
            if self._logger:
                track = self.tracks[track_id]
                is_confirmed = track.detection_count >= self.min_hits
                self._logger.info(
                    "state_transition",
                    entity_type="face_track",
                    entity_id=str(track_id),
                    old_state="confirmed" if is_confirmed else "unconfirmed",
                    new_state="deleted",
                    reason="missed_frames_exceeded",
                    duration_sec=track.last_seen - track.first_seen,
                    previous_context={
                        "first_seen": track.first_seen,
                        "last_seen": track.last_seen,
                        "detection_count": track.detection_count,
                        "missed_frames": track.missed_frames,
                        "stability_score": track.stability_score,
                    },
                )
            del self.tracks[track_id]

        # Step 7: Return all active tracks
        return list(self.tracks.values()), matches

    def set_frame_size(self, width: int, height: int) -> None:
        """Update frame dimensions for distance normalization."""
        self.frame_diagonal = math.sqrt(width**2 + height**2)

    def clear(self) -> None:
        """Clear all tracks (for manual reset)."""
        # CP-011: emit transition record before mass clear
        if self._logger and self.tracks:
            self._logger.info(
                "state_transition",
                entity_type="face_track",
                entity_id="bulk",
                old_state="mixed",
                new_state="deleted",
                reason="manual_clear",
                previous_context={"track_count": len(self.tracks)},
            )
        self.tracks.clear()
        self.last_timestamp = None
