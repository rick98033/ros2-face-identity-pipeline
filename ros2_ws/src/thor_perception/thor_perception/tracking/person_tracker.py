"""Person tracker with IoU matching, motion prediction, and ambiguity detection.

This tracker provides frame-to-frame association for person detections using:
- IoU + distance matching with continuity bias
- Constant-velocity motion model (normalized coords)
- TIME-based lifecycle (not frame counts)
- Ambiguity detection for downstream authorization
- Reassociation cooldown to prevent thrashing

Designed for Phase 3.1 of PERCEPTION_FOUNDATION.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
import math

from thor_telemetry import get_logger as _get_structured_logger, ErrorCode
_structured_logger = _get_structured_logger("person_tracker")


class TrackState(IntEnum):
    """Track lifecycle states."""
    UNKNOWN = -1    # Uninitialized / unrecognized (CP-010)
    TENTATIVE = 0   # New track, not yet confirmed
    CONFIRMED = 1   # Stable track with sufficient detections
    OCCLUDED = 2    # Missing detections, in retention window
    LOST = 3        # About to be retired (not published)


TERMINAL_STATES = frozenset({TrackState.LOST})
ACTIVE_STATES = frozenset({TrackState.CONFIRMED, TrackState.OCCLUDED})


class TimeStatus(IntEnum):
    """Timestamp health status."""
    UNKNOWN = -1                # Uninitialized / unrecognized (CP-010)
    OK = 0
    DISCONTINUITY_DETECTED = 1  # Brief anomaly, tracks preserved
    FROZEN = 2                  # Persistent anomaly, tracks cleared


class QualityReason(IntEnum):
    """Quality degradation reason flags (bitmask)."""
    UNKNOWN = -1           # Uninitialized / unrecognized (CP-010)
    OK = 0
    LOW_CONF = 1           # detector_confidence < threshold
    AMBIGUOUS_MATCH = 2    # match_margin < threshold
    STALE = 4              # last_update_age_ms > threshold
    TIME_ANOMALY = 8       # time_status != OK


@dataclass
class TrackedPerson:
    """State for a tracked person across frames."""
    track_id: int
    # Bounding box in center+size format (cx, cy, w, h) - pixels
    cx: float
    cy: float
    w: float
    h: float
    confidence: float
    # Timestamps (ROS time as float seconds)
    first_seen: float
    last_seen: float
    # Counters
    detection_count: int = 1
    # Motion model (normalized coords: fraction of frame/sec)
    vx: float = 0.0
    vy: float = 0.0
    # State
    state: TrackState = TrackState.TENTATIVE
    # Stability (EWMA of hit/miss)
    stability_score: float = 0.5  # Start at 0.5 for new tracks
    # Ambiguity
    match_margin: float = float('inf')  # High = unambiguous
    quality_reason: int = QualityReason.OK
    # Reassociation tracking
    last_det_idx: int = -1
    last_reassoc_time: float = 0.0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Return bbox as (cx, cy, w, h) tuple."""
        return (self.cx, self.cy, self.w, self.h)

    @property
    def quality(self) -> float:
        """Compute quality as confidence * stability."""
        return self.confidence * self.stability_score


@dataclass
class PersonDetection:
    """Input detection from person detector."""
    cx: float
    cy: float
    w: float
    h: float
    confidence: float
    det_idx: int  # Original index in Detection2DArray


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


class PersonTracker:
    """Person tracker with IoU matching, motion prediction, and ambiguity detection.

    Key features:
    - TIME-based lifecycle thresholds (seconds, not frame counts)
    - Normalized velocity (resolution-invariant)
    - Ambiguity detection via match_margin
    - Continuity bias to reduce thrashing
    - Reassociation cooldown to enforce max 2/sec
    - Timestamp anomaly handling (freeze, not clear)
    """

    # Stability EWMA alpha (time constant ~10 updates)
    STABILITY_ALPHA = 0.1

    def __init__(
        self,
        # Association thresholds
        iou_threshold: float = 0.3,
        dist_threshold: float = 0.20,
        size_ratio_threshold: float = 0.5,
        # Lifecycle thresholds (TIME-based)
        max_gap_tentative_sec: float = 0.5,
        max_gap_confirmed_sec: float = 1.0,
        min_hits_to_confirm: int = 3,
        # Cost function
        cost_weight_iou: float = 0.6,
        cost_weight_dist: float = 0.4,
        continuity_bias: float = 0.9,
        # Ambiguity
        margin_threshold: float = 0.1,
        # Reassociation cooldown
        reassoc_cooldown_sec: float = 0.5,
        # Motion model
        vmax_normalized: float = 0.6,  # fraction of frame_width/sec
        # Frame dimensions
        frame_width: int = 640,
        frame_height: int = 480,
        # Telemetry (CP-011: optional logger for transition records)
        logger=None,
    ):
        self.iou_threshold = iou_threshold
        self.dist_threshold = dist_threshold
        self.size_ratio_threshold = size_ratio_threshold
        self.max_gap_tentative_sec = max_gap_tentative_sec
        self.max_gap_confirmed_sec = max_gap_confirmed_sec
        self.min_hits_to_confirm = min_hits_to_confirm
        self.cost_weight_iou = cost_weight_iou
        self.cost_weight_dist = cost_weight_dist
        self.continuity_bias = continuity_bias
        self.margin_threshold = margin_threshold
        self.reassoc_cooldown_sec = reassoc_cooldown_sec
        self.vmax_normalized = vmax_normalized
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_diagonal = math.sqrt(frame_width**2 + frame_height**2)
        self._logger = logger

        # State
        self.tracks: dict[int, TrackedPerson] = {}
        self.next_id: int = 1
        self.last_timestamp: Optional[float] = None
        self.time_status: TimeStatus = TimeStatus.OK
        self.consecutive_anomalies: int = 0

        # Metrics
        self.total_tracks_created: int = 0

    def update(
        self,
        detections: list[PersonDetection],
        timestamp: float,
    ) -> list[TrackedPerson]:
        """Associate detections to tracks and return active tracks.

        Args:
            detections: List of person detections from current frame
            timestamp: ROS time as float seconds

        Returns:
            List of active tracks (excludes LOST)
        """
        # Handle timestamp and compute dt
        dt = self._handle_timestamp(timestamp)

        # If frozen (persistent anomaly), return empty
        if self.time_status == TimeStatus.FROZEN:
            return []

        # Predict track positions
        predicted_positions = self._predict_positions(dt)

        # Compute cost entries with gates
        cost_entries = self._compute_costs(detections, predicted_positions, timestamp)

        # Build per-track candidate lists for ambiguity computation
        track_candidates: dict[int, list[tuple[float, int]]] = {}
        for cost, det_idx, track_id in cost_entries:
            if track_id not in track_candidates:
                track_candidates[track_id] = []
            track_candidates[track_id].append((cost, det_idx))

        # Greedy assignment
        matched_dets, matched_tracks, assignments = self._greedy_assign(
            cost_entries, timestamp
        )

        # Update matched tracks
        self._update_matched(
            detections, assignments, timestamp, dt, track_candidates
        )

        # Spawn new tracks for unmatched detections
        self._spawn_new_tracks(detections, matched_dets, timestamp)

        # Age unmatched tracks
        self._age_tracks(matched_tracks, timestamp)

        # Remove LOST tracks
        self._remove_lost_tracks()

        # Return active tracks (not LOST)
        return [t for t in self.tracks.values() if t.state not in TERMINAL_STATES]

    def _handle_timestamp(self, timestamp: float) -> float:
        """Handle timestamp discontinuities. Returns dt for velocity update."""
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            self.time_status = TimeStatus.OK
            return 0.0

        dt = timestamp - self.last_timestamp

        # Check for anomaly
        if dt < -0.1 or dt > 2.0:
            self.consecutive_anomalies += 1
            self.time_status = TimeStatus.DISCONTINUITY_DETECTED

            if self.consecutive_anomalies >= 5:
                # Persistent anomaly: clear tracks
                # CP-011: emit transition record before mass clear
                if self._logger and self.tracks:
                    self._logger.info(
                        "state_transition",
                        entity_type="track",
                        entity_id="bulk",
                        old_state="mixed",
                        new_state="deleted",
                        reason="timestamp_anomaly_persistent",
                        previous_context={"track_count": len(self.tracks)},
                    )
                self.tracks.clear()
                self.time_status = TimeStatus.FROZEN
                _structured_logger.emit_failure(
                    operation="handle_timestamp",
                    error_code=ErrorCode.INTERNAL,
                    error_detail=f"Persistent timestamp anomaly ({self.consecutive_anomalies} consecutive), tracks cleared",
                    trigger="timestamp_anomaly_persistent",
                )
                self.last_timestamp = timestamp
                return 0.0
            else:
                # Freeze prediction, treat as missed update
                # Don't clear - preserve tracks for graceful recovery
                return 0.0
        else:
            # Normal timestamp
            self.consecutive_anomalies = 0
            self.time_status = TimeStatus.OK
            self.last_timestamp = timestamp

            # Guard: very small or zero dt
            if dt <= 0:
                return 0.0

            # Guard: moderate gap (>0.5s) - disable velocity update
            if dt > 0.5:
                return 0.0

            return dt

    def _predict_positions(self, dt: float) -> dict[int, tuple[float, float]]:
        """Predict track positions using constant-velocity model."""
        predicted = {}
        for track_id, track in self.tracks.items():
            if dt > 0:
                # Velocity in normalized coords, convert to pixels
                pred_cx = track.cx + track.vx * dt * self.frame_width
                pred_cy = track.cy + track.vy * dt * self.frame_height
            else:
                pred_cx = track.cx
                pred_cy = track.cy
            predicted[track_id] = (pred_cx, pred_cy)
        return predicted

    def _compute_costs(
        self,
        detections: list[PersonDetection],
        predicted_positions: dict[int, tuple[float, float]],
        timestamp: float,
    ) -> list[tuple[float, int, int]]:
        """Compute cost entries with hard gates and continuity bias.

        Returns: list of (cost, det_idx, track_id)
        """
        entries = []

        for det in detections:
            det_box = (det.cx, det.cy, det.w, det.h)
            det_size = min(det.w, det.h)

            for track_id, track in self.tracks.items():
                if track.state == TrackState.LOST:
                    continue

                pred_cx, pred_cy = predicted_positions[track_id]
                pred_box = (pred_cx, pred_cy, track.w, track.h)

                # Compute metrics
                iou = compute_iou(det_box, pred_box)
                dist = math.sqrt((det.cx - pred_cx)**2 + (det.cy - pred_cy)**2)
                norm_dist = dist / self.frame_diagonal
                size_ratio = compute_size_ratio(det_size, min(track.w, track.h))

                # Hard gates
                if norm_dist > self.dist_threshold:
                    continue
                if size_ratio < self.size_ratio_threshold:
                    continue

                # IoU gate with proximity exception
                iou_exception = norm_dist < 0.05 and size_ratio > 0.7
                if iou < self.iou_threshold and not iou_exception:
                    continue

                # Base cost
                base_cost = (
                    self.cost_weight_iou * (1 - iou) +
                    self.cost_weight_dist * norm_dist
                )

                # Continuity bias: favor previous assignment
                if track.last_det_idx == det.det_idx:
                    cost = base_cost * self.continuity_bias
                else:
                    cost = base_cost

                entries.append((cost, det.det_idx, track_id))

        return entries

    def _greedy_assign(
        self,
        cost_entries: list[tuple[float, int, int]],
        timestamp: float,
    ) -> tuple[set[int], set[int], dict[int, int]]:
        """Greedy assignment with reassociation cooldown.

        Returns: (matched_dets, matched_tracks, assignments: det_idx -> track_id)
        """
        # Sort by cost ascending
        cost_entries.sort(key=lambda x: x[0])

        matched_dets: set[int] = set()
        matched_tracks: set[int] = set()
        assignments: dict[int, int] = {}  # det_idx -> track_id

        # Build candidate lists for cooldown check
        track_best_costs: dict[int, list[tuple[float, int]]] = {}
        for cost, det_idx, track_id in cost_entries:
            if track_id not in track_best_costs:
                track_best_costs[track_id] = []
            track_best_costs[track_id].append((cost, det_idx))

        for cost, det_idx, track_id in cost_entries:
            if det_idx in matched_dets or track_id in matched_tracks:
                continue

            track = self.tracks[track_id]

            # Check reassociation cooldown
            is_reassociation = (
                track.last_det_idx >= 0 and
                track.last_det_idx != det_idx
            )

            if is_reassociation:
                time_since_reassoc = timestamp - track.last_reassoc_time
                if time_since_reassoc < self.reassoc_cooldown_sec:
                    # During cooldown, compute margin for this candidate
                    candidates = track_best_costs.get(track_id, [])
                    if len(candidates) >= 2:
                        sorted_cands = sorted(candidates, key=lambda x: x[0])
                        best_cost = sorted_cands[0][0]
                        second_cost = sorted_cands[1][0]
                        margin = second_cost - best_cost
                    else:
                        margin = float('inf')

                    # Only allow if margin >> threshold
                    if margin <= 3 * self.margin_threshold:
                        continue  # Reject reassociation during cooldown

            # Accept match
            matched_dets.add(det_idx)
            matched_tracks.add(track_id)
            assignments[det_idx] = track_id

        return matched_dets, matched_tracks, assignments

    def _update_matched(
        self,
        detections: list[PersonDetection],
        assignments: dict[int, int],
        timestamp: float,
        dt: float,
        track_candidates: dict[int, list[tuple[float, int]]],
    ):
        """Update tracks that were matched to detections."""
        for det_idx, track_id in assignments.items():
            det = detections[det_idx]
            track = self.tracks[track_id]

            # Check if this is a reassociation
            is_reassociation = (
                track.last_det_idx >= 0 and
                track.last_det_idx != det_idx
            )
            if is_reassociation:
                track.last_reassoc_time = timestamp

            # Update velocity (normalized coords)
            if dt > 0:
                # Compute pixel velocity then normalize
                new_vx_pixels = (det.cx - track.cx) / dt
                new_vy_pixels = (det.cy - track.cy) / dt
                new_vx = new_vx_pixels / self.frame_width
                new_vy = new_vy_pixels / self.frame_height
                track.vx = clamp(new_vx, -self.vmax_normalized, self.vmax_normalized)
                track.vy = clamp(new_vy, -self.vmax_normalized, self.vmax_normalized)

            # Update position
            track.cx = det.cx
            track.cy = det.cy
            track.w = det.w
            track.h = det.h
            track.confidence = det.confidence
            track.last_seen = timestamp
            track.detection_count += 1
            track.last_det_idx = det_idx

            # Update stability (EWMA)
            track.stability_score = (
                (1 - self.STABILITY_ALPHA) * track.stability_score +
                self.STABILITY_ALPHA * 1.0
            )

            # Compute ambiguity (match_margin)
            candidates = track_candidates.get(track_id, [])
            if len(candidates) >= 2:
                sorted_cands = sorted(candidates, key=lambda x: x[0])
                best_cost = sorted_cands[0][0]
                second_cost = sorted_cands[1][0]
                track.match_margin = second_cost - best_cost
            else:
                track.match_margin = float('inf')

            # Update quality_reason
            track.quality_reason = QualityReason.OK
            if track.confidence < 0.5:
                track.quality_reason |= QualityReason.LOW_CONF
            if track.match_margin < self.margin_threshold:
                track.quality_reason |= QualityReason.AMBIGUOUS_MATCH
                # Also degrade stability on ambiguous match
                track.stability_score *= 0.5
            if self.time_status != TimeStatus.OK:
                track.quality_reason |= QualityReason.TIME_ANOMALY

            # Update state
            if track.state == TrackState.TENTATIVE:
                if track.detection_count >= self.min_hits_to_confirm:
                    track.state = TrackState.CONFIRMED
            elif track.state in (TrackState.OCCLUDED, TrackState.LOST):
                # Recovered
                track.state = TrackState.CONFIRMED

    def _spawn_new_tracks(
        self,
        detections: list[PersonDetection],
        matched_dets: set[int],
        timestamp: float,
    ):
        """Create new tracks for unmatched detections."""
        for det in detections:
            if det.det_idx in matched_dets:
                continue

            new_track = TrackedPerson(
                track_id=self.next_id,
                cx=det.cx,
                cy=det.cy,
                w=det.w,
                h=det.h,
                confidence=det.confidence,
                first_seen=timestamp,
                last_seen=timestamp,
                last_det_idx=det.det_idx,
            )
            self.tracks[self.next_id] = new_track
            self.next_id += 1
            self.total_tracks_created += 1

    def _age_tracks(self, matched_tracks: set[int], timestamp: float):
        """Age unmatched tracks and update their states."""
        for track_id, track in self.tracks.items():
            if track_id in matched_tracks:
                continue

            # Track was not matched
            # Update stability (EWMA with miss)
            track.stability_score = (
                (1 - self.STABILITY_ALPHA) * track.stability_score +
                self.STABILITY_ALPHA * 0.0
            )

            # Check gap
            gap = timestamp - track.last_seen
            is_confirmed = track.state in ACTIVE_STATES
            max_gap = (
                self.max_gap_confirmed_sec if is_confirmed
                else self.max_gap_tentative_sec
            )

            if gap > max_gap:
                track.state = TrackState.LOST
            elif track.state == TrackState.CONFIRMED:
                track.state = TrackState.OCCLUDED

            # Update stale quality reason
            age_ms = int((timestamp - track.last_seen) * 1000)
            if age_ms > 200:  # 200ms threshold for stale
                track.quality_reason |= QualityReason.STALE

    def _remove_lost_tracks(self):
        """Remove LOST tracks from internal dict."""
        to_remove = [
            track_id for track_id, track in self.tracks.items()
            if track.state == TrackState.LOST
        ]
        for track_id in to_remove:
            # CP-011: emit transition record before destruction
            if self._logger:
                track = self.tracks[track_id]
                is_confirmed = track.detection_count >= self.min_hits_to_confirm
                reason = (
                    "gap_exceeded_confirmed" if is_confirmed
                    else "gap_exceeded_tentative"
                )
                self._logger.info(
                    "state_transition",
                    entity_type="track",
                    entity_id=str(track_id),
                    old_state="LOST",
                    new_state="deleted",
                    reason=reason,
                    duration_sec=track.last_seen - track.first_seen,
                    previous_context={
                        "first_seen": track.first_seen,
                        "last_seen": track.last_seen,
                        "detection_count": track.detection_count,
                        "stability_score": track.stability_score,
                        "quality_reason": int(track.quality_reason),
                    },
                )
            del self.tracks[track_id]

    def set_frame_size(self, width: int, height: int):
        """Update frame dimensions."""
        self.frame_width = width
        self.frame_height = height
        self.frame_diagonal = math.sqrt(width**2 + height**2)

    def clear(self):
        """Clear all tracks."""
        # CP-011: emit transition record before mass clear
        if self._logger and self.tracks:
            self._logger.info(
                "state_transition",
                entity_type="track",
                entity_id="bulk",
                old_state="mixed",
                new_state="deleted",
                reason="manual_clear",
                previous_context={"track_count": len(self.tracks)},
            )
        self.tracks.clear()
        self.last_timestamp = None
        self.time_status = TimeStatus.OK
        self.consecutive_anomalies = 0
