"""Face-to-person spatial association.

Associates detected faces with tracked persons based on spatial containment
and geometric plausibility. Uses time-coherent person track selection via
a thread-safe ring buffer.

Phase 3.2 implementation for authorized target tracking.
"""

import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

# PersonTrack state constants (from thor_msgs/PersonTrack.msg)
STATE_TENTATIVE = 0
STATE_CONFIRMED = 1
STATE_OCCLUDED = 2
STATE_LOST = 3


@dataclass
class AssociationResult:
    """Result of face-to-person association."""

    person_track_id: int  # 0 if no association
    confidence: float  # 0.0-1.0 score
    ambiguous: bool  # True if rejected due to close second candidate


class PersonTrackBuffer:
    """Thread-safe ring buffer of recent PersonTracks for time-coherent association.

    Stores PersonTracks keyed by timestamp, allowing lookup of the closest
    snapshot to a given face frame timestamp. This prevents association
    artifacts when face and person detections have slightly different timestamps.
    """

    def __init__(self, max_age_sec: float = 0.5):
        """Initialize buffer.

        Args:
            max_age_sec: Maximum age of entries to keep (default 0.5s)
        """
        self.buffer: List[Tuple[float, object]] = []  # (timestamp, PersonTracks)
        self.max_age_sec = max_age_sec
        self._lock = threading.Lock()

    def add(self, msg) -> None:
        """Add PersonTracks to buffer (called from subscription callback).

        Args:
            msg: PersonTracks message with header.stamp
        """
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self._lock:
            self.buffer.append((stamp, msg))
            # Prune old entries
            cutoff = stamp - self.max_age_sec
            self.buffer = [(t, m) for t, m in self.buffer if t > cutoff]

    def get_closest(self, target_stamp: float, tolerance_sec: float = 0.15):
        """Get PersonTracks closest to target timestamp within tolerance.

        Args:
            target_stamp: Target timestamp (seconds)
            tolerance_sec: Maximum allowed time difference

        Returns:
            PersonTracks message or None if no match within tolerance
        """
        with self._lock:
            if not self.buffer:
                return None
            best = min(self.buffer, key=lambda x: abs(x[0] - target_stamp))
            if abs(best[0] - target_stamp) <= tolerance_sec:
                return best[1]
            return None

    def clear(self) -> None:
        """Clear all entries from buffer."""
        with self._lock:
            self.buffer.clear()


def associate_face_to_person(
    face_cx: float,
    face_cy: float,
    face_w: float,
    face_h: float,
    person_tracks: List,
    require_confirmed: bool = True,
    area_ratio_min: float = 0.02,
    area_ratio_max: float = 0.30,
    margin_threshold: float = 0.15,
    min_score: float = 0.3,
) -> AssociationResult:
    """Associate face to best matching person track.

    Uses spatial containment and geometric plausibility to find the best
    person track for a given face detection. Returns no association (0)
    if ambiguous rather than guessing.

    Coordinate Convention (center-based bbox):
    - cx, cy = center coordinates (pixels)
    - w, h = width, height (pixels)
    - top = cy - h/2, bottom = cy + h/2
    - left = cx - w/2, right = cx + w/2

    Args:
        face_cx, face_cy, face_w, face_h: Face bbox (center+size, pixels)
        person_tracks: List of PersonTrack messages
        require_confirmed: If True, only consider CONFIRMED tracks (reduces false locks)
        area_ratio_min: Minimum face/person area ratio gate
        area_ratio_max: Maximum face/person area ratio gate
        margin_threshold: Score margin below which match is ambiguous
        min_score: Minimum score to accept association

    Returns:
        AssociationResult with person_track_id, confidence, ambiguous flag
    """
    candidates: List[Tuple[int, float]] = []

    for track in person_tracks:
        # Hard gate: require CONFIRMED tracks (configurable)
        if require_confirmed and track.state != STATE_CONFIRMED:
            continue

        # Select bbox: use predicted_bbox if track is OCCLUDED
        if track.state == STATE_OCCLUDED:
            pcx = track.predicted_bbox.center.position.x
            pcy = track.predicted_bbox.center.position.y
            pw = track.predicted_bbox.size_x
            ph = track.predicted_bbox.size_y
        else:
            pcx = track.bbox.center.position.x
            pcy = track.bbox.center.position.y
            pw = track.bbox.size_x
            ph = track.bbox.size_y

        # Skip invalid person bboxes
        if pw <= 0 or ph <= 0:
            continue

        # Derived coordinates (explicit to avoid sign errors)
        face_top = face_cy - face_h / 2
        person_top = pcy - ph / 2

        # Gate 1: Face center must be inside person bbox
        person_left = pcx - pw / 2
        person_right = pcx + pw / 2
        person_bottom = pcy + ph / 2
        if not (
            person_left <= face_cx <= person_right
            and person_top <= face_cy <= person_bottom
        ):
            continue

        # Gate 2: Area ratio (face typically 2-30% of person area)
        face_area = face_w * face_h
        person_area = pw * ph
        area_ratio = face_area / person_area
        if area_ratio < area_ratio_min or area_ratio > area_ratio_max:
            continue

        # Gate 3: Vertical position (face in upper 60% of person)
        relative_y = (face_top - person_top) / ph
        if relative_y > 0.6:
            continue

        # Score components:
        # - Horizontal centering (35%): penalize faces near edges
        horiz_offset = abs(face_cx - pcx) / (pw / 2)
        horiz_score = max(0.0, 1.0 - horiz_offset)

        # - Size appropriateness (30%): optimal ~15% area ratio
        size_score = max(0.0, 1.0 - abs(area_ratio - 0.15) / 0.15)

        # - Vertical position (25%): higher in bbox is better
        vert_score = 1.0 - relative_y

        # - Track quality tie-breaker (10%)
        quality_score = track.quality

        score = (
            horiz_score * 0.35
            + size_score * 0.30
            + vert_score * 0.25
            + quality_score * 0.10
        )
        candidates.append((track.track_id, score))

    # No candidates
    if not candidates:
        return AssociationResult(person_track_id=0, confidence=0.0, ambiguous=False)

    # Sort by score descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_id, best_score = candidates[0]

    # Ambiguity check: aligned with Phase 3.1 semantics (margin-based)
    if len(candidates) >= 2:
        second_score = candidates[1][1]
        score_margin = best_score - second_score

        if score_margin < margin_threshold:
            # Ambiguous: return 0 to avoid wrong association
            return AssociationResult(person_track_id=0, confidence=0.0, ambiguous=True)

    # Accept if above minimum score
    if best_score >= min_score:
        return AssociationResult(
            person_track_id=best_id, confidence=best_score, ambiguous=False
        )

    return AssociationResult(person_track_id=0, confidence=0.0, ambiguous=False)
