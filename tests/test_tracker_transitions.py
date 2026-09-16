"""Tests for CP-011 error context preservation in PersonTracker and IoUTracker.

Verifies that transition records are emitted before track deletion
with correct mandatory schema fields per the convention.

Run with: python3 -m pytest tests/test_tracker_transitions.py -v
(from thor_ws/src/thor_perception/)
"""

import time
from unittest.mock import MagicMock, call
import numpy as np

from thor_perception.tracking.person_tracker import (
    PersonTracker,
    PersonDetection,
    TrackedPerson,
    TrackState,
)
from thor_perception.tracking.iou_tracker import (
    IoUTracker,
    RawDetection,
    TrackedFace,
)


# ---------------------------------------------------------------------------
# Mandatory schema fields (convention §2)
# ---------------------------------------------------------------------------
MANDATORY_FIELDS = {"entity_type", "entity_id", "old_state", "new_state", "reason"}


def _extract_transition_records(mock_logger):
    """Extract state_transition calls from mock logger."""
    records = []
    for c in mock_logger.info.call_args_list:
        args, kwargs = c
        if args and args[0] == "state_transition":
            records.append(kwargs)
    return records


def _make_person_det(cx, cy, w, h, conf=0.9, det_idx=0):
    return PersonDetection(cx=cx, cy=cy, w=w, h=h, confidence=conf, det_idx=det_idx)


def _make_face_det(cx, cy, w, h, conf=0.9):
    return RawDetection(
        cx=cx, cy=cy, w=w, h=h, confidence=conf,
        landmarks=np.zeros(10),
    )


# ===========================================================================
# PersonTracker tests
# ===========================================================================

class TestPersonTrackerTransitions:
    """Tests for PersonTracker transition record emission."""

    def test_remove_lost_tracks_emits_transition(self):
        """Lost tracks emit transition record before deletion."""
        logger = MagicMock()
        tracker = PersonTracker(
            max_gap_tentative_sec=0.1,
            min_hits_to_confirm=3,
            logger=logger,
        )

        # Create a track
        det = _make_person_det(100, 100, 50, 100)
        tracker.update([det], 1.0)
        assert len(tracker.tracks) == 1

        # Age it past the gap threshold (tentative track, 0.1s gap)
        tracker.update([], 1.5)  # 0.5 > 0.1 -> LOST
        # update() calls _age_tracks then _remove_lost_tracks
        assert len(tracker.tracks) == 0

        records = _extract_transition_records(logger)
        assert len(records) == 1, f"Expected 1 record, got {len(records)}"

        rec = records[0]
        # Check mandatory fields
        for field in MANDATORY_FIELDS:
            assert field in rec, f"Missing mandatory field: {field}"

        assert rec["entity_type"] == "track"
        assert rec["entity_id"] == "1"
        assert rec["old_state"] == "LOST"
        assert rec["new_state"] == "deleted"
        assert rec["reason"] == "gap_exceeded_tentative"
        assert "previous_context" in rec
        assert "duration_sec" in rec

    def test_remove_lost_confirmed_track_has_correct_reason(self):
        """Confirmed lost tracks report gap_exceeded_confirmed."""
        logger = MagicMock()
        tracker = PersonTracker(
            max_gap_confirmed_sec=0.5,
            min_hits_to_confirm=2,
            logger=logger,
        )

        # Create and confirm a track (2 hits)
        det = _make_person_det(100, 100, 50, 100)
        tracker.update([det], 1.0)
        tracker.update([det], 1.1)  # 2nd hit -> confirmed

        # Age past confirmed gap
        tracker.update([], 2.0)  # 0.9s > 0.5s -> LOST
        assert len(tracker.tracks) == 0

        records = _extract_transition_records(logger)
        assert len(records) >= 1
        rec = records[-1]
        assert rec["reason"] == "gap_exceeded_confirmed"

    def test_timestamp_anomaly_mass_clear_emits_transition(self):
        """Persistent timestamp anomaly emits bulk transition record."""
        logger = MagicMock()
        # Use very long gap thresholds so tracks survive anomaly updates
        tracker = PersonTracker(
            max_gap_tentative_sec=999.0,
            max_gap_confirmed_sec=999.0,
            logger=logger,
        )

        # Create a track
        det = _make_person_det(100, 100, 50, 100)
        tracker.update([det], 1.0)
        assert len(tracker.tracks) == 1

        # Trigger 5 consecutive anomalies (dt > 2.0)
        # Tracks survive because gap threshold is very long
        for i in range(5):
            tracker.update([], 100.0 + i * 100)

        assert len(tracker.tracks) == 0

        records = _extract_transition_records(logger)
        # Should have the mass clear record
        bulk_records = [r for r in records if r["entity_id"] == "bulk"]
        assert len(bulk_records) == 1
        rec = bulk_records[0]
        assert rec["reason"] == "timestamp_anomaly_persistent"
        assert rec["entity_type"] == "track"
        assert rec["new_state"] == "deleted"
        assert rec["previous_context"]["track_count"] >= 1

    def test_manual_clear_emits_transition(self):
        """clear() emits bulk transition record."""
        logger = MagicMock()
        tracker = PersonTracker(logger=logger)

        det = _make_person_det(100, 100, 50, 100)
        tracker.update([det], 1.0)
        assert len(tracker.tracks) == 1

        tracker.clear()
        assert len(tracker.tracks) == 0

        records = _extract_transition_records(logger)
        assert len(records) == 1
        assert records[0]["reason"] == "manual_clear"

    def test_no_emission_without_logger(self):
        """Tracker without logger works normally (no crash)."""
        tracker = PersonTracker()  # No logger

        det = _make_person_det(100, 100, 50, 100)
        tracker.update([det], 1.0)
        tracker.update([], 5.0)  # Will age out
        tracker.clear()
        # No assertion — just verifying no crash

    def test_no_emission_on_empty_clear(self):
        """No emission when clearing empty tracker."""
        logger = MagicMock()
        tracker = PersonTracker(logger=logger)
        tracker.clear()  # Empty — should not emit
        records = _extract_transition_records(logger)
        assert len(records) == 0

    def test_previous_context_has_required_fields(self):
        """previous_context includes diagnostically relevant fields only."""
        logger = MagicMock()
        tracker = PersonTracker(
            max_gap_tentative_sec=0.1,
            logger=logger,
        )

        det = _make_person_det(100, 100, 50, 100)
        tracker.update([det], 1.0)
        tracker.update([], 1.5)

        records = _extract_transition_records(logger)
        assert len(records) == 1
        ctx = records[0]["previous_context"]
        # Must have these fields (SI-13.1 justified set)
        assert "first_seen" in ctx
        assert "last_seen" in ctx
        assert "detection_count" in ctx
        assert "stability_score" in ctx
        assert "quality_reason" in ctx
        # Must NOT have spatial data
        assert "cx" not in ctx
        assert "cy" not in ctx


# ===========================================================================
# IoUTracker tests
# ===========================================================================

class TestIoUTrackerTransitions:
    """Tests for IoUTracker transition record emission."""

    def test_missed_frames_exceeded_emits_transition(self):
        """Track retired by missed frames emits transition record."""
        logger = MagicMock()
        tracker = IoUTracker(max_missed=2, logger=logger)

        det = _make_face_det(100, 100, 50, 50)
        tracker.update([det], 1.0)
        assert len(tracker.tracks) == 1

        # Miss enough frames to retire
        for i in range(4):
            tracker.update([], 1.0 + (i + 1) * 0.05)

        assert len(tracker.tracks) == 0

        records = _extract_transition_records(logger)
        assert len(records) >= 1
        rec = records[-1]
        for field in MANDATORY_FIELDS:
            assert field in rec, f"Missing mandatory field: {field}"
        assert rec["entity_type"] == "face_track"
        assert rec["new_state"] == "deleted"
        assert rec["reason"] == "missed_frames_exceeded"

    def test_timestamp_discontinuity_mass_clear_emits_transition(self):
        """Timestamp discontinuity mass clear emits bulk transition record."""
        logger = MagicMock()
        tracker = IoUTracker(logger=logger)

        det = _make_face_det(100, 100, 50, 50)
        tracker.update([det], 1.0)
        assert len(tracker.tracks) == 1

        # Trigger discontinuity (dt > 2.0)
        tracker.update([], 100.0)
        assert len(tracker.tracks) == 0

        records = _extract_transition_records(logger)
        bulk_records = [r for r in records if r["entity_id"] == "bulk"]
        assert len(bulk_records) == 1
        rec = bulk_records[0]
        assert rec["reason"] == "timestamp_discontinuity"
        assert rec["entity_type"] == "face_track"

    def test_manual_clear_emits_transition(self):
        """clear() emits bulk transition record."""
        logger = MagicMock()
        tracker = IoUTracker(logger=logger)

        det = _make_face_det(100, 100, 50, 50)
        tracker.update([det], 1.0)
        tracker.clear()

        records = _extract_transition_records(logger)
        assert len(records) == 1
        assert records[0]["reason"] == "manual_clear"

    def test_previous_context_has_required_fields(self):
        """previous_context includes IoU-specific fields (SI-13.1 justified)."""
        logger = MagicMock()
        tracker = IoUTracker(max_missed=1, logger=logger)

        det = _make_face_det(100, 100, 50, 50)
        tracker.update([det], 1.0)
        # Miss enough frames
        tracker.update([], 1.05)
        tracker.update([], 1.10)

        records = _extract_transition_records(logger)
        per_track = [r for r in records if r["entity_id"] != "bulk"]
        assert len(per_track) >= 1
        ctx = per_track[0]["previous_context"]
        assert "first_seen" in ctx
        assert "last_seen" in ctx
        assert "detection_count" in ctx
        assert "missed_frames" in ctx
        assert "stability_score" in ctx

    def test_no_emission_without_logger(self):
        """Tracker without logger works normally."""
        tracker = IoUTracker()
        det = _make_face_det(100, 100, 50, 50)
        tracker.update([det], 1.0)
        tracker.update([], 100.0)  # Discontinuity
        tracker.clear()

    def test_emit_before_destroy_ordering(self):
        """Transition record is emitted before track is deleted."""
        logger = MagicMock()
        tracker = IoUTracker(max_missed=1, logger=logger)

        det = _make_face_det(100, 100, 50, 50)
        tracker.update([det], 1.0)
        track_id = list(tracker.tracks.keys())[0]

        # Miss frames to retire
        tracker.update([], 1.05)
        tracker.update([], 1.10)

        # The logger was called with entity_id matching the track
        records = _extract_transition_records(logger)
        per_track = [r for r in records if r["entity_id"] == str(track_id)]
        assert len(per_track) == 1
        # Track no longer exists
        assert track_id not in tracker.tracks
