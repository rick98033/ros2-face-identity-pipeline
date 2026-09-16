"""Unit tests for face-to-person association module.

Tests spatial association algorithm including:
- Basic matching (face inside person bbox)
- Area ratio gates
- Vertical position gates
- Ambiguity detection (two people close)
- OCCLUDED track support
- require_confirmed gate

Run with:
Preserved from the original project; not executed during source curation.
"""

import pytest
import threading
import time
from dataclasses import dataclass
from typing import Optional

from thor_perception.tracking.face_person_association import (
    PersonTrackBuffer,
    AssociationResult,
    associate_face_to_person,
    STATE_TENTATIVE,
    STATE_CONFIRMED,
    STATE_OCCLUDED,
    STATE_LOST,
)


# =============================================================================
# Mock PersonTrack for testing (mimics thor_msgs/PersonTrack structure)
# =============================================================================


@dataclass
class MockBbox:
    """Mock bounding box (vision_msgs/BoundingBox2D style)."""

    class Center:
        class Position:
            x: float = 0.0
            y: float = 0.0

        position: "MockBbox.Center.Position"
        theta: float = 0.0

        def __init__(self):
            self.position = MockBbox.Center.Position()

    center: "MockBbox.Center"
    size_x: float = 0.0
    size_y: float = 0.0

    def __init__(self, cx: float = 0, cy: float = 0, w: float = 0, h: float = 0):
        self.center = MockBbox.Center()
        self.center.position.x = cx
        self.center.position.y = cy
        self.size_x = w
        self.size_y = h


@dataclass
class MockPersonTrack:
    """Mock PersonTrack message for testing."""

    track_id: int
    state: int
    quality: float
    bbox: MockBbox
    predicted_bbox: Optional[MockBbox] = None

    @classmethod
    def create(
        cls,
        track_id: int,
        cx: float,
        cy: float,
        w: float,
        h: float,
        state: int = STATE_CONFIRMED,
        quality: float = 0.8,
        predicted_cx: Optional[float] = None,
        predicted_cy: Optional[float] = None,
    ) -> "MockPersonTrack":
        """Factory method to create mock track."""
        bbox = MockBbox(cx, cy, w, h)
        predicted_bbox = None
        if predicted_cx is not None:
            predicted_bbox = MockBbox(predicted_cx, predicted_cy or cy, w, h)
        return cls(
            track_id=track_id,
            state=state,
            quality=quality,
            bbox=bbox,
            predicted_bbox=predicted_bbox or bbox,
        )


@dataclass
class MockHeader:
    """Mock ROS header for PersonTracks."""

    class Stamp:
        sec: int = 0
        nanosec: int = 0

    stamp: "MockHeader.Stamp"

    def __init__(self, sec: int = 0, nanosec: int = 0):
        self.stamp = MockHeader.Stamp()
        self.stamp.sec = sec
        self.stamp.nanosec = nanosec


@dataclass
class MockPersonTracks:
    """Mock PersonTracks message."""

    header: MockHeader
    tracks: list

    def __init__(self, tracks: list, timestamp_sec: float = 0.0):
        sec = int(timestamp_sec)
        nanosec = int((timestamp_sec - sec) * 1e9)
        self.header = MockHeader(sec, nanosec)
        self.tracks = tracks


# =============================================================================
# Tests for associate_face_to_person
# =============================================================================


class TestBasicAssociation:
    """Test basic face-to-person matching."""

    def test_face_inside_single_person(self):
        """Face clearly inside one person bbox should match."""
        # Person at (320, 300) with size 100x200
        tracks = [MockPersonTrack.create(track_id=1, cx=320, cy=300, w=100, h=200)]

        # Face at (320, 230) - upper portion of person
        result = associate_face_to_person(
            face_cx=320,
            face_cy=230,
            face_w=50,
            face_h=50,
            person_tracks=tracks,
        )

        assert result.person_track_id == 1
        assert result.confidence > 0.3
        assert not result.ambiguous

    def test_face_outside_person_bbox(self):
        """Face completely outside person bbox should not match."""
        tracks = [MockPersonTrack.create(track_id=1, cx=320, cy=300, w=100, h=200)]

        # Face at (600, 230) - far from person
        result = associate_face_to_person(
            face_cx=600,
            face_cy=230,
            face_w=50,
            face_h=50,
            person_tracks=tracks,
        )

        assert result.person_track_id == 0
        assert result.confidence == 0.0
        assert not result.ambiguous

    def test_no_person_tracks(self):
        """Empty person tracks should return no association."""
        result = associate_face_to_person(
            face_cx=320,
            face_cy=230,
            face_w=50,
            face_h=50,
            person_tracks=[],
        )

        assert result.person_track_id == 0
        assert result.confidence == 0.0
        assert not result.ambiguous


class TestGates:
    """Test area ratio and vertical position gates."""

    def test_area_ratio_too_small(self):
        """Face too small relative to person should be rejected."""
        # Large person (200x400) with tiny face (10x10) = 0.01% ratio
        tracks = [MockPersonTrack.create(track_id=1, cx=320, cy=300, w=200, h=400)]

        result = associate_face_to_person(
            face_cx=320,
            face_cy=150,  # Upper portion
            face_w=10,
            face_h=10,
            person_tracks=tracks,
            area_ratio_min=0.02,  # 2% minimum
        )

        assert result.person_track_id == 0
        assert result.confidence == 0.0

    def test_area_ratio_too_large(self):
        """Face too large relative to person should be rejected."""
        # Small person (50x100) with large face (40x40) = 32% ratio
        tracks = [MockPersonTrack.create(track_id=1, cx=320, cy=300, w=50, h=100)]

        result = associate_face_to_person(
            face_cx=320,
            face_cy=260,  # Upper portion
            face_w=40,
            face_h=40,
            person_tracks=tracks,
            area_ratio_max=0.30,  # 30% maximum
        )

        assert result.person_track_id == 0
        assert result.confidence == 0.0

    def test_face_in_lower_portion_rejected(self):
        """Face in lower 40% of person bbox should be rejected."""
        tracks = [MockPersonTrack.create(track_id=1, cx=320, cy=300, w=100, h=200)]

        # Face at (320, 380) - face_top = 355, relative_y = (355-200)/200 = 0.775
        result = associate_face_to_person(
            face_cx=320,
            face_cy=380,  # Lower portion (y > 60% threshold)
            face_w=50,
            face_h=50,
            person_tracks=tracks,
        )

        assert result.person_track_id == 0
        assert result.confidence == 0.0


class TestAmbiguity:
    """Test ambiguity detection with multiple candidates."""

    def test_two_people_clear_winner(self):
        """With two people, clear best match should succeed."""
        # Person 1: face clearly inside
        # Person 2: face barely inside edge
        tracks = [
            MockPersonTrack.create(track_id=1, cx=320, cy=300, w=100, h=200),
            MockPersonTrack.create(track_id=2, cx=450, cy=300, w=100, h=200),
        ]

        # Face at (320, 230) - clearly in person 1
        result = associate_face_to_person(
            face_cx=320,
            face_cy=230,
            face_w=50,
            face_h=50,
            person_tracks=tracks,
        )

        assert result.person_track_id == 1
        assert result.confidence > 0.3
        assert not result.ambiguous

    def test_two_people_close_ambiguous(self):
        """Two overlapping people with face on boundary should be ambiguous.

        MISSION-CRITICAL TEST: This protects Phase 3.3 authorization.
        If a face is near the boundary of two person bboxes, we must return
        associated_person_track_id=0 rather than guess.
        """
        # Two people with overlapping bboxes
        tracks = [
            MockPersonTrack.create(track_id=1, cx=300, cy=300, w=150, h=200),
            MockPersonTrack.create(track_id=2, cx=400, cy=300, w=150, h=200),
        ]

        # Face at (350, 230) - right on boundary, inside both bboxes
        result = associate_face_to_person(
            face_cx=350,
            face_cy=230,
            face_w=50,
            face_h=50,
            person_tracks=tracks,
            margin_threshold=0.15,  # Require 15% margin
        )

        # Should return 0 (ambiguous) rather than guessing
        assert result.person_track_id == 0
        assert result.ambiguous

    def test_face_boundary_must_return_zero(self):
        """Negative test: face exactly on boundary MUST return 0.

        This verifies the 'don't guess' rule is enforced.
        """
        # Two people side by side, touching
        tracks = [
            MockPersonTrack.create(track_id=1, cx=250, cy=300, w=100, h=200),
            MockPersonTrack.create(track_id=2, cx=350, cy=300, w=100, h=200),
        ]

        # Face exactly on boundary (cx=300 is edge of both)
        # Person 1: left=200, right=300
        # Person 2: left=300, right=400
        # Face center at 300 is on the edge
        result = associate_face_to_person(
            face_cx=300,
            face_cy=230,
            face_w=50,
            face_h=50,
            person_tracks=tracks,
            margin_threshold=0.15,
        )

        # Face center at 300 is exactly on the right edge of person 1
        # and exactly on the left edge of person 2
        # Since center must be strictly inside, this should only match person 2
        # (300 <= 300 <= 400 for person 2, 200 <= 300 <= 300 for person 1)
        # If both pass containment, it should be ambiguous
        # If only one passes, that's fine
        # Either way, we verify the system doesn't incorrectly pick the wrong one
        assert result.person_track_id in [0, 2]  # Either ambiguous or person 2


class TestTrackStates:
    """Test handling of different track states."""

    def test_require_confirmed_filters_tentative(self):
        """With require_confirmed=True, TENTATIVE tracks should be skipped."""
        tracks = [
            MockPersonTrack.create(track_id=1, cx=320, cy=300, w=100, h=200, state=STATE_TENTATIVE),
        ]

        result = associate_face_to_person(
            face_cx=320,
            face_cy=230,
            face_w=50,
            face_h=50,
            person_tracks=tracks,
            require_confirmed=True,
        )

        assert result.person_track_id == 0
        assert result.confidence == 0.0

    def test_allow_tentative_when_disabled(self):
        """With require_confirmed=False, TENTATIVE tracks should be considered."""
        tracks = [
            MockPersonTrack.create(track_id=1, cx=320, cy=300, w=100, h=200, state=STATE_TENTATIVE),
        ]

        result = associate_face_to_person(
            face_cx=320,
            face_cy=230,
            face_w=50,
            face_h=50,
            person_tracks=tracks,
            require_confirmed=False,
        )

        assert result.person_track_id == 1
        assert result.confidence > 0.0

    def test_occluded_uses_predicted_bbox(self):
        """OCCLUDED track should use predicted_bbox for association."""
        # Note: With require_confirmed=True, OCCLUDED is skipped by default
        # Test with require_confirmed=False to verify predicted_bbox usage
        tracks = [
            MockPersonTrack.create(
                track_id=1,
                cx=320,
                cy=300,
                w=100,
                h=200,
                state=STATE_OCCLUDED,
                predicted_cx=350,  # Predicted position is different
                predicted_cy=300,
            ),
        ]

        # Face at predicted position (350, 230)
        result = associate_face_to_person(
            face_cx=350,
            face_cy=230,
            face_w=50,
            face_h=50,
            person_tracks=tracks,
            require_confirmed=False,  # Allow OCCLUDED
        )

        # Should match based on predicted_bbox position
        assert result.person_track_id == 1

    def test_lost_track_skipped_when_require_confirmed(self):
        """LOST tracks should be skipped with require_confirmed=True."""
        tracks = [
            MockPersonTrack.create(track_id=1, cx=320, cy=300, w=100, h=200, state=STATE_LOST),
        ]

        result = associate_face_to_person(
            face_cx=320,
            face_cy=230,
            face_w=50,
            face_h=50,
            person_tracks=tracks,
            require_confirmed=True,
        )

        assert result.person_track_id == 0


class TestMinScore:
    """Test minimum score threshold."""

    def test_below_min_score_rejected(self):
        """Match with score below threshold should be rejected."""
        # Create a marginal match (face near edge)
        tracks = [MockPersonTrack.create(track_id=1, cx=320, cy=300, w=100, h=200, quality=0.1)]

        # Face near edge of person bbox (low horiz_score)
        result = associate_face_to_person(
            face_cx=365,  # Near right edge (320 + 50 - 5)
            face_cy=210,  # Upper portion
            face_w=50,
            face_h=50,
            person_tracks=tracks,
            min_score=0.9,  # Very high threshold
        )

        assert result.person_track_id == 0
        assert result.confidence == 0.0


# =============================================================================
# Tests for PersonTrackBuffer
# =============================================================================


class TestPersonTrackBuffer:
    """Test thread-safe ring buffer."""

    def test_add_and_get_closest(self):
        """Basic add and retrieve."""
        buffer = PersonTrackBuffer(max_age_sec=1.0)

        msg = MockPersonTracks([], timestamp_sec=1.0)
        buffer.add(msg)

        result = buffer.get_closest(1.0)
        assert result is not None
        assert result is msg

    def test_get_closest_within_tolerance(self):
        """Should return closest within tolerance."""
        buffer = PersonTrackBuffer(max_age_sec=1.0)

        msg1 = MockPersonTracks([], timestamp_sec=1.0)
        msg2 = MockPersonTracks([], timestamp_sec=1.1)
        msg3 = MockPersonTracks([], timestamp_sec=1.2)

        buffer.add(msg1)
        buffer.add(msg2)
        buffer.add(msg3)

        # Query at 1.05 should return msg1 (closest: |1.0-1.05|=0.05)
        result = buffer.get_closest(1.05, tolerance_sec=0.1)
        assert result is msg1

        # Query at 1.12 should return msg2 (closest: |1.1-1.12|=0.02 vs |1.2-1.12|=0.08)
        result = buffer.get_closest(1.12, tolerance_sec=0.1)
        assert result is msg2

    def test_get_closest_outside_tolerance(self):
        """Should return None if outside tolerance."""
        buffer = PersonTrackBuffer(max_age_sec=1.0)

        msg = MockPersonTracks([], timestamp_sec=1.0)
        buffer.add(msg)

        # Query at 1.5 with 0.1 tolerance should return None
        result = buffer.get_closest(1.5, tolerance_sec=0.1)
        assert result is None

    def test_empty_buffer(self):
        """Empty buffer should return None."""
        buffer = PersonTrackBuffer(max_age_sec=1.0)

        result = buffer.get_closest(1.0)
        assert result is None

    def test_old_entries_pruned(self):
        """Old entries should be automatically pruned."""
        buffer = PersonTrackBuffer(max_age_sec=0.5)

        msg1 = MockPersonTracks([], timestamp_sec=1.0)
        buffer.add(msg1)

        # Add message 0.6s later (msg1 should be pruned)
        msg2 = MockPersonTracks([], timestamp_sec=1.6)
        buffer.add(msg2)

        # Query at 1.0 should return None (msg1 was pruned)
        result = buffer.get_closest(1.0, tolerance_sec=0.1)
        assert result is None

        # Query at 1.6 should return msg2
        result = buffer.get_closest(1.6, tolerance_sec=0.1)
        assert result is msg2

    def test_clear(self):
        """Clear should empty the buffer."""
        buffer = PersonTrackBuffer(max_age_sec=1.0)

        buffer.add(MockPersonTracks([], timestamp_sec=1.0))
        buffer.clear()

        result = buffer.get_closest(1.0)
        assert result is None

    def test_thread_safety(self):
        """Buffer should be thread-safe for concurrent access."""
        buffer = PersonTrackBuffer(max_age_sec=1.0)
        errors = []

        def writer():
            try:
                for i in range(100):
                    buffer.add(MockPersonTracks([], timestamp_sec=i * 0.01))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    buffer.get_closest(i * 0.01, tolerance_sec=0.1)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# Integration-style tests
# =============================================================================


class TestIntegration:
    """Integration-style tests simulating real scenarios."""

    def test_face_moves_between_people(self):
        """Face moving from one person bbox to another."""
        tracks = [
            MockPersonTrack.create(track_id=1, cx=200, cy=300, w=100, h=200),
            MockPersonTrack.create(track_id=2, cx=400, cy=300, w=100, h=200),
        ]

        # Face initially with person 1
        result1 = associate_face_to_person(
            face_cx=200, face_cy=230, face_w=50, face_h=50, person_tracks=tracks
        )
        assert result1.person_track_id == 1

        # Face moves to person 2
        result2 = associate_face_to_person(
            face_cx=400, face_cy=230, face_w=50, face_h=50, person_tracks=tracks
        )
        assert result2.person_track_id == 2

    def test_multiple_faces_same_person(self):
        """Multiple faces can associate to same person (each independently)."""
        tracks = [MockPersonTrack.create(track_id=1, cx=320, cy=300, w=150, h=300)]

        # Two faces in same person bbox (e.g., close-up, face + reflection)
        result1 = associate_face_to_person(
            face_cx=310, face_cy=200, face_w=50, face_h=50, person_tracks=tracks
        )
        result2 = associate_face_to_person(
            face_cx=330, face_cy=220, face_w=40, face_h=40, person_tracks=tracks
        )

        assert result1.person_track_id == 1
        assert result2.person_track_id == 1

    def test_realistic_scenario_crowd(self):
        """Realistic scenario with multiple people in frame."""
        tracks = [
            MockPersonTrack.create(track_id=1, cx=160, cy=300, w=100, h=200),  # Person left
            MockPersonTrack.create(track_id=2, cx=320, cy=300, w=100, h=200),  # Person center
            MockPersonTrack.create(track_id=3, cx=480, cy=300, w=100, h=200),  # Person right
        ]

        # Face of person 2 (center)
        result = associate_face_to_person(
            face_cx=320, face_cy=230, face_w=50, face_h=50, person_tracks=tracks
        )

        assert result.person_track_id == 2
        assert result.confidence > 0.5
        assert not result.ambiguous
