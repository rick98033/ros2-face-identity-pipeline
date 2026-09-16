"""Unit tests for authorization node.

Tests state machine logic including:
- State transitions (UNAUTHORIZED -> ACQUIRING -> AUTHORIZED -> SUSPENDED)
- 2-of-N consistency rule
- Confidence decay
- Ambiguity detection
- Conflict persistence gate

Run with:
Preserved from the original project; not executed during source curation.
"""

import pytest
import math
from dataclasses import dataclass, field
from typing import Optional, List
from collections import deque
from enum import IntEnum


# =============================================================================
# Replicated data structures from authorization_node.py (to avoid ROS imports)
# =============================================================================


class AuthState(IntEnum):
    """Authorization states matching AuthorizedTarget.msg."""
    UNAUTHORIZED = 0
    ACQUIRING = 1
    AUTHORIZED = 2
    SUSPENDED = 3


@dataclass
class FaceMatchEvidence:
    """Single face match observation for 2-of-N consistency check."""
    timestamp: float
    face_track_id: int
    person_track_id: int
    user_id: str
    score: float
    margin: float
    quality_ok: bool
    association_confidence: float


@dataclass
class ConflictObservation:
    """Observation of face evidence on a different track (for persistence gate)."""
    timestamp: float
    different_person_track_id: int
    user_id: str
    association_confidence: float


@dataclass
class AuthorizationState:
    """Current authorization state."""
    state: AuthState = AuthState.UNAUTHORIZED
    authorized_track_id: int = 0
    authorized_user_id: str = ""
    confidence: float = 0.0
    last_face_evidence_time: float = 0.0
    state_entry_time: float = 0.0
    acquisition_attempts: int = 0
    refresh_count: int = 0
    reason_code: int = 0
    reason_text: str = ""
    track_state: int = 0
    track_match_margin: float = 0.0
    track_last_update_time: float = 0.0


# =============================================================================
# Mock structures for testing without ROS messages
# =============================================================================


@dataclass
class MockPersonTrack:
    """Mock PersonTrack message for testing."""
    track_id: int
    state: int  # 0=TENTATIVE, 1=CONFIRMED, 2=OCCLUDED, 3=LOST
    match_margin: float
    stamp_sec: float = 0.0


@dataclass
class MockFaceTrack:
    """Mock FaceTrack message for testing."""
    track_id: int
    associated_person_track_id: int
    association_confidence: float
    quality_ok: bool
    last_seen_sec: float = 0.0


@dataclass
class MockIdentityCandidate:
    """Mock FaceIdentityCandidate message for testing."""
    track_id: int  # Face track ID
    user_id: str
    score: float
    margin: float


# =============================================================================
# Helper class that extracts testable logic from AuthorizationNode
# =============================================================================


class AuthorizationLogic:
    """Testable authorization logic without ROS dependencies."""

    def __init__(
        self,
        acquisition_window_sec: float = 0.5,
        acquisition_min_consistent: int = 2,
        min_face_match_score: float = 0.5,
        min_face_match_margin: float = 0.08,
        min_association_confidence: float = 0.3,
        initial_confidence: float = 0.9,
        decay_half_life_sec: float = 60.0,
        min_confidence_threshold: float = 0.3,
        track_ambiguity_threshold: float = 0.1,
        conflict_persistence_count: int = 2,
        conflict_persistence_window_sec: float = 1.0,
        suspended_timeout_sec: float = 30.0,
    ):
        self.acquisition_window_sec = acquisition_window_sec
        self.acquisition_min_consistent = acquisition_min_consistent
        self.min_face_match_score = min_face_match_score
        self.min_face_match_margin = min_face_match_margin
        self.min_association_confidence = min_association_confidence
        self.initial_confidence = initial_confidence
        self.decay_half_life_sec = decay_half_life_sec
        self.min_confidence_threshold = min_confidence_threshold
        self.track_ambiguity_threshold = track_ambiguity_threshold
        self.conflict_persistence_count = conflict_persistence_count
        self.conflict_persistence_window_sec = conflict_persistence_window_sec
        self.suspended_timeout_sec = suspended_timeout_sec

        self.state = AuthorizationState()
        self.face_match_buffer: deque = deque(maxlen=20)
        self.conflict_buffer: deque = deque(maxlen=10)

    def start_acquisition(self, now: float):
        """Start acquisition (called when RequestFollow received)."""
        self.state.state = AuthState.ACQUIRING
        self.state.state_entry_time = now
        self.face_match_buffer.clear()

    def add_evidence(self, evidence: FaceMatchEvidence, now: float) -> bool:
        """Add evidence and check for state transitions. Returns True if AUTHORIZED."""
        if not self._evidence_meets_threshold(evidence):
            return False

        self.face_match_buffer.append(evidence)

        if self.state.state == AuthState.ACQUIRING:
            result = self._check_2_of_n_consistency(now)
            if result is not None:
                person_track_id, user_id, avg_score = result
                self._transition_to_authorized(person_track_id, user_id, avg_score, now)
                return True

        return False

    def _evidence_meets_threshold(self, evidence: FaceMatchEvidence) -> bool:
        """Check if evidence meets quality and score thresholds."""
        if not evidence.quality_ok:
            return False
        if evidence.score < self.min_face_match_score:
            return False
        if evidence.margin < self.min_face_match_margin:
            return False
        if evidence.association_confidence < self.min_association_confidence:
            return False
        return True

    def _check_2_of_n_consistency(self, now: float, user_filter: Optional[str] = None):
        """Check for 2 consistent matches within window."""
        window_start = now - self.acquisition_window_sec

        recent = [e for e in self.face_match_buffer if e.timestamp >= window_start]

        if len(recent) < self.acquisition_min_consistent:
            return None

        # Group by (person_track_id, user_id)
        groups = {}
        for e in recent:
            if user_filter is not None and e.user_id != user_filter:
                continue
            key = (e.person_track_id, e.user_id)
            if key not in groups:
                groups[key] = []
            groups[key].append(e)

        # Find groups with enough consistent matches
        valid_groups = [
            (key, evs) for key, evs in groups.items()
            if len(evs) >= self.acquisition_min_consistent
        ]

        if not valid_groups:
            return None

        # Check for multi-user ambiguity
        users = set(key[1] for key, _ in valid_groups)
        if len(users) > 1:
            return None  # AMBIGUOUS_USERS

        best_key, best_evs = max(
            valid_groups,
            key=lambda x: (len(x[1]), sum(e.score for e in x[1]) / len(x[1]))
        )

        avg_score = sum(e.score for e in best_evs) / len(best_evs)
        return (best_key[0], best_key[1], avg_score)

    def _transition_to_authorized(
        self,
        person_track_id: int,
        user_id: str,
        confidence: float,
        now: float,
    ):
        """Transition to AUTHORIZED state."""
        self.state.state = AuthState.AUTHORIZED
        self.state.authorized_track_id = person_track_id
        self.state.authorized_user_id = user_id
        self.state.confidence = min(confidence, self.initial_confidence)
        self.state.last_face_evidence_time = now
        self.state.state_entry_time = now
        self.face_match_buffer.clear()
        self.conflict_buffer.clear()

    def update_confidence_decay(self, now: float):
        """Apply confidence decay based on time since last face evidence."""
        if self.state.state != AuthState.AUTHORIZED:
            return

        if self.state.last_face_evidence_time <= 0:
            return

        elapsed = now - self.state.last_face_evidence_time
        if elapsed <= 0:
            return

        decay_factor = math.pow(0.5, elapsed / self.decay_half_life_sec)
        self.state.confidence = self.initial_confidence * decay_factor

        if self.state.confidence < self.min_confidence_threshold:
            self._transition_to_suspended("CONFIDENCE_DECAY", now)

    def check_track_triggers(self, track: MockPersonTrack, now: float):
        """Check for triggers that should move AUTHORIZED -> SUSPENDED."""
        if self.state.state != AuthState.AUTHORIZED:
            return

        # Track LOST
        if track.state == 3:  # STATE_LOST
            self._transition_to_suspended("TARGET_LOST", now)
            return

        # Track ambiguous
        if track.match_margin < self.track_ambiguity_threshold:
            self._transition_to_suspended("TRACK_AMBIGUOUS", now)
            return

    def record_conflict(self, evidence: FaceMatchEvidence, now: float):
        """Record conflict observation (authorized user on different track)."""
        obs = ConflictObservation(
            timestamp=now,
            different_person_track_id=evidence.person_track_id,
            user_id=evidence.user_id,
            association_confidence=evidence.association_confidence,
        )
        self.conflict_buffer.append(obs)
        self._check_conflict_persistence(now)

    def _check_conflict_persistence(self, now: float):
        """Check if conflicts meet persistence gate."""
        window_start = now - self.conflict_persistence_window_sec

        recent = [o for o in self.conflict_buffer if o.timestamp >= window_start]

        if len(recent) >= self.conflict_persistence_count:
            self._transition_to_suspended("CONFLICT_PERSISTED", now)

    def _transition_to_suspended(self, reason: str, now: float):
        """Transition to SUSPENDED state."""
        self.state.state = AuthState.SUSPENDED
        self.state.state_entry_time = now
        self.state.reason_text = reason
        self.face_match_buffer.clear()
        self.conflict_buffer.clear()

    def transition_to_unauthorized(self, reason: str, now: float):
        """Transition to UNAUTHORIZED state."""
        self.state.state = AuthState.UNAUTHORIZED
        self.state.authorized_track_id = 0
        self.state.authorized_user_id = ""
        self.state.confidence = 0.0
        self.state.last_face_evidence_time = 0.0
        self.state.state_entry_time = now
        self.state.reason_text = reason
        self.face_match_buffer.clear()
        self.conflict_buffer.clear()


# =============================================================================
# Tests for state transitions
# =============================================================================


class TestStateTransitions:
    """Test basic state machine transitions."""

    def test_initial_state_is_unauthorized(self):
        """Node should start in UNAUTHORIZED state."""
        logic = AuthorizationLogic()
        assert logic.state.state == AuthState.UNAUTHORIZED

    def test_start_acquisition(self):
        """RequestFollow should transition to ACQUIRING."""
        logic = AuthorizationLogic()
        logic.start_acquisition(now=1.0)
        assert logic.state.state == AuthState.ACQUIRING
        assert logic.state.state_entry_time == 1.0

    def test_acquisition_to_authorized_with_2_of_n(self):
        """Two consistent matches should transition to AUTHORIZED."""
        logic = AuthorizationLogic()
        logic.start_acquisition(now=1.0)

        # First evidence
        evidence1 = FaceMatchEvidence(
            timestamp=1.1,
            face_track_id=100,
            person_track_id=1,
            user_id="alice",
            score=0.8,
            margin=0.15,
            quality_ok=True,
            association_confidence=0.7,
        )
        result1 = logic.add_evidence(evidence1, now=1.1)
        assert not result1  # Not enough yet
        assert logic.state.state == AuthState.ACQUIRING

        # Second evidence (consistent with first)
        evidence2 = FaceMatchEvidence(
            timestamp=1.2,
            face_track_id=100,
            person_track_id=1,
            user_id="alice",
            score=0.85,
            margin=0.12,
            quality_ok=True,
            association_confidence=0.8,
        )
        result2 = logic.add_evidence(evidence2, now=1.2)
        assert result2  # Should trigger AUTHORIZED
        assert logic.state.state == AuthState.AUTHORIZED
        assert logic.state.authorized_track_id == 1
        assert logic.state.authorized_user_id == "alice"

    def test_acquisition_inconsistent_person_track(self):
        """Matches on different person tracks should not trigger AUTHORIZED."""
        logic = AuthorizationLogic()
        logic.start_acquisition(now=1.0)

        # Evidence on track 1
        evidence1 = FaceMatchEvidence(
            timestamp=1.1,
            face_track_id=100,
            person_track_id=1,
            user_id="alice",
            score=0.8,
            margin=0.15,
            quality_ok=True,
            association_confidence=0.7,
        )
        logic.add_evidence(evidence1, now=1.1)

        # Evidence on track 2 (different person track)
        evidence2 = FaceMatchEvidence(
            timestamp=1.2,
            face_track_id=101,
            person_track_id=2,
            user_id="alice",
            score=0.85,
            margin=0.12,
            quality_ok=True,
            association_confidence=0.8,
        )
        result = logic.add_evidence(evidence2, now=1.2)
        assert not result
        assert logic.state.state == AuthState.ACQUIRING

    def test_acquisition_first_user_with_2_of_n_wins(self):
        """First user to reach 2-of-N consistency wins authorization.

        This is the expected behavior: we don't wait for potential competing
        users. Once one user has 2 consistent matches, they are authorized.
        """
        logic = AuthorizationLogic(acquisition_window_sec=0.5)
        logic.start_acquisition(now=1.0)

        # Alice gets first match
        evidence_alice_1 = FaceMatchEvidence(
            timestamp=1.1,
            face_track_id=100,
            person_track_id=1,
            user_id="alice",
            score=0.8,
            margin=0.15,
            quality_ok=True,
            association_confidence=0.7,
        )
        logic.add_evidence(evidence_alice_1, now=1.1)
        assert logic.state.state == AuthState.ACQUIRING

        # Bob gets first match
        evidence_bob_1 = FaceMatchEvidence(
            timestamp=1.15,
            face_track_id=101,
            person_track_id=1,
            user_id="bob",
            score=0.82,
            margin=0.14,
            quality_ok=True,
            association_confidence=0.75,
        )
        logic.add_evidence(evidence_bob_1, now=1.15)
        assert logic.state.state == AuthState.ACQUIRING

        # Bob gets second match - should trigger AUTHORIZED for bob
        evidence_bob_2 = FaceMatchEvidence(
            timestamp=1.2,
            face_track_id=101,
            person_track_id=1,
            user_id="bob",
            score=0.83,
            margin=0.13,
            quality_ok=True,
            association_confidence=0.76,
        )
        result = logic.add_evidence(evidence_bob_2, now=1.2)

        # Bob reaches 2-of-N first, so gets authorized
        assert result
        assert logic.state.state == AuthState.AUTHORIZED
        assert logic.state.authorized_user_id == "bob"

    def test_acquisition_multi_user_ambiguity_detected(self):
        """When both users have 2-of-N at check time, should be ambiguous.

        This tests the case where we manually add evidence to buffer
        (simulating batch processing) to have both users with 2+ matches.
        """
        logic = AuthorizationLogic(acquisition_window_sec=0.5)
        logic.start_acquisition(now=1.0)

        # Manually add evidence to buffer without triggering intermediate checks
        # (simulating a scenario where multiple observations arrive simultaneously)
        evidences = [
            FaceMatchEvidence(
                timestamp=1.1, face_track_id=100, person_track_id=1,
                user_id="alice", score=0.8, margin=0.15,
                quality_ok=True, association_confidence=0.7,
            ),
            FaceMatchEvidence(
                timestamp=1.15, face_track_id=100, person_track_id=1,
                user_id="alice", score=0.81, margin=0.14,
                quality_ok=True, association_confidence=0.72,
            ),
            FaceMatchEvidence(
                timestamp=1.2, face_track_id=101, person_track_id=1,
                user_id="bob", score=0.82, margin=0.14,
                quality_ok=True, association_confidence=0.75,
            ),
            FaceMatchEvidence(
                timestamp=1.25, face_track_id=101, person_track_id=1,
                user_id="bob", score=0.83, margin=0.13,
                quality_ok=True, association_confidence=0.76,
            ),
        ]

        # Add all evidence to buffer directly
        for e in evidences:
            logic.face_match_buffer.append(e)

        # Now check 2-of-N - should return None due to multi-user ambiguity
        result = logic._check_2_of_n_consistency(now=1.25)
        assert result is None  # Ambiguous: both users have 2+ matches


class TestEvidenceGating:
    """Test evidence quality/threshold gating."""

    def test_low_score_rejected(self):
        """Evidence with score below threshold should be rejected."""
        logic = AuthorizationLogic(min_face_match_score=0.5)
        logic.start_acquisition(now=1.0)

        evidence = FaceMatchEvidence(
            timestamp=1.1,
            face_track_id=100,
            person_track_id=1,
            user_id="alice",
            score=0.4,  # Below threshold
            margin=0.15,
            quality_ok=True,
            association_confidence=0.7,
        )
        result = logic.add_evidence(evidence, now=1.1)
        assert not result
        assert len(logic.face_match_buffer) == 0

    def test_low_margin_rejected(self):
        """Evidence with margin below threshold should be rejected."""
        logic = AuthorizationLogic(min_face_match_margin=0.08)
        logic.start_acquisition(now=1.0)

        evidence = FaceMatchEvidence(
            timestamp=1.1,
            face_track_id=100,
            person_track_id=1,
            user_id="alice",
            score=0.8,
            margin=0.05,  # Below threshold
            quality_ok=True,
            association_confidence=0.7,
        )
        result = logic.add_evidence(evidence, now=1.1)
        assert not result
        assert len(logic.face_match_buffer) == 0

    def test_quality_not_ok_rejected(self):
        """Evidence with quality_ok=False should be rejected."""
        logic = AuthorizationLogic()
        logic.start_acquisition(now=1.0)

        evidence = FaceMatchEvidence(
            timestamp=1.1,
            face_track_id=100,
            person_track_id=1,
            user_id="alice",
            score=0.8,
            margin=0.15,
            quality_ok=False,  # Failed quality gate
            association_confidence=0.7,
        )
        result = logic.add_evidence(evidence, now=1.1)
        assert not result
        assert len(logic.face_match_buffer) == 0

    def test_low_association_confidence_rejected(self):
        """Evidence with low association confidence should be rejected."""
        logic = AuthorizationLogic(min_association_confidence=0.3)
        logic.start_acquisition(now=1.0)

        evidence = FaceMatchEvidence(
            timestamp=1.1,
            face_track_id=100,
            person_track_id=1,
            user_id="alice",
            score=0.8,
            margin=0.15,
            quality_ok=True,
            association_confidence=0.2,  # Below threshold
        )
        result = logic.add_evidence(evidence, now=1.1)
        assert not result
        assert len(logic.face_match_buffer) == 0


class TestConfidenceDecay:
    """Test confidence decay model."""

    def test_no_decay_when_unauthorized(self):
        """Confidence should not decay when UNAUTHORIZED."""
        logic = AuthorizationLogic()
        logic.update_confidence_decay(now=100.0)
        assert logic.state.state == AuthState.UNAUTHORIZED

    def test_decay_follows_half_life(self):
        """Confidence should halve every half_life_sec."""
        logic = AuthorizationLogic(
            decay_half_life_sec=60.0,
            initial_confidence=0.9,
            min_confidence_threshold=0.1,
        )

        # Manually set AUTHORIZED state with face evidence at t=1.0
        logic.state.state = AuthState.AUTHORIZED
        logic.state.confidence = 0.9
        logic.state.last_face_evidence_time = 1.0  # Face evidence at t=1.0

        # After 60 seconds from evidence (t=61), confidence should be ~0.45
        logic.update_confidence_decay(now=61.0)
        assert 0.44 < logic.state.confidence < 0.46

        # After 120 seconds from evidence (t=121), confidence should be ~0.225
        logic.state.confidence = 0.9  # Reset
        logic.state.last_face_evidence_time = 1.0  # Reset
        logic.update_confidence_decay(now=121.0)
        assert 0.22 < logic.state.confidence < 0.24

    def test_decay_triggers_suspended(self):
        """Confidence below threshold should trigger SUSPENDED."""
        logic = AuthorizationLogic(
            decay_half_life_sec=60.0,
            initial_confidence=0.9,
            min_confidence_threshold=0.3,
        )

        logic.state.state = AuthState.AUTHORIZED
        logic.state.confidence = 0.9
        logic.state.last_face_evidence_time = 1.0  # Face evidence at t=1.0
        logic.state.authorized_track_id = 1

        # After ~95 seconds from evidence, confidence drops below 0.3
        # 0.9 * 0.5^(95/60) = 0.9 * 0.5^1.58 = 0.9 * 0.334 = 0.30
        logic.update_confidence_decay(now=101.0)  # 100 seconds after evidence
        assert logic.state.state == AuthState.SUSPENDED


class TestAmbiguityDetection:
    """Test ambiguity triggers for SUSPENDED."""

    def test_track_lost_triggers_suspended(self):
        """Track entering LOST state should trigger SUSPENDED."""
        logic = AuthorizationLogic()

        # Set up AUTHORIZED state
        logic.state.state = AuthState.AUTHORIZED
        logic.state.authorized_track_id = 1

        # Track becomes LOST
        track = MockPersonTrack(track_id=1, state=3, match_margin=1.0)  # STATE_LOST=3
        logic.check_track_triggers(track, now=1.0)

        assert logic.state.state == AuthState.SUSPENDED
        assert "LOST" in logic.state.reason_text

    def test_track_ambiguous_triggers_suspended(self):
        """Track with low match_margin should trigger SUSPENDED."""
        logic = AuthorizationLogic(track_ambiguity_threshold=0.1)

        # Set up AUTHORIZED state
        logic.state.state = AuthState.AUTHORIZED
        logic.state.authorized_track_id = 1

        # Track becomes ambiguous
        track = MockPersonTrack(track_id=1, state=1, match_margin=0.05)  # Below threshold
        logic.check_track_triggers(track, now=1.0)

        assert logic.state.state == AuthState.SUSPENDED
        assert "AMBIGUOUS" in logic.state.reason_text

    def test_high_margin_no_suspend(self):
        """Track with high match_margin should not trigger SUSPENDED."""
        logic = AuthorizationLogic(track_ambiguity_threshold=0.1)

        # Set up AUTHORIZED state
        logic.state.state = AuthState.AUTHORIZED
        logic.state.authorized_track_id = 1

        # Track is clear (high margin)
        track = MockPersonTrack(track_id=1, state=1, match_margin=0.5)  # Above threshold
        logic.check_track_triggers(track, now=1.0)

        assert logic.state.state == AuthState.AUTHORIZED


class TestConflictPersistence:
    """Test conflict persistence gate (K of M rule)."""

    def test_single_conflict_no_suspend(self):
        """Single conflict observation should not trigger SUSPENDED."""
        logic = AuthorizationLogic(
            conflict_persistence_count=2,
            conflict_persistence_window_sec=1.0,
        )

        logic.state.state = AuthState.AUTHORIZED
        logic.state.authorized_track_id = 1
        logic.state.authorized_user_id = "alice"

        # Single conflict
        evidence = FaceMatchEvidence(
            timestamp=1.0,
            face_track_id=200,
            person_track_id=2,  # Different track
            user_id="alice",
            score=0.8,
            margin=0.15,
            quality_ok=True,
            association_confidence=0.7,
        )
        logic.record_conflict(evidence, now=1.0)

        assert logic.state.state == AuthState.AUTHORIZED

    def test_two_conflicts_triggers_suspended(self):
        """Two conflicts within window should trigger SUSPENDED."""
        logic = AuthorizationLogic(
            conflict_persistence_count=2,
            conflict_persistence_window_sec=1.0,
        )

        logic.state.state = AuthState.AUTHORIZED
        logic.state.authorized_track_id = 1
        logic.state.authorized_user_id = "alice"

        # First conflict
        evidence1 = FaceMatchEvidence(
            timestamp=1.0,
            face_track_id=200,
            person_track_id=2,
            user_id="alice",
            score=0.8,
            margin=0.15,
            quality_ok=True,
            association_confidence=0.7,
        )
        logic.record_conflict(evidence1, now=1.0)
        assert logic.state.state == AuthState.AUTHORIZED

        # Second conflict
        evidence2 = FaceMatchEvidence(
            timestamp=1.3,
            face_track_id=200,
            person_track_id=2,
            user_id="alice",
            score=0.82,
            margin=0.14,
            quality_ok=True,
            association_confidence=0.75,
        )
        logic.record_conflict(evidence2, now=1.3)

        assert logic.state.state == AuthState.SUSPENDED

    def test_conflicts_outside_window_no_suspend(self):
        """Conflicts outside window should not trigger SUSPENDED."""
        logic = AuthorizationLogic(
            conflict_persistence_count=2,
            conflict_persistence_window_sec=1.0,
        )

        logic.state.state = AuthState.AUTHORIZED
        logic.state.authorized_track_id = 1
        logic.state.authorized_user_id = "alice"

        # First conflict at t=1.0
        evidence1 = FaceMatchEvidence(
            timestamp=1.0,
            face_track_id=200,
            person_track_id=2,
            user_id="alice",
            score=0.8,
            margin=0.15,
            quality_ok=True,
            association_confidence=0.7,
        )
        logic.record_conflict(evidence1, now=1.0)

        # Second conflict at t=3.0 (outside 1s window)
        evidence2 = FaceMatchEvidence(
            timestamp=3.0,
            face_track_id=200,
            person_track_id=2,
            user_id="alice",
            score=0.82,
            margin=0.14,
            quality_ok=True,
            association_confidence=0.75,
        )
        logic.record_conflict(evidence2, now=3.0)

        # Should still be AUTHORIZED (conflicts not in same window)
        assert logic.state.state == AuthState.AUTHORIZED


class TestSuspendedTimeout:
    """Test SUSPENDED state timeout."""

    def test_suspended_timeout_to_unauthorized(self):
        """SUSPENDED should timeout to UNAUTHORIZED."""
        logic = AuthorizationLogic(suspended_timeout_sec=30.0)

        logic.state.state = AuthState.SUSPENDED
        logic.state.state_entry_time = 0.0

        # Simulate timeout check at t=31.0
        elapsed = 31.0 - logic.state.state_entry_time
        if elapsed >= logic.suspended_timeout_sec:
            logic.transition_to_unauthorized("TIMEOUT", now=31.0)

        assert logic.state.state == AuthState.UNAUTHORIZED


class TestReacquisition:
    """Test reacquisition from SUSPENDED state."""

    def test_reacquisition_same_track(self):
        """Reacquisition on same track should work."""
        logic = AuthorizationLogic()

        # Set up SUSPENDED state
        logic.state.state = AuthState.SUSPENDED
        logic.state.authorized_track_id = 1
        logic.state.authorized_user_id = "alice"
        logic.state.state_entry_time = 0.0

        # Clear buffer and attempt reacquisition
        logic.face_match_buffer.clear()

        # Two consistent matches (same user, same track)
        for i in range(2):
            evidence = FaceMatchEvidence(
                timestamp=1.0 + i * 0.1,
                face_track_id=100,
                person_track_id=1,  # Same track
                user_id="alice",  # Same user
                score=0.8,
                margin=0.15,
                quality_ok=True,
                association_confidence=0.7,
            )
            logic.face_match_buffer.append(evidence)

        # Check 2-of-N manually (since state is SUSPENDED, add_evidence routing differs)
        result = logic._check_2_of_n_consistency(now=1.2, user_filter="alice")
        assert result is not None
        assert result[0] == 1  # Same track
        assert result[1] == "alice"

    def test_reacquisition_different_track_allowed(self):
        """Reacquisition on different track should be allowed (old track may be LOST)."""
        logic = AuthorizationLogic()

        # Set up SUSPENDED state (old track was 1)
        logic.state.state = AuthState.SUSPENDED
        logic.state.authorized_track_id = 1
        logic.state.authorized_user_id = "alice"
        logic.state.state_entry_time = 0.0

        logic.face_match_buffer.clear()

        # Two consistent matches (same user, different track - old track is LOST)
        for i in range(2):
            evidence = FaceMatchEvidence(
                timestamp=1.0 + i * 0.1,
                face_track_id=100,
                person_track_id=2,  # Different track (new track after LOST)
                user_id="alice",  # Same user
                score=0.8,
                margin=0.15,
                quality_ok=True,
                association_confidence=0.7,
            )
            logic.face_match_buffer.append(evidence)

        # Check 2-of-N
        result = logic._check_2_of_n_consistency(now=1.2, user_filter="alice")
        assert result is not None
        assert result[0] == 2  # New track
        assert result[1] == "alice"

    def test_reacquisition_wrong_user_rejected(self):
        """Reacquisition with different user should be rejected."""
        logic = AuthorizationLogic()

        # Set up SUSPENDED state
        logic.state.state = AuthState.SUSPENDED
        logic.state.authorized_track_id = 1
        logic.state.authorized_user_id = "alice"

        logic.face_match_buffer.clear()

        # Two consistent matches from bob (wrong user)
        for i in range(2):
            evidence = FaceMatchEvidence(
                timestamp=1.0 + i * 0.1,
                face_track_id=100,
                person_track_id=1,
                user_id="bob",  # Wrong user
                score=0.8,
                margin=0.15,
                quality_ok=True,
                association_confidence=0.7,
            )
            logic.face_match_buffer.append(evidence)

        # Check 2-of-N with user filter
        result = logic._check_2_of_n_consistency(now=1.2, user_filter="alice")
        assert result is None  # Should not match


class TestWindowTiming:
    """Test acquisition window timing."""

    def test_matches_outside_window_not_counted(self):
        """Matches outside acquisition window should not count."""
        logic = AuthorizationLogic(acquisition_window_sec=0.5)
        logic.start_acquisition(now=1.0)

        # First match at t=1.1
        evidence1 = FaceMatchEvidence(
            timestamp=1.1,
            face_track_id=100,
            person_track_id=1,
            user_id="alice",
            score=0.8,
            margin=0.15,
            quality_ok=True,
            association_confidence=0.7,
        )
        logic.add_evidence(evidence1, now=1.1)

        # Second match at t=2.0 (0.9s later, outside 0.5s window)
        evidence2 = FaceMatchEvidence(
            timestamp=2.0,
            face_track_id=100,
            person_track_id=1,
            user_id="alice",
            score=0.85,
            margin=0.12,
            quality_ok=True,
            association_confidence=0.8,
        )
        result = logic.add_evidence(evidence2, now=2.0)

        # Should not trigger AUTHORIZED (first match is outside window)
        assert not result
        assert logic.state.state == AuthState.ACQUIRING

    def test_matches_inside_window_counted(self):
        """Matches inside acquisition window should count."""
        logic = AuthorizationLogic(acquisition_window_sec=0.5)
        logic.start_acquisition(now=1.0)

        # Two matches 0.3s apart (inside 0.5s window)
        for i in range(2):
            evidence = FaceMatchEvidence(
                timestamp=1.1 + i * 0.3,
                face_track_id=100,
                person_track_id=1,
                user_id="alice",
                score=0.8,
                margin=0.15,
                quality_ok=True,
                association_confidence=0.7,
            )
            result = logic.add_evidence(evidence, now=1.1 + i * 0.3)

        assert result  # Second match should trigger AUTHORIZED
        assert logic.state.state == AuthState.AUTHORIZED
