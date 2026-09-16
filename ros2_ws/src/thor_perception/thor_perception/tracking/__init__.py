"""Tracking module for thor_perception."""

from .iou_tracker import IoUTracker, TrackedFace, MatchDebug
from .crop_buffer import FaceCropBuffer, CropEntry, CROP_DIR
from .person_tracker import (
    PersonTracker,
    TrackedPerson,
    PersonDetection,
    TrackState,
    TimeStatus,
    QualityReason,
)
from .face_person_association import (
    PersonTrackBuffer,
    AssociationResult,
    associate_face_to_person,
)

__all__ = [
    # Face tracking
    "IoUTracker",
    "TrackedFace",
    "MatchDebug",
    "FaceCropBuffer",
    "CropEntry",
    "CROP_DIR",
    # Person tracking
    "PersonTracker",
    "TrackedPerson",
    "PersonDetection",
    "TrackState",
    "TimeStatus",
    "QualityReason",
    # Face-to-person association
    "PersonTrackBuffer",
    "AssociationResult",
    "associate_face_to_person",
]
