# Tracked face state with quality metrics
#
# BOUNDING BOX CONVENTION:
# - Uses vision_msgs/BoundingBox2D: CENTER + SIZE format (not corner!)
# - center.position.x/y are FLOATS derived from pixel coordinates
# - Coordinates are in SOURCE FRAME resolution (scaling from streammux undone)
# - Units: pixels

# Constants at top (quality flag bitmask values)
uint32 LOW_CONFIDENCE=1
uint32 TOO_SMALL=2
uint32 BLUR=4
uint32 EXTREME_POSE=8
uint32 TOO_NEW=16                    # Track too young (temporal, set by state server)

uint64 track_id                      # From nvtracker (treat as signal, not ground truth)
builtin_interfaces/Time first_seen
builtin_interfaces/Time last_seen
vision_msgs/BoundingBox2D bbox       # CENTER format: center.position.x/y, size_x/size_y
thor_msgs/FaceLandmarks landmarks

# Quality metrics (computed by deepstream_face_node)
float32 detector_confidence          # YuNet confidence (0-1)
float32 face_size_px                 # min(width, height) in pixels

# Quality gate results
# SPLIT: deepstream_face_node sets per-frame flags (LOW_CONF, TOO_SMALL, BLUR, EXTREME_POSE)
#        face_state_server adds temporal flags (TOO_NEW) and computes final quality_ok
bool quality_ok                      # True if passes all thresholds (set by state server)
string quality_reason                # Human-readable (for logs)
uint32 quality_flags                 # Bitmask for programmatic checks

# Tracking state (following PersonState patterns)
uint32 detection_count               # Frames detected in this track
uint32 age_ms                        # Milliseconds since first_seen
uint32 last_seen_ms                  # Milliseconds since last detection (staleness)
float32 stability_score              # 0-1 composite

# Source frame
string source_frame_id
uint32 source_width
uint32 source_height

# Spatial awareness
# bearing_deg: Horizontal angle relative to camera optical axis
# Sign convention: POSITIVE = right of center, NEGATIVE = left of center
# Assumes pinhole projection with known HFOV (default 65 deg)
# Valid only for rectified/undistorted frames
float32 bearing_deg

# Future-proofing for Phase 2 person association (optional, default 0 = unset)
uint64 associated_person_track_id    # PeopleNet track_id if associated (0 = none)
float32 association_confidence       # IoU or spatial confidence (0 = unset)
