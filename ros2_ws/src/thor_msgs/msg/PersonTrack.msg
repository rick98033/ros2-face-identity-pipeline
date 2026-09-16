# Tracked person state with lifecycle and ambiguity signals
#
# BOUNDING BOX CONVENTION:
# - Uses vision_msgs/BoundingBox2D: CENTER + SIZE format (not corner!)
# - center.position.x/y are FLOATS derived from pixel coordinates
# - Coordinates are in source frame resolution
# - Units: pixels
#
# FRAME_ID NOTE:
# - frame_id is camera_head_optical_frame (image-space) in Phase 3.1
# - Phase 4 servoing will transform to base/head frame as needed

# Track state enum
uint8 STATE_TENTATIVE=0          # New track, not yet confirmed
uint8 STATE_CONFIRMED=1          # Stable track with sufficient detections
uint8 STATE_OCCLUDED=2           # Missed detections, in retention window
uint8 STATE_LOST=3               # About to be retired (NOT published)

# Quality reason bitmask (for debugging why quality is low)
uint8 QUALITY_OK=0
uint8 QUALITY_LOW_CONF=1         # detector_confidence < threshold
uint8 QUALITY_AMBIGUOUS_MATCH=2  # match_margin < threshold
uint8 QUALITY_STALE=4            # last_update_age_ms > threshold
uint8 QUALITY_TIME_ANOMALY=8     # time_status != OK

# Core identity
uint64 track_id                  # Stable, monotonically increasing ID
builtin_interfaces/Time stamp    # Time of this estimate (from detection header)
string frame_id                  # camera_head_optical_frame (image-space)

# Bounding boxes (pixels, CENTER format)
vision_msgs/BoundingBox2D bbox           # Last observed bbox
vision_msgs/BoundingBox2D predicted_bbox # Motion-predicted bbox (small lookahead)

# Tracking state
uint32 last_update_age_ms        # Computed as (now_ros - stamp), NOT wall time
uint8 state                      # STATE_* enum
float32 quality                  # confidence * stability (0-1)
uint8 quality_reason             # Bitmask of QUALITY_* flags

# Ambiguity signal for Phase 3.3 authorization
# match_margin = second_best_cost - best_cost
# High value = unambiguous match, inf if no competitor
# Low value (<0.1) = ambiguous, should degrade quality
float32 match_margin

# Tracking details
uint32 detection_count           # Total frames detected in this track
float32 detector_confidence      # Latest detection confidence (0-1)
float32 stability_score          # EWMA of hit/miss: 0.9*prev + 0.1*(hit?1:0)

# Velocity (normalized coords: fraction of frame/sec)
# Resolution-invariant: scales correctly if camera resolution changes
# Clamp: 0.6 frame_widths/sec max (~400px/sec at 640px)
float32 vx
float32 vy

# Source info
uint32 source_width
uint32 source_height

# Bearing (approximate unless calibrated)
# Computed from HFOV parameter; prefer CameraInfo intrinsics (fx, cx) when available
# Sign convention: POSITIVE = right of center, NEGATIVE = left of center
float32 bearing_deg
