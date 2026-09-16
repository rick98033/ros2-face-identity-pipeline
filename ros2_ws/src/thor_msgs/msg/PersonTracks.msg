# Array of active person tracks
#
# Published on /tracking/person_tracks
# LOST tracks are NOT included (removed from active list)
#
# Timestamp semantics: header.stamp comes from detection header (image timestamp)

# Time status enum (for downstream graceful degradation)
uint8 TIME_OK=0
uint8 TIME_DISCONTINUITY_DETECTED=1  # Brief anomaly, tracks preserved
uint8 TIME_FROZEN=2                  # Persistent anomaly, tracks cleared

std_msgs/Header header               # Timestamp and frame_id

thor_msgs/PersonTrack[] tracks       # All active tracks (excludes LOST)

# Pipeline metrics
float32 fps                          # Current processing FPS
uint32 total_persons_tracked         # Cumulative track count since startup
uint32 active_track_count            # Current number of active tracks

# Diagnostics (for debugging, not acceptance criteria)
uint8 time_status                    # TIME_* enum
