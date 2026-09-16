# Aggregated state of a tracked person

uint64 track_id                      # Stable track ID (0 = untracked/unknown)
builtin_interfaces/Time first_seen   # When first detected in window
builtin_interfaces/Time last_seen    # Most recent detection
uint32 detection_count               # How many frames detected

# Latest bounding box (pixels, from vision_msgs)
vision_msgs/BoundingBox2D bbox       # center.position.x/y in pixels, size_x/size_y
float32 confidence                   # Latest confidence score

# Derived state
float32 presence_ratio               # detection_count / total_frames (frame-based, 0-1)
float32 time_visible_sec             # Total time tracked within window
bool is_stationary                   # True if median(|Δcenter_px|) < stationary_threshold over window

# Quality-aware tracking (track IDs are provisional, not identities)
uint32 age_ms                        # Milliseconds since first_seen
uint32 last_seen_ms                  # Milliseconds since last detection (staleness)
float32 stability_score              # 0-1 composite: age + bbox smoothness + detection continuity
bool is_primary_candidate            # Computed: best follow target in current frame

# Spatial awareness (bearing always available; range requires depth)
float32 bearing_deg                  # Horizontal angle relative to optical axis (+right, -left)
float32 bearing_std_deg              # Bearing uncertainty (trend-removed residual jitter)
float32 range_m                      # Distance in meters (NaN = unavailable)
float32 range_std_m                  # Range uncertainty (NaN = unavailable)
float32 range_quality                # 0-1 confidence in range (0 = no depth sensor)
