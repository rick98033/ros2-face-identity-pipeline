# Perception pipeline health and performance metrics
# Published by perception nodes at configurable interval
#
# All counters are monotonic (total since node startup).
# Consumers can compute deltas between publishes if needed.

std_msgs/Header header
# header.stamp = publish time (get_clock().now())
# header.frame_id = sanitized camera identifier (e.g., "camera0", "rtsp_front")
#                   DO NOT use raw RTSP URIs (may contain credentials)

# Performance metrics
float32 fps                      # Frames processed per second (over metrics window)
float32 latency_p95_ms           # 95th percentile end-to-end latency in ms
uint32 dropped_frames_total      # Monotonic: total frames dropped since startup
uint32 queue_depth               # Current input queue depth

# Model info
string model_name                # e.g., "PeopleNet"
string model_version             # e.g., "2.6"
bool tracker_enabled             # Whether object tracker is active

# Health status
# Health is computed by producer using configurable thresholds (ROS params).
# reason_code is the PRIMARY issue (single string). If multiple issues exist,
# use precedence: RTSP_DISCONNECTED > RTSP_RECONNECTING > QUEUE_OVERFLOW > HIGH_LATENCY > LOW_FPS
# Vocabulary: "", "LOW_FPS", "HIGH_LATENCY", "RTSP_RECONNECTING", "RTSP_DISCONNECTED", "QUEUE_OVERFLOW"
uint8 health
uint8 OK=0
uint8 DEGRADED=1
uint8 ERROR=2
string reason_code               # Empty if OK, else highest-precedence issue

# Connection info (for RTSP source)
uint32 reconnect_count_total     # Monotonic: total RTSP reconnections since startup
