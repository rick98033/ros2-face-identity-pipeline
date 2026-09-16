# Camera frame timing for latency analysis
# Enables debugging network vs decode vs scheduling issues
#
# Published by camera_ingest node alongside sensor_msgs/Image.
# Use this to diagnose pipeline latency without modifying Image messages.

std_msgs/Header header
# header.stamp = Thor publish time (ROS clock)
# header.frame_id = camera optical frame

# Frame identity (both survive reconnects - never reset)
uint64 frame_seq              # Sender-side monotonic (from RTP unwrap)
uint64 thor_frame_seq         # Thor-side monotonic (distinguishes receiver drops)

# Sender timestamps (RTP unwrapping is mandatory)
uint32 rtp_timestamp          # Raw RTP 90kHz (for reference only)
uint64 rtp_timestamp_unwrapped # Unwrapped (handles 13-hour wrap at 90kHz)
uint64 sender_pts_ns          # GStreamer PTS from sender (nanoseconds)

# Thor-side timestamps (nanoseconds since epoch)
uint64 appsink_receive_ns     # When GStreamer delivered frame
uint64 publish_ns             # When ROS publish started

# Latency breakdown (milliseconds)
float32 network_latency_estimate_ms  # Inter-arrival vs RTP delta
float32 decode_latency_ms            # PTS to appsink receive
float32 queue_wait_ms                # appsink receive to publish
float32 total_latency_ms             # Sum of above

# Pipeline state
uint8 STATE_UNKNOWN = 0
uint8 STATE_CONNECTING = 1
uint8 STATE_STREAMING = 2
uint8 STATE_RECONNECTING = 3
uint8 pipeline_state

# Trust metadata - downstream should NOT assume synchronized clocks
string timestamp_source       # "thor_receive", "sender_pts", "ptp_sync"
string intrinsics_source      # "hfov_estimate" or "calibration_file"
string intrinsics_warning     # "SYNTHETIC_CALIBRATION" or "" (empty if calibrated)
