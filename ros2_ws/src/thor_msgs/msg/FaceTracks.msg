# Array of active face tracks, published on /perception/faces/tracks

std_msgs/Header header               # Timestamp and frame_id (camera identifier)

thor_msgs/FaceTrack[] tracks         # All active face tracks

# Pipeline metrics (inline for real-time monitoring)
float32 fps                          # Current processing FPS
uint32 total_faces_detected          # Cumulative since startup
