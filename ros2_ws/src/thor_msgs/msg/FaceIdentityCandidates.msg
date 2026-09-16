# Array of identity candidates, one per active track
#
# Published on /perception/face_id/candidates

std_msgs/Header header
thor_msgs/FaceIdentityCandidate[] candidates

# Metrics
uint32 tracks_processed
uint32 tracks_gated_out             # Skipped due to quality/stability/alignment
float32 avg_embed_latency_ms
