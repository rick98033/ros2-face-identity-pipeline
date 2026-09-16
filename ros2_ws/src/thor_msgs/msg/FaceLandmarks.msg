# YuNet 5-point facial landmarks (pixels, source frame coords)
# Canonical order: left_eye, right_eye, nose_tip, left_mouth, right_mouth
#
# LEFT/RIGHT CONVENTION (IMPORTANT):
# "Left eye" = subject's left eye (appears on VIEWER'S RIGHT for frontal face)
# For frontal faces: left_eye.x > right_eye.x in image coordinates
# ASSERTION: debug_face_dump.py must verify this with test images

# Constants at top (ROS2 convention)
uint8 YUNET_5PT=1

# Format identifier (publisher MUST set format=YUNET_5PT)
uint8 format

# Array form (preferred for iteration)
geometry_msgs/Point32[5] points    # [left_eye, right_eye, nose, left_mouth, right_mouth]
float32[5] confidence              # per-point confidence (set equal if model provides single score)

# Explicit fields (for quick access without indexing)
float32 left_eye_x
float32 left_eye_y
float32 right_eye_x
float32 right_eye_y
float32 nose_tip_x
float32 nose_tip_y
float32 left_mouth_x
float32 left_mouth_y
float32 right_mouth_x
float32 right_mouth_y
