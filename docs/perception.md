---
Title: Perception Pipeline
Type: reference
Canonical-ID: witness/perception/pipeline
Owner: witness
Scope: Camera ingest, person detection, person tracking, face detection, face identification, authorization state machine, QoS
Non-Scope: Deployment setup, diagnosis procedures, gaze control, base control, and depth integration
Last-Reviewed: 2026-02-01
---

# Perception Pipeline

Archived camera-ingestion and identity-pipeline design for an RTSP source.

See the [repository README](../README.md) for the documentation entry point and
publication status.

## Architecture: Detection & Tracking

```
X1 (x1_ros_color_rtsp)
       |
       | RTSP H.264 (rtsp://<IP>:8554/camera)
       v
+------------------------------------------------------------------+
| Thor (thor-ros-perception container)                             |
|                                                                  |
|  camera_ingest (GStreamer/NVDEC)                                 |
|       |                                                          |
|       v                                                          |
|  /sensors/camera/head/rgb/image --+-------------------------+    |
|  /sensors/camera/head/rgb/camera_info                       |    |
|  /sensors/camera/head/rgb/timing                            |    |
|       |                           |                         |    |
|       v                           v                         |    |
|  person_detector             face_detection                 |    |
|  (PeopleNet TensorRT)        (YuNet TensorRT)               |    |
|       |                           |                         |    |
|       v                           |                         |    |
|  /perception/person_detections    |                         |    |
|       |                           |                         |    |
|       v                           |                         |    |
|  person_tracker                   |                         |    |
|  (IoU tracking)                   |                         |    |
|       |                           |                         |    |
|       v                           |                         |    |
|  /tracking/person_tracks ---------+                         |    |
|                    (face-to-person association)              |    |
|                                   |                         |    |
|                                   v                         |    |
|                        /perception/faces/tracks             |    |
|                        (with associated_person_track_id)    |    |
|                                   |                         |    |
|                                   v                         |    |
|                              auraface_id                    |    |
|                                   |                         |    |
|                                   v                         |    |
|                        /perception/face_id/candidates       |    |
|                                   |                         |    |
|                                   v                         |    |
|                          authorization_node                 |    |
|                                   |                         |    |
|                                   v                         |    |
|                      /world/authorized_target               |    |
+------------------------------------------------------------------+
```

## Historical downstream integration

The following control architecture documents the original integration context.
The gaze controller, base controller, depth gateway, actuator gateway, and
motion gateway are deliberately not included in this extraction.

### Follow-Me Control

The companion `safe-follow-me-reference` repository contains the extracted
coordinator state machine and stop-escalation design.

```
Voice "follow me" --> FollowMeCoordinator (ROS Gateway daemon)
                           |
                           v
                    RequestFollow (ROS service)
                           |
                           v
/world/authorized_target --+----> gaze_controller ----> X1 Actuator (:50053)
                           |           |                  (head yaw)
/tracking/person_tracks ---+           |
                                       v
                              /gaze/head_yaw_cmd ---> base_controller
                                                           |
                              X1 Depth (:50054) ---------->|
                                (safety + range)           |
                                                           v
                                                    X1 Motion (:50052)
                                                      (cmd_vel)
```

**Key principle:** `camera_ingest` is the ONLY RTSP decoder. All perception nodes subscribe to ROS sensor topics. This eliminates duplicate decoding and enables rosbag record/replay.

**QoS convention:** All perception topics use `BEST_EFFORT` reliability. For real-time sensor data, dropping stale frames is preferable to queuing and adding latency.

Implementation: `ros2_ws/src/thor_perception/`

---

## Camera Ingestion

The `camera_ingest` node decodes the RTSP stream once and publishes standard ROS 2 sensor topics.

| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `/sensors/camera/head/rgb/image` | `sensor_msgs/Image` | ~15-25 Hz | BGR8 decoded frames |
| `/sensors/camera/head/rgb/camera_info` | `sensor_msgs/CameraInfo` | ~15-25 Hz | Intrinsics (from HFOV) |
| `/sensors/camera/head/rgb/timing` | `thor_msgs/CameraFrameTiming` | ~15-25 Hz | Latency breakdown |
| `/perception/camera_ingest/metrics` | `thor_msgs/PerceptionMetrics` | 0.2 Hz | Health, FPS, latency |

### TF Frames

```
base_link
  +-- head_link (stub, identity)
        +-- camera_head_link (physical mount)
              +-- camera_head_optical_frame (REP-103: Z-forward)
```

### Timing Message

`CameraFrameTiming` fields: `frame_seq`, `thor_frame_seq` (monotonic, survive reconnects), `rtp_timestamp_unwrapped` (64-bit, handles 13-hour wrap), `queue_wait_ms`, `pipeline_state` (CONNECTING / STREAMING / RECONNECTING), `timestamp_source` ("thor_receive"), `intrinsics_warning` ("SYNTHETIC_CALIBRATION").

---

## Person Detection

| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `/perception/person_detections` | `vision_msgs/Detection2DArray` | ~15 Hz | Person bounding boxes |
| `/perception/person_detector/metrics` | `thor_msgs/PerceptionMetrics` | 0.2 Hz | Health, FPS, latency |

- Subscribes to `/sensors/camera/head/rgb/image` (from camera_ingest)
- TensorRT FP16 inference (PeopleNet)
- Bbox in original image pixel space; `header.stamp` copied from input image
- `detection.id` = stable per-detection index for tracking correlation

---

## Person Tracking

IoU-based person tracking with stable track IDs.

- **Input:** `/perception/person_detections`
- **Output:** `/tracking/person_tracks` (thor_msgs/PersonTracks)

| State | Description |
|-------|-------------|
| TENTATIVE | New track, needs `min_hits_to_confirm` detections |
| CONFIRMED | Stable track with consistent detections |
| OCCLUDED | Lost detection but within `max_gap_confirmed_sec` |
| LOST | Exceeded gap timeout, track retired (not published) |

Key features: IoU + distance cost matrix with Hungarian assignment, constant-velocity prediction for occlusions, quality scoring (size, confidence, age), ambiguity signal (`match_margin`) for downstream authorization.

---

## Face Detection

Optional YuNet face detection with IoU tracking. See [face-detection.md](face-detection.md).

- **Topic:** `/perception/faces/tracks` (thor_msgs/FaceTracks)
- **Service:** `/perception/faces/get_snapshot`
- **Decode:** YOLO-style `cx = (col + tx) * stride` with `score = sqrt(cls * obj)`

### Face-to-Person Association

Each face track is spatially associated with a person track:

| Field | Description |
|-------|-------------|
| `associated_person_track_id` | PersonTrack.track_id (0 if no match) |
| `association_confidence` | 0.0-1.0 spatial match score |

Association algorithm: face center must be inside person bbox, area ratio 2-30% of person, vertical position in upper 60%, ambiguity returns 0 if margin < threshold. Time-coherent matching uses ring buffer (+/-150ms tolerance).

### Face Identification (AuraFace)

- **Topic:** `/perception/face_id/candidates` (thor_msgs/FaceIdentityCandidates)
- **Model:** AuraFace-v1 (212-dim embeddings)

See [auraface.md](auraface.md).

---

## Authorization

Voice-triggered authorization binds a user to a persistent person track_id.

- **Topic:** `/world/authorized_target` (thor_msgs/AuthorizedTarget, BEST_EFFORT)
- **Services:** `/authorization/request_follow`, `/authorization/cancel_follow`

| State | Description |
|-------|-------------|
| UNAUTHORIZED | No target, waiting for "Follow me" |
| ACQUIRING | Voice trigger received, acquiring face evidence |
| AUTHORIZED | Active authorization, track locked |
| SUSPENDED | Lost/ambiguous, prompting re-face |

Key features: 2-of-N acquisition rule (2 consistent matches within 0.5s), confidence decay (60s half-life, refreshes on re-sight), no silent switching (track ID locked while AUTHORIZED), ambiguity handling (SUSPENDED on LOST, low margin, or conflicting faces).

Implementation: `ros2_ws/src/thor_perception/thor_perception/nodes/authorization_node.py`

---

## Historical components not included

The original project also contained the following downstream and diagnostic
components. Their descriptions are retained only to explain system boundaries;
their implementations and deployment endpoints are not part of this repository.

### Debug Overlay

Time-coherent visualization for debugging the perception pipeline.

- **Topic:** `/perception/debug/overlay` (sensor_msgs/Image, BEST_EFFORT, ~10 Hz)
- **MJPEG server:** HTTP at `:8090` when `enable_mjpeg_server:=true`

Features: bounded timestamped buffers (+/-200ms), non-blocking worker thread with drop-under-load, STALE indicators, person bboxes (green=CONF, yellow=TENT, orange=OCCL), face bboxes (cyan) with association lines (magenta), authorized track highlight (gold, "AUTH:{user}"), depth overlay with safety zones.

---

### Gaze Controller

Head yaw tracking keeps the authorized target centered in the camera frame.

- **Node:** `gaze_controller_node`
- **Output:** `/gaze/head_yaw_cmd` (std_msgs/Float32, 10 Hz)

| State | Behavior | Exit Condition |
|-------|----------|----------------|
| **IDLE** | No commands, head in SAFE_HOLD | auth_state == AUTHORIZED (after 500ms stabilization) |
| **TRACKING** | P-control on x_error, send yaw commands | unauthorized, stale, or missing track |
| **CENTERING** | Command toward 0 deg, timeout after 5s | centered (+/-5 deg) OR timeout |

| Parameter | Value | Description |
|-----------|-------|-------------|
| `deadband_px` | 20 | No correction below this error |
| `k_p` | 0.02 | P-gain (degrees per pixel) |
| `max_yaw_rate` | 10.0 | Max yaw rate (deg/s) |
| `yaw_limit_deg` | 45.0 | Soft limit from neutral |
| `control_rate_hz` | 10.0 | Control loop rate |
| `track_stale_ms` | 300 | Go CENTERING if tracks older than this |
| `centering_timeout_s` | 5.0 | Max time to center before giving up |

**Known limitation:** X1 actuator gateway returns hardcoded head/eye angles (pymycobot reads arm joints 1-7 only, not middle joints 11-12). Gaze controller operates open-loop using internal `target_yaw` state.

Implementation not included.

---

### Base Controller

Wheelbase follow-me: rotates to follow head yaw drift and maintains follow distance using depth.

- **Input:** `/gaze/head_yaw_cmd`, `/world/authorized_target`, X1 Depth Gateway
- **Output:** X1 Motion Gateway (gRPC SetCmdVel)

### Two-Rate Architecture

- **Compute timer (5 Hz):** Read inputs, compute desired velocity, apply slew limiting
- **Keepalive timer (10 Hz):** Send SetCmdVel to satisfy deadman (TTL=300ms)

| State | Behavior | Exit Condition |
|-------|----------|----------------|
| **IDLE** | No commands, SAFE_HOLD mode | auth_state == AUTHORIZED |
| **FOLLOWING** | Active following, sending SetCmdVel | unauthorized, safety_stop, stale data, or gRPC error |
| **STOPPING** | Decelerate to zero (500ms) | velocity zeroed, then IDLE |

| Parameter | Value | Description |
|-----------|-------|-------------|
| `desired_range_m` | 1.5 | Target follow distance |
| `k_angular` | 0.3 | Angular P-gain |
| `k_linear` | 0.15 | Linear P-gain |
| `max_linear_mps` | 0.4 | Max forward speed |
| `max_angular_rps` | 0.5 | Max angular speed |
| `min_angular_rps` | 0.35 | Hardware minimum rotation speed |
| `head_yaw_deadband_deg` | 8.0 | Angular deadband |
| `range_deadband_m` | 0.2 | Range deadband |
| `cmd_ttl_ms` | 300 | Command time-to-live (deadman) |

### Staleness Thresholds

| Input | Threshold | Action |
|-------|-----------|--------|
| Head yaw command | 500 ms | Transition to STOPPING |
| Auth target | 500 ms | Transition to STOPPING |
| Depth cache | 300 ms | Use safety speed limit only |

Implementation not included.

---

### Depth Integration

Depth camera on X1 provides range-to-target and safety signals.

The original platform used a separate depth boundary for follow distance and
independent obstacle stopping. That boundary is intentionally not included.

---

## Safe Defaults

| Condition | Default Behavior |
|-----------|-----------------|
| TensorRT engine missing | Node logs error and exits; perception continues without that stage |
| RTSP stream unavailable | `camera_ingest` reconnects automatically; `pipeline_state` = RECONNECTING |
| Person detection confidence < `min_confidence` (0.5) | Detection dropped, not published |
| Face detection confidence < `face_min_confidence` (0.85) | Face dropped, not published |
| Face ID below threshold (0.5 cosine) | Identity = UNKNOWN |
| No enrolled faces | All identities = UNKNOWN; authorization cannot acquire |
| Authorization evidence timeout | Confidence decays with 60s half-life; enters SUSPENDED when below threshold |
| Track LOST during AUTHORIZED | State → SUSPENDED, prompts re-face |
| Gaze: stale track (>300ms) | State → CENTERING (return head to neutral) |
| Base: stale input (>500ms) | State → STOPPING (decelerate to zero) |
| Base: depth unavailable | Use safety speed limit only (no range-based control) |
| Base: depth valid ratio < 0.3 | Ignore depth sample, use safety speed only |

## Forbidden Actions

- MUST NOT create a second RTSP decoder. `camera_ingest` is the single source.
- MUST NOT use `RELIABLE` QoS for perception topics. All perception uses `BEST_EFFORT`.
- MUST NOT allow silent target switching while AUTHORIZED. Track ID is locked.
- MUST NOT bypass the face-to-person association spatial gates (area ratio, vertical position).
- MUST NOT send base velocity commands without an active authorized target.
- MUST NOT exceed `max_linear_mps` (0.4 m/s) or `max_angular_rps` (0.5 rad/s) for follow-me.
