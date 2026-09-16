---
Title: Face Detection Pipeline
Type: reference
Canonical-ID: witness/perception/face-detection
Owner: witness
Scope: YuNet face detection, TensorRT engine build, IoU tracking, ROS 2 topic interface, confidence thresholds
Non-Scope: Face identification/embeddings (see auraface.md), person detection (see perception.md)
Last-Reviewed: 2026-02-01
---

# Face Detection Pipeline

YuNet face detection with IoU tracking on Thor, using TensorRT for GPU inference.

---

## Historical execution surface

The original project built a device-specific TensorRT engine and launched the
node inside its perception container. This extraction retains the engine-build
script and a curated launch file as reference material, but neither was run or
validated during publication. Review [MODEL_SOURCES.md](../MODEL_SOURCES.md)
before acquiring the model.

Conceptually, the retained launch surface is:

```bash
ros2 launch thor_perception identity_pipeline.launch.py enable_face_id:=false
```

---

## Architecture

```
thor-ros-perception container
├── PeopleNet pipeline → /perception/person_detections
└── YuNet pipeline → /perception/faces/tracks
     ├── face_detection_node (TensorRT + IoU tracker)
     └── face_state_server → GetFaceSnapshot service
          └── ROS Gateway → /api/v1/faces/snapshot
```

**Key difference from PeopleNet**: Uses TensorRT Python API (pycuda) instead of DeepStream nvinfer, with a custom Python IoU tracker instead of nvtracker.

---

## Output

- **Topic**: `/perception/faces/tracks` (thor_msgs/FaceTracks)
- **Service**: `/perception/faces/get_snapshot` (thor_msgs/GetFaceSnapshot)
- **Model**: YuNet 2023mar (MIT licensed, 228KB ONNX → 5MB TRT engine)
- **Inference**: FP16 on Thor GPU, ~315 qps, 3ms latency

### FaceTrack Fields

| Field | Description |
|-------|-------------|
| `track_id` | Persistent ID across frames |
| `bbox` | Center + size format (vision_msgs/BoundingBox2D) |
| `landmarks` | 5-point: left_eye, right_eye, nose, left_mouth, right_mouth |
| `detector_confidence` | YuNet confidence (0-1) |
| `quality_ok` | True if passes all quality gates |
| `quality_flags` | Bitmask: LOW_CONFIDENCE, TOO_SMALL, BLUR, EXTREME_POSE, TOO_NEW |
| `stability_score` | 0-1, increases with consecutive detections |
| `bearing_deg` | Horizontal angle from camera center |

---

## Tracking

IoU-based tracker with motion prediction:

- **Greedy assignment** with weighted cost (IoU + distance)
- **Constant-velocity model** for position prediction
- **Hard gates**: IoU > 0.3, distance < 15% frame diagonal, size ratio > 0.65
- **Track lifecycle**: 3 frames to confirm, 10 frames missed to retire (15 for confirmed)

Debug mode: `DEBUG_TRACKING=1 ros2 launch ...`

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_face_detection` | true | Enable face pipeline |
| `face_min_confidence` | 0.6 | Detection threshold |
| `face_min_size_px` | 40 | Minimum face dimension |
| `quality_min_confidence` | 0.7 | Quality gate threshold |
| `quality_min_size_px` | 60 | Quality gate size |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `No module named 'pycuda'` | Rebuild container or `pip3 install pycuda` |
| Engine 0 bytes | Run `build_face_engine.sh`, check for Myelin errors |
| No face tracks | Check RTSP stream, verify `enable_face_detection:=true` |
| ID churn | Adjust `tracker_iou_threshold`, check lighting/motion |

---

## Known Limitations & Future Work

### YOLO-Style Decode Fix (Jan 2026)

YuNet output decoding was corrected to use proper YOLO-style formulas. The model outputs per-stride tensors (`cls_8`, `obj_8`, `bbox_8`, `kps_8`, etc.) that require grid-relative decoding:

**Wrong (SSD-style with +0.5 offset):**
```python
cx = (col + 0.5 + tx) * stride  # causes ~16px shift at stride 32
```

**Correct (YOLO-style, no offset):**
```python
cx = (col + tx) * stride
cy = (row + ty) * stride
w = exp(tw) * stride
h = exp(th) * stride
```

**Score fusion** (per OpenCV implementation):
```python
score = sqrt(cls * obj)  # not just cls alone
```

The previous decode caused bounding boxes and landmarks to be shifted by ~16 pixels (0.5 * stride at the winning stride, typically 32). This is now fixed in `face_detection_node.py`.

**Letterboxing** remains center-based: 640x480 → 640x640 with `pad_y=80` (top and bottom).

### Profile View Limitation

YuNet is optimized for frontal faces and performs poorly on profile/side views. This is a model architecture limitation, not a bug. Landmarks on profile faces may be inaccurate or detection may fail entirely.

**Current status**: Acceptable for our use case (face ID during frontal interaction).

### Future Improvement: ROI-Based Detection

For more robust detection across pose variations, consider this architecture:

```
PeopleNet → person bboxes + face bboxes (class 2)
                              ↓
                    PeopleNet face bbox + expansion → head ROI
                              ↓
                    YuNet on ROI → 5-point landmarks
                              ↓
                    AuraFace alignment (requires landmarks)
```

**Benefits**:
- PeopleNet face detection handles profile views better than YuNet
- YuNet runs on smaller, targeted crop (face fills more of the 640x640 tensor)
- Single PeopleNet inference already produces both person and face bboxes
- Reduces false positives from background patterns (wallpaper, etc.)

**Implementation notes**:
- `PeopleNetTRT.detect_faces()` already exists (class 2 detection)
- YuNet on ROI requires adding crop offset to output coordinates
- May need different confidence thresholds: higher for full-frame, lower for ROI

---

## Safe Defaults

| Condition | Default Behavior |
|-----------|-----------------|
| TensorRT engine missing | Face detection node exits; perception continues without faces |
| Detection confidence < `face_min_confidence` (0.85) | Face dropped, not published |
| Face size < `face_min_size_px` (50) | Face ignored |
| No faces detected in frame | Empty FaceTracks published; downstream nodes handle gracefully |
| IoU tracker lost all tracks | Track IDs reset; downstream face-to-person re-associates |

## Forbidden Actions

- MUST NOT use SSD-style decode for YuNet. Use YOLO-style: `cx = (col + tx) * stride`, `score = sqrt(cls * obj)`.
- MUST NOT bypass the IoU tracker to publish raw detections as tracks.
- MUST NOT lower `face_min_confidence` below 0.7 without testing false-positive rates.
