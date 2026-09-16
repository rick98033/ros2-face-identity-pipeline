---
Title: AuraFace Face Identification
Type: reference
Canonical-ID: witness/perception/auraface
Owner: witness
Scope: AuraFace-v1 embeddings, TensorRT engine build, face enrollment, identification thresholds, ROS 2 topic interface
Non-Scope: Face detection (see face-detection.md), application-level identity policy
Last-Reviewed: 2026-02-01
---

# AuraFace Face Identification

Face identification using AuraFace-v1 embeddings on top of YuNet face detection.

---

## Historical execution surface

The original project built a device-specific TensorRT engine and provisioned
enrollments with a separate private workflow. This extraction includes the
engine-build script and local enrollment-store implementation, but deliberately
does not include enrollment images, embeddings, databases, or the provisioning
CLI. Nothing in this section was run while preparing the repository.

Conceptually, the retained launch surface is:

```bash
ros2 launch thor_perception identity_pipeline.launch.py enable_face_id:=true
```

---

## Model Details

| Property | Value |
|----------|-------|
| Model | fal/AuraFace-v1 |
| Input | 192x192 BGR |
| Output | 212-dim L2-normalized |
| Inference | TensorRT FP16 (~0.26ms) |

---

## Files

| File | Description |
|------|-------------|
| `thor_perception/nodes/auraface_id_node.py` | Main embedding node |
| `thor_perception/face_id/alignment.py` | Face crop/alignment (InsightFace template) |
| `thor_perception/face_id/enrollment_store.py` | Enrollment storage, cosine matching |
| `scripts/build_auraface_engine.sh` | TensorRT engine builder |
| `thor_msgs/msg/FaceIdentityCandidate.msg` | Single identity hypothesis |
| `thor_msgs/msg/FaceIdentityCandidates.msg` | Array with metrics |

---

## Decision States

| State | Meaning |
|-------|---------|
| UNKNOWN | Score below threshold (no match) |
| AMBIGUOUS | Score ok but margin too low (close call) |
| TENTATIVE | Good match, awaiting temporal confirmation |
| CONFIRMED | Same identity stable for 1.5s |

Key design: Identity hypothesis always published (top-k + scores). Decision state controls actionability. AMBIGUOUS never confirms.

---

## Enrollment boundary

`FaceEnrollmentStore` reads and updates a local JSON enrollment store. A real
deployment must provide an independently reviewed consent, enrollment,
revocation, access-control, and deletion workflow. See
[DATA_GOVERNANCE.md](../DATA_GOVERNANCE.md).

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `face_id_threshold` | 0.5 | Min score for non-UNKNOWN |
| `face_id_margin_threshold` | 0.08 | Min gap to second-best |
| `face_id_confirm_time_sec` | 1.5 | Time for CONFIRMED |
| `face_id_embed_rate_hz` | 2.0 | Max embeddings per track/sec |
| `face_id_buffer_size` | 5 | Rolling embedding buffer |
| `face_id_min_stability` | 0.5 | Min track stability to compute |

---

## Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/perception/face_id/candidates` | FaceIdentityCandidates | Identity hypotheses |
| `/perception/camera/frame` | sensor_msgs/Image | Frame for alignment (internal) |

---

## Archived status

- No model weights, engines, enrollment records, or face media are included.
- The pipeline requires both the AuraFace and YuNet model artifacts described
  in [MODEL_SOURCES.md](../MODEL_SOURCES.md).
- Runtime defaults must be reviewed and calibrated for any new deployment.

---

## Safe Defaults

| Condition | Default Behavior |
|-----------|-----------------|
| No enrollments exist | All identities = UNKNOWN; authorization cannot acquire |
| TensorRT engine missing | Face ID node exits; perception continues without identification |
| Face alignment fails | Frame skipped; next frame retried |
| Cosine similarity below threshold (0.5) | Identity = UNKNOWN |
| Multiple candidates above threshold | Best match returned; ambiguity signal passed to authorization |

## Forbidden Actions

- MUST NOT lower `face_id_threshold` below 0.4 without testing false-accept rates.
- MUST NOT enroll faces from low-quality images (blurry, occluded, extreme angle).
- MUST NOT bypass face alignment before embedding extraction — coordinates must be corrected for crop offset.
