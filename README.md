# ROS 2 Face Identity Pipeline

An archived ROS 2 reference pipeline for turning camera frames into a bounded
authorized-target signal:

```text
RTSP camera
    |
    v
single camera ingest
    +--------------------------+
    |                          |
    v                          v
PeopleNet                 YuNet
person detections         face detections + landmarks
    |                          |
    v                          v
person tracking           face tracking
    +-----------+--------------+
                v
        face-to-person association
                |
                v
        AuraFace embeddings
                |
                v
   2-of-N authorization state machine
                |
                v
          AuthorizedTarget
```

The repository was extracted from a functioning but shelved humanoid-assistance
project. It is published as source and design reference material, not as a
supported biometric product.

## Design points

- Decode the camera stream once and distribute frames through ROS topics.
- Keep person tracks and face tracks distinct, then associate them by time and
  geometry.
- Require repeated, coherent evidence before authorization.
- Represent ambiguity, suspension, confidence decay, and loss explicitly.
- Publish one authoritative target state for downstream behaviors.
- Keep model adapters separate from enrollment and authorization policy.

## Included

- YuNet ONNX and TensorRT adapters with landmark decoding
- Face tracking, person tracking, and face-to-person association
- Optional PeopleNet person-detection adapter
- AuraFace ONNX and TensorRT embedding adapters
- Local enrollment-store implementation
- Authorization state machine and ROS 2 message/service contracts
- Camera ingest and state-server nodes required by the pipeline
- Model acquisition/build scripts retained as unexecuted reference material
- Focused tests retained from the original project

## Deliberately excluded

- Model weights and TensorRT engines
- Enrollment databases, embeddings, face crops, photographs, recordings, and
  other biometric data
- The original repository history
- Site addresses, customer identifiers, credentials, and runtime logs
- Downstream motion control

The companion `safe-follow-me-reference` repository consumes the
`AuthorizedTarget` output but is intentionally not part of this perception
pipeline.

## Repository layout

- `ros2_ws/src/thor_perception/`: curated perception ROS 2 package
- `ros2_ws/src/thor_msgs/`: message and service contracts used by the pipeline
- `support/`: original shared behavior and telemetry helpers required by the
  ROS nodes
- `tests/`: preserved focused tests
- `docs/`: architecture and model-specific references

Historical `thor_*` package names are retained to minimize changes to the last
working source.

The two Python packages under `support/` must be made available to the ROS
workspace before `thor_perception` is packaged. Target-specific dependencies
such as TensorRT, PyCUDA, ONNX Runtime, GStreamer, and ROS are deliberately not
installed or pinned by this archival repository.

## Model setup

Models are not redistributed. Review [MODEL_SOURCES.md](MODEL_SOURCES.md) and
the upstream terms before downloading anything. TensorRT engines are
hardware-, CUDA-, and TensorRT-version specific and must be built on the target
system.

## Verification status

No application code, tests, builds, containers, ROS nodes, downloads, model
inference, or hardware commands were run while preparing this source release.
See [STATUS.md](STATUS.md).

Extraction provenance and the path mapping are recorded in
[SOURCE_MANIFEST.md](SOURCE_MANIFEST.md).

## Privacy and appropriate use

Face embeddings and enrollment records are biometric data. The repository does
not provide consent, retention, access-control, bias-evaluation, or legal
compliance for a deployment. Review [DATA_GOVERNANCE.md](DATA_GOVERNANCE.md)
before adapting the code.

## License

Original code in this curated project is licensed under Apache-2.0. Models,
adapted configuration, and dependencies retain separate terms. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
