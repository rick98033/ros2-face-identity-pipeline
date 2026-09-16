# Source manifest

This repository is a curated, history-free extraction prepared on 2026-09-15.
It is not a mirror of the source repository.

## Source revision

- `rick98033/witness` at
  `87759bf75a3363d0de3fca19e40f9bf3face2f43`

## Curated path mapping

| Published path | Source area |
| --- | --- |
| `ros2_ws/src/thor_perception/` | selected files from `thor_ws/src/thor_perception/` |
| `ros2_ws/src/thor_msgs/` | selected face, person, health, and authorization interfaces from `thor_ws/src/thor_msgs/` |
| `support/thor_behavior/` | runtime watchdog and degradation helpers from `packages/thor_behavior/` |
| `support/thor_telemetry/` | telemetry and boundary helpers from `packages/thor_telemetry/` |
| `tests/` | selected perception, association, tracker, and authorization tests |
| `docs/` | selected perception, YuNet, and AuraFace references from `docs/reference/` |

## Curation changes

- Removed unrelated robot behaviors, motion control, cloud services, user data,
  and deployment tooling.
- Removed the original Git history rather than attempting to sanitize it.
- Excluded model weights, TensorRT engines, enrollments, embeddings, face
  images, crops, recordings, logs, and runtime databases.
- Replaced deployment addresses and private-workspace references with
  loopback/reference values or explicit out-of-scope notes.
- Reduced ROS message/service manifests to the selected pipeline's contracts.
- Added a narrow launch surface, publication metadata, biometric-governance
  warnings, model provenance, and an Apache-2.0 license.

No project code, tests, builds, containers, ROS nodes, model downloads,
inference, or hardware commands were run while creating this extraction.
