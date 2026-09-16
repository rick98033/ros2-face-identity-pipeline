# Shared support packages

The selected ROS nodes import a small set of watchdog, degradation, telemetry,
freshness, boundary-contract, and retention helpers from the original project.
Those helpers are retained here as two local Python packages:

- `thor_behavior`
- `thor_telemetry`

They are implementation dependencies of the archived pipeline, not separate
public APIs. The broader behavior-coordinator design and tests belong in the
companion `safe-follow-me-reference` repository and are not duplicated here.
