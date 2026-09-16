# CP-004: Health status for ROS nodes (Category C).
# Published at 1 Hz to /health/<node_name>, consumed by ros-gateway.

string component           # Component catalog identifier
uint64 uptime_ms           # Milliseconds since node start (monotonic)
bool ok                    # True if node is healthy
string status              # healthy|degraded|unhealthy
string counters_json       # JSON-encoded counters dict
string last_error          # Error code string (empty if none)
string last_error_ts       # ISO 8601 timestamp (empty if none)
string ts                  # ISO 8601 timestamp of this message
