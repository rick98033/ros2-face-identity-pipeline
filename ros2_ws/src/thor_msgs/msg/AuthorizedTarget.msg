# Authorization state for follow-me target
#
# Published on /world/authorized_target
# Subscribe pattern: Best-effort, keep-last-1 (real-time state)
#
# CRITICAL: Follow-me (Phase 4) must depend ONLY on this topic, not on detection topics
#
# HARD RULE: Never silently switch authorized_track_id while AUTHORIZED.
# Ambiguity or loss transitions to SUSPENDED first, requiring explicit re-face.

# Auth state enum
uint8 AUTH_UNAUTHORIZED=0      # No target, waiting for "Follow me"
uint8 AUTH_ACQUIRING=1         # "Follow me" received, acquiring face evidence
uint8 AUTH_AUTHORIZED=2        # Active authorization, track locked
uint8 AUTH_SUSPENDED=3         # Lost/ambiguous, prompting re-face

# Reason codes (can be combined as bitmask for diagnostics)
uint16 REASON_NONE=0
uint16 REASON_ACQUIRED_FACE_MATCH=1        # Successful face acquisition
uint16 REASON_FACE_LOW_QUALITY=2           # Face not usable for matching
uint16 REASON_FACE_NO_MATCH=4              # Face doesn't match enrolled user
uint16 REASON_AMBIGUOUS_MULTIPLE=8         # Multiple candidates, can't disambiguate
uint16 REASON_TARGET_LOST=16               # Track went to LOST state
uint16 REASON_CONFIDENCE_DECAY=32          # Confidence below threshold
uint16 REASON_TRACK_AMBIGUOUS=64           # Person tracks became indistinguishable
uint16 REASON_TIMEOUT=128                  # Acquisition or suspension timeout
uint16 REASON_REACQUIRED_FACE_MATCH=256    # Re-bound in SUSPENDED (may have new track_id)
uint16 REASON_CANCELLED=512                # Explicit cancel via CancelFollow
uint16 REASON_AMBIGUOUS_USERS=1024         # Multiple enrolled users within margin

# Core state
std_msgs/Header header
uint8 auth_state                          # AUTH_* enum

# Target identification (valid when AUTHORIZED or SUSPENDED)
uint64 authorized_track_id                # Person track_id (0 = no target)
string authorized_user_id                 # From face enrollment (e.g., "user_alice")

# Confidence and evidence (valid when AUTHORIZED)
float32 confidence                        # 0.0-1.0, decays over time without face evidence
uint32 last_face_evidence_age_ms          # Time since last successful face refresh

# Diagnostics
uint16 reason_code                        # REASON_* value (most recent transition reason)
string reason_text                        # Human-readable (for logs/UX prompts)

# Copied from authorized PersonTrack (for Phase 4 safety checks)
uint8 track_state                         # PersonTrack.STATE_* enum
float32 track_match_margin                # From PersonTrack.match_margin (higher=less ambiguous)
uint32 authorized_track_age_ms            # Age since last track update (stop if stale)

# Metrics
uint32 time_in_state_ms                   # Time since last state transition
uint32 acquisition_attempts               # Failed acquisitions since last AUTHORIZED
uint32 refresh_count                      # Successful face refreshes while AUTHORIZED
