# Single identity hypothesis for a face track
#
# Identity hypothesis is always published (top-k + scores).
# Decision state controls whether we treat it as actionable.
# We never confirm AMBIGUOUS.

uint64 track_id                     # From FaceTracks
string user_id                      # Best match or "UNKNOWN"
float32 score                       # Cosine similarity [0-1]
float32 margin                      # Gap to second-best
string decision_state               # UNKNOWN | AMBIGUOUS | TENTATIVE | CONFIRMED
builtin_interfaces/Time timestamp

# Decision states:
# - UNKNOWN: score < threshold (no match)
# - AMBIGUOUS: score >= threshold but margin < margin_threshold (too close to call)
# - TENTATIVE: good match, waiting for temporal confirmation
# - CONFIRMED: same identity stable for confirm_time_sec

# Top-K alternatives
string[] alt_user_ids
float32[] alt_scores

# Quality metrics
uint32 embedding_count              # Embeddings in buffer for this track
float32 buffer_stability            # Std dev of recent embeddings (0 if not computed)
