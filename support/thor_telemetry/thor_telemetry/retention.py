"""Retention declaration schema for data governance (SI-12.4, SI-13.2).

Each component that produces or stores data declares a retention manifest
covering all its data categories.  RetentionDeclaration is a validated
dataclass — construction fails if required fields are missing or empty.

Usage:
    from thor_telemetry import RetentionDeclaration

    decl = RetentionDeclaration(
        data_category="face_embeddings",
        owner="face_state_server",
        retention_period="indefinite_until_explicit_delete",
        access_control="component-internal",
        deletion_mechanism="explicit_api",
    )
"""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(frozen=True)
class RetentionDeclaration:
    """Static declaration of data retention, ownership, and deletion semantics.

    Required fields must be non-empty strings.  ``legal_basis`` is optional.
    Raises ``ValueError`` on construction if any required field is missing
    or empty.
    """

    data_category: str
    owner: str
    retention_period: str
    access_control: str
    deletion_mechanism: str
    legal_basis: Optional[str] = None

    def __post_init__(self):
        required = {
            "data_category": self.data_category,
            "owner": self.owner,
            "retention_period": self.retention_period,
            "access_control": self.access_control,
            "deletion_mechanism": self.deletion_mechanism,
        }
        missing = [k for k, v in required.items() if not v or not v.strip()]
        if missing:
            raise ValueError(
                f"RetentionDeclaration requires non-empty: {', '.join(missing)}"
            )

    def to_dict(self) -> dict:
        """Serialize to dict, omitting None fields."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}
