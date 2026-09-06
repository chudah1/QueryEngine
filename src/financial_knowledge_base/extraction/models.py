from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ClaimType = Literal[
    "financial_result",
    "business_driver",
    "management_outlook",
    "risk",
    "capital_allocation",
]
EvidenceStatus = Literal["verified", "not_found", "ambiguous"]


class ProposedClaim(BaseModel):
    """Claim returned directly by the model."""

    model_config = ConfigDict(extra="forbid")

    claim_type: ClaimType
    statement: str = Field(min_length=1)
    evidence_quote: str = Field(min_length=1)
    metric: str | None = None
    value: str | None = None
    period: str | None = None


class Item2ModelOutput(BaseModel):
    """Structured output requested from the model for an Item 2 section."""

    model_config = ConfigDict(extra="forbid")

    claims: list[ProposedClaim]


class GroundedClaim(ProposedClaim):
    """Model-proposed claim with deterministic evidence verification."""

    evidence_status: EvidenceStatus
    evidence_character_start: int | None = None
    evidence_character_end: int | None = None


class Item2ExtractionProposal(BaseModel):
    """Auditable proposal derived from one parsed filing section."""

    schema_version: str
    prompt_version: str
    generated_at: datetime
    source_content_sha256: str
    text_content_sha256: str
    parser_version: str
    section_id: str
    model: str
    response_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    raw_response: str
    claims: list[GroundedClaim]


class StoredExtractionProposal(BaseModel):
    """Result of storing an immutable extraction proposal."""

    status: Literal["stored", "unchanged"]
    proposal_path: Path
    claim_count: int
    verified_claim_count: int
