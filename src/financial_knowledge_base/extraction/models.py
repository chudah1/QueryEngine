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
    scope_quote: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
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
    scope_character_start: int | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    scope_character_end: int | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class Item2ExtractionResponse(BaseModel):
    """One model response produced while mapping over an Item 2 chunk."""

    chunk_id: str
    chunk_character_start: int
    chunk_character_end: int
    heading_context: list[str]
    model: str
    response_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    raw_response: str
    proposed_claim_count: int
    response_status: Literal["completed", "content_filtered"] = "completed"
    failure_reason: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class Item2ChunkCheckpoint(BaseModel):
    """A successfully parsed model response for one immutable input chunk."""

    schema_version: Literal["item2-chunk-checkpoint-v1"] = (
        "item2-chunk-checkpoint-v1"
    )
    prompt_version: str
    prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    text_content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    chunk_text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    response: Item2ExtractionResponse
    output: Item2ModelOutput


class Item2ExtractionProposal(BaseModel):
    """Auditable proposal derived from one parsed filing section."""

    schema_version: str
    prompt_version: str
    prompt_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
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
    responses: list[Item2ExtractionResponse] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
    )


class StoredExtractionProposal(BaseModel):
    """Result of storing an immutable extraction proposal."""

    status: Literal["stored", "unchanged"]
    proposal_path: Path
    claim_count: int
    verified_claim_count: int
