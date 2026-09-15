from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from financial_knowledge_base.extraction.models import GroundedClaim, ProposedClaim

ReviewDecision = Literal["pending", "accepted", "rejected", "needs_edit"]
ClaimMatchMethod = Literal[
    "exact_evidence",
    "evidence_overlap",
    "field_similarity",
]
RejectionReason = Literal[
    "unsupported",
    "not_material",
    "duplicate",
    "wrong_scope",
    "other",
]


class ClaimReview(BaseModel):
    """Human decision for one claim in an extraction proposal."""

    claim_index: int = Field(ge=0)
    claim_fingerprint: str
    proposed_claim: GroundedClaim
    decision: ReviewDecision = "pending"
    rejection_reason: RejectionReason | None = None
    corrected_claim: ProposedClaim | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def validate_decision_details(self) -> "ClaimReview":
        if self.decision == "rejected" and self.rejection_reason is None:
            raise ValueError("A rejected claim requires rejection_reason")
        if self.decision == "needs_edit" and self.corrected_claim is None:
            raise ValueError("A claim marked needs_edit requires corrected_claim")
        if self.decision != "rejected" and self.rejection_reason is not None:
            raise ValueError("rejection_reason is only valid for rejected claims")
        if self.decision != "needs_edit" and self.corrected_claim is not None:
            raise ValueError("corrected_claim is only valid for needs_edit claims")
        return self


class Item2Review(BaseModel):
    """Editable review document tied to one immutable proposal."""

    schema_version: Literal["item2-review-v1"] = "item2-review-v1"
    created_at: datetime
    reviewer: str = Field(min_length=1)
    proposal_response_id: str | None
    proposal_model: str
    prompt_version: str
    prompt_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    parser_version: str
    source_content_sha256: str
    text_content_sha256: str
    section_id: str
    claim_reviews: list[ClaimReview]
    missed_claims: list[ProposedClaim] = Field(default_factory=list)


class ReviewSummary(BaseModel):
    """Counts retained as provenance for a finalized gold case."""

    accepted: int
    rejected: int
    needs_edit: int
    missed: int


class GoldItem2Case(BaseModel):
    """Human-labelled Item 2 input and expected claims."""

    schema_version: Literal["item2-gold-v1"] = "item2-gold-v1"
    case_name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    created_at: datetime
    reviewer: str
    source_content_sha256: str
    text_content_sha256: str
    parser_version: str
    section_id: str
    section_text: str
    created_from_response_id: str | None
    created_from_model: str
    created_from_prompt_version: str | None = None
    created_from_prompt_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    review_summary: ReviewSummary
    expected_claims: list[ProposedClaim]


class Item2ClaimMatch(BaseModel):
    """Auditable link between one predicted claim and one expected claim."""

    predicted_claim_index: int = Field(ge=0)
    expected_claim_index: int = Field(ge=0)
    method: ClaimMatchMethod
    score: float = Field(ge=0.0, le=1.0)
    evidence_overlap: float = Field(ge=0.0, le=1.0)
    statement_similarity: float = Field(ge=0.0, le=1.0)
    field_similarity: float = Field(ge=0.0, le=1.0)


class Item2EvaluationReport(BaseModel):
    """Deterministic comparison of one proposal with a gold case."""

    schema_version: Literal[
        "item2-evaluation-v1",
        "item2-evaluation-v2",
    ] = "item2-evaluation-v2"
    case_name: str
    proposal_response_id: str | None
    model: str
    prompt_version: str
    prompt_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    predicted_claim_count: int
    expected_claim_count: int
    matched_claim_count: int
    false_positive_count: int
    false_negative_count: int
    exact_evidence_match_count: int = 0
    evidence_overlap_match_count: int = 0
    field_similarity_match_count: int = 0
    correct_claim_type_count: int
    exact_claim_match_count: int
    claim_precision: float | None
    claim_recall: float | None
    claim_f1: float | None
    claim_type_accuracy: float | None
    exact_claim_match_rate: float | None
    evidence_verification_rate: float | None
    ungrounded_claim_rate: float | None
    unmatched_claim_rate: float | None
    matches: list[Item2ClaimMatch] = Field(default_factory=list)


class StoredEvaluationArtifact(BaseModel):
    """Result of storing a review, gold case, or evaluation report."""

    status: Literal["stored", "unchanged"]
    path: Path
