import hashlib
import json
from datetime import UTC, datetime

from financial_knowledge_base.extraction.models import (
    GroundedClaim,
    Item2ExtractionProposal,
    ProposedClaim,
)
from financial_knowledge_base.ingestion.models import ParsedDocument

from .models import (
    ClaimReview,
    GoldItem2Case,
    Item2Review,
    ReviewSummary,
)


class ReviewValidationError(RuntimeError):
    """Raised when a review cannot become a trustworthy gold case."""


class Item2ReviewService:
    """Create review templates and finalize them as reusable gold cases."""

    def create_template(
        self,
        *,
        proposal: Item2ExtractionProposal,
        reviewer: str,
    ) -> Item2Review:
        claim_reviews = [
            ClaimReview(
                claim_index=index,
                claim_fingerprint=self._fingerprint(claim),
                proposed_claim=claim,
            )
            for index, claim in enumerate(proposal.claims)
        ]
        return Item2Review(
            created_at=datetime.now(UTC),
            reviewer=reviewer,
            proposal_response_id=proposal.response_id,
            proposal_model=proposal.model,
            prompt_version=proposal.prompt_version,
            prompt_sha256=proposal.prompt_sha256,
            parser_version=proposal.parser_version,
            source_content_sha256=proposal.source_content_sha256,
            text_content_sha256=proposal.text_content_sha256,
            section_id=proposal.section_id,
            claim_reviews=claim_reviews,
        )

    def finalize(
        self,
        *,
        case_name: str,
        proposal: Item2ExtractionProposal,
        review: Item2Review,
        document: ParsedDocument,
    ) -> GoldItem2Case:
        self._validate_provenance(proposal, review, document)
        self._validate_claim_reviews(proposal, review)

        section = next(
            (
                item
                for item in document.sections
                if item.section_id == review.section_id
            ),
            None,
        )
        if section is None:
            raise ReviewValidationError(
                f"Document does not contain section {review.section_id}"
            )

        expected_claims: list[ProposedClaim] = []
        for claim_review in review.claim_reviews:
            if claim_review.decision == "accepted":
                expected_claims.append(
                    self._as_proposed_claim(claim_review.proposed_claim)
                )
            elif claim_review.decision == "needs_edit":
                if claim_review.corrected_claim is None:
                    raise ReviewValidationError(
                        "needs_edit review is missing corrected_claim"
                    )
                expected_claims.append(claim_review.corrected_claim)

        expected_claims.extend(review.missed_claims)
        self._validate_gold_evidence(section.text, expected_claims)

        return GoldItem2Case(
            case_name=case_name,
            created_at=datetime.now(UTC),
            reviewer=review.reviewer,
            source_content_sha256=document.source_content_sha256,
            text_content_sha256=document.text_content_sha256,
            parser_version=document.parser_version,
            section_id=section.section_id,
            section_text=section.text,
            created_from_response_id=proposal.response_id,
            created_from_model=proposal.model,
            created_from_prompt_version=proposal.prompt_version,
            created_from_prompt_sha256=proposal.prompt_sha256,
            review_summary=ReviewSummary(
                accepted=sum(
                    item.decision == "accepted" for item in review.claim_reviews
                ),
                rejected=sum(
                    item.decision == "rejected" for item in review.claim_reviews
                ),
                needs_edit=sum(
                    item.decision == "needs_edit" for item in review.claim_reviews
                ),
                missed=len(review.missed_claims),
            ),
            expected_claims=expected_claims,
        )

    def _validate_provenance(
        self,
        proposal: Item2ExtractionProposal,
        review: Item2Review,
        document: ParsedDocument,
    ) -> None:
        comparisons = {
            "proposal response ID": (
                review.proposal_response_id,
                proposal.response_id,
            ),
            "proposal model": (review.proposal_model, proposal.model),
            "prompt version": (review.prompt_version, proposal.prompt_version),
            "prompt hash": (review.prompt_sha256, proposal.prompt_sha256),
            "parser version": (review.parser_version, proposal.parser_version),
            "source hash": (
                review.source_content_sha256,
                proposal.source_content_sha256,
            ),
            "text hash": (review.text_content_sha256, proposal.text_content_sha256),
            "section ID": (review.section_id, proposal.section_id),
        }
        mismatches = [
            label
            for label, (reviewed, proposed) in comparisons.items()
            if reviewed != proposed
        ]
        if mismatches:
            raise ReviewValidationError(
                "Review does not match proposal: " + ", ".join(mismatches)
            )

        if document.source_content_sha256 != proposal.source_content_sha256:
            raise ReviewValidationError("Document source hash does not match proposal")
        if document.text_content_sha256 != proposal.text_content_sha256:
            raise ReviewValidationError("Document text hash does not match proposal")

    def _validate_claim_reviews(
        self,
        proposal: Item2ExtractionProposal,
        review: Item2Review,
    ) -> None:
        if len(review.claim_reviews) != len(proposal.claims):
            raise ReviewValidationError(
                "Review must contain exactly one entry for every proposed claim"
            )

        seen_indices: set[int] = set()
        for claim_review in review.claim_reviews:
            index = claim_review.claim_index
            if index in seen_indices:
                raise ReviewValidationError(f"Duplicate claim index: {index}")
            if index >= len(proposal.claims):
                raise ReviewValidationError(f"Claim index is out of range: {index}")
            seen_indices.add(index)

            proposed_claim = proposal.claims[index]
            if claim_review.proposed_claim != proposed_claim:
                raise ReviewValidationError(f"Claim {index} differs from the proposal")
            if claim_review.claim_fingerprint != self._fingerprint(proposed_claim):
                raise ReviewValidationError(f"Claim {index} fingerprint does not match")
            if claim_review.decision == "pending":
                raise ReviewValidationError(
                    f"Claim {index} still has a pending decision"
                )

    def _validate_gold_evidence(
        self,
        section_text: str,
        claims: list[ProposedClaim],
    ) -> None:
        seen_evidence: set[tuple[str | None, str]] = set()
        for index, claim in enumerate(claims):
            evidence_identity = (claim.scope_quote, claim.evidence_quote)
            if evidence_identity in seen_evidence:
                raise ReviewValidationError(
                    f"Gold claims contain duplicate evidence at index {index}"
                )
            seen_evidence.add(evidence_identity)

            if claim.scope_quote is not None:
                scope_count = section_text.count(claim.scope_quote)
                if scope_count != 1:
                    raise ReviewValidationError(
                        f"Gold claim {index} scope occurs {scope_count} times; "
                        "expected 1"
                    )
                scope_end = section_text.index(claim.scope_quote) + len(
                    claim.scope_quote
                )
                match_count = section_text[scope_end:].count(claim.evidence_quote)
            else:
                match_count = section_text.count(claim.evidence_quote)
            if match_count < 1:
                raise ReviewValidationError(
                    f"Gold claim {index} evidence does not occur in its scope"
                )

    def _fingerprint(self, claim: GroundedClaim) -> str:
        canonical = json.dumps(
            claim.model_dump(mode="json", exclude_unset=True),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _as_proposed_claim(self, claim: GroundedClaim) -> ProposedClaim:
        return ProposedClaim(
            claim_type=claim.claim_type,
            statement=claim.statement,
            evidence_quote=claim.evidence_quote,
            scope_quote=claim.scope_quote,
            metric=claim.metric,
            value=claim.value,
            period=claim.period,
        )
