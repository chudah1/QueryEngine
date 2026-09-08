from financial_knowledge_base.extraction.models import (
    GroundedClaim,
    Item2ExtractionProposal,
    ProposedClaim,
)

from .models import GoldItem2Case, Item2EvaluationReport


class EvaluationInputError(RuntimeError):
    """Raised when a proposal and gold case cannot be compared."""


class Item2Evaluator:
    """Score proposals using exact evidence quotes as stable claim identities."""

    def evaluate(
        self,
        *,
        proposal: Item2ExtractionProposal,
        gold_case: GoldItem2Case,
    ) -> Item2EvaluationReport:
        self._validate_inputs(proposal, gold_case)

        gold_by_quote = {
            claim.evidence_quote: claim for claim in gold_case.expected_claims
        }
        if len(gold_by_quote) != len(gold_case.expected_claims):
            raise EvaluationInputError("Gold case contains duplicate evidence quotes")

        matched_quotes: set[str] = set()
        matched_count = 0
        correct_type_count = 0
        exact_match_count = 0

        for prediction in proposal.claims:
            gold_claim = gold_by_quote.get(prediction.evidence_quote)
            if (
                prediction.evidence_status != "verified"
                or gold_claim is None
                or prediction.evidence_quote in matched_quotes
            ):
                continue

            matched_quotes.add(prediction.evidence_quote)
            matched_count += 1
            if prediction.claim_type == gold_claim.claim_type:
                correct_type_count += 1
            if self._as_proposed_claim(prediction) == gold_claim:
                exact_match_count += 1

        predicted_count = len(proposal.claims)
        expected_count = len(gold_case.expected_claims)
        precision = self._ratio(matched_count, predicted_count)
        recall = self._ratio(matched_count, expected_count)

        return Item2EvaluationReport(
            case_name=gold_case.case_name,
            proposal_response_id=proposal.response_id,
            model=proposal.model,
            prompt_version=proposal.prompt_version,
            predicted_claim_count=predicted_count,
            expected_claim_count=expected_count,
            matched_claim_count=matched_count,
            false_positive_count=predicted_count - matched_count,
            false_negative_count=expected_count - matched_count,
            correct_claim_type_count=correct_type_count,
            exact_claim_match_count=exact_match_count,
            claim_precision=precision,
            claim_recall=recall,
            claim_f1=self._f1(precision, recall),
            claim_type_accuracy=self._ratio(correct_type_count, matched_count),
            exact_claim_match_rate=self._ratio(exact_match_count, matched_count),
            evidence_verification_rate=self._ratio(
                sum(claim.evidence_status == "verified" for claim in proposal.claims),
                predicted_count,
            ),
            ungrounded_claim_rate=self._ratio(
                sum(claim.evidence_status != "verified" for claim in proposal.claims),
                predicted_count,
            ),
            unmatched_claim_rate=self._ratio(
                predicted_count - matched_count,
                predicted_count,
            ),
        )

    def _validate_inputs(
        self,
        proposal: Item2ExtractionProposal,
        gold_case: GoldItem2Case,
    ) -> None:
        if proposal.source_content_sha256 != gold_case.source_content_sha256:
            raise EvaluationInputError("Proposal and gold case source hashes differ")
        if proposal.text_content_sha256 != gold_case.text_content_sha256:
            raise EvaluationInputError("Proposal and gold case text hashes differ")
        if proposal.section_id != gold_case.section_id:
            raise EvaluationInputError("Proposal and gold case section IDs differ")

    def _as_proposed_claim(self, claim: GroundedClaim) -> ProposedClaim:
        return ProposedClaim(
            claim_type=claim.claim_type,
            statement=claim.statement,
            evidence_quote=claim.evidence_quote,
            metric=claim.metric,
            value=claim.value,
            period=claim.period,
        )

    def _ratio(self, numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    def _f1(self, precision: float | None, recall: float | None) -> float | None:
        if precision is None or recall is None:
            return None
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
