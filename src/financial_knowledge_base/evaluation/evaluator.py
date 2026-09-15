import re
from dataclasses import dataclass

from financial_knowledge_base.extraction.models import (
    GroundedClaim,
    Item2ExtractionProposal,
    ProposedClaim,
)

from .models import (
    ClaimMatchMethod,
    GoldItem2Case,
    Item2ClaimMatch,
    Item2EvaluationReport,
)


@dataclass(frozen=True)
class _MatchCandidate:
    predicted_index: int
    expected_index: int
    method: ClaimMatchMethod
    score: float
    evidence_overlap: float
    statement_similarity: float
    field_similarity: float


class EvaluationInputError(RuntimeError):
    """Raised when a proposal and gold case cannot be compared."""


class Item2Evaluator:
    """Score proposals using auditable, deterministic one-to-one claim matches."""

    _EVIDENCE_OVERLAP_THRESHOLD = 0.8
    _STATEMENT_SIMILARITY_THRESHOLD = 0.72
    _FIELD_SIMILARITY_THRESHOLD = 0.6

    def evaluate(
        self,
        *,
        proposal: Item2ExtractionProposal,
        gold_case: GoldItem2Case,
    ) -> Item2EvaluationReport:
        self._validate_inputs(proposal, gold_case)

        matches = self._match_claims(proposal, gold_case)
        matched_count = len(matches)
        correct_type_count = 0
        exact_match_count = 0

        for match in matches:
            prediction = proposal.claims[match.predicted_claim_index]
            gold_claim = gold_case.expected_claims[match.expected_claim_index]
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
            prompt_sha256=proposal.prompt_sha256,
            predicted_claim_count=predicted_count,
            expected_claim_count=expected_count,
            matched_claim_count=matched_count,
            false_positive_count=predicted_count - matched_count,
            false_negative_count=expected_count - matched_count,
            exact_evidence_match_count=sum(
                match.method == "exact_evidence" for match in matches
            ),
            evidence_overlap_match_count=sum(
                match.method == "evidence_overlap" for match in matches
            ),
            field_similarity_match_count=sum(
                match.method == "field_similarity" for match in matches
            ),
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
            matches=matches,
        )

    def _match_claims(
        self,
        proposal: Item2ExtractionProposal,
        gold_case: GoldItem2Case,
    ) -> list[Item2ClaimMatch]:
        candidates: list[_MatchCandidate] = []
        gold_spans = [
            self._evidence_span(
                gold_case.section_text,
                claim.evidence_quote,
                claim.scope_quote,
            )
            for claim in gold_case.expected_claims
        ]

        for predicted_index, prediction in enumerate(proposal.claims):
            if prediction.evidence_status != "verified":
                continue
            predicted_span = self._evidence_span(
                gold_case.section_text,
                prediction.evidence_quote,
                prediction.scope_quote,
            )
            for expected_index, expected in enumerate(gold_case.expected_claims):
                evidence_overlap = self._span_overlap(
                    predicted_span,
                    gold_spans[expected_index],
                )
                statement_similarity = self._text_similarity(
                    prediction.statement,
                    expected.statement,
                )
                field_similarity = self._claim_field_similarity(
                    prediction,
                    expected,
                )
                candidate = self._candidate(
                    predicted_index=predicted_index,
                    expected_index=expected_index,
                    exact_evidence=(
                        prediction.evidence_quote == expected.evidence_quote
                    ),
                    evidence_overlap=evidence_overlap,
                    statement_similarity=statement_similarity,
                    field_similarity=field_similarity,
                )
                if candidate is not None:
                    candidates.append(candidate)

        candidates.sort(
            key=lambda item: (
                -item.score,
                item.predicted_index,
                item.expected_index,
            )
        )
        used_predictions: set[int] = set()
        used_expected: set[int] = set()
        matches: list[Item2ClaimMatch] = []
        for candidate in candidates:
            if (
                candidate.predicted_index in used_predictions
                or candidate.expected_index in used_expected
            ):
                continue
            used_predictions.add(candidate.predicted_index)
            used_expected.add(candidate.expected_index)
            matches.append(
                Item2ClaimMatch(
                    predicted_claim_index=candidate.predicted_index,
                    expected_claim_index=candidate.expected_index,
                    method=candidate.method,
                    score=candidate.score,
                    evidence_overlap=candidate.evidence_overlap,
                    statement_similarity=candidate.statement_similarity,
                    field_similarity=candidate.field_similarity,
                )
            )

        return sorted(matches, key=lambda item: item.predicted_claim_index)

    def _candidate(
        self,
        *,
        predicted_index: int,
        expected_index: int,
        exact_evidence: bool,
        evidence_overlap: float,
        statement_similarity: float,
        field_similarity: float,
    ) -> _MatchCandidate | None:
        if exact_evidence:
            method: ClaimMatchMethod = "exact_evidence"
            score = 1.0
        elif evidence_overlap >= self._EVIDENCE_OVERLAP_THRESHOLD:
            method = "evidence_overlap"
            score = 0.85 * evidence_overlap + 0.15 * statement_similarity
        elif (
            statement_similarity >= self._STATEMENT_SIMILARITY_THRESHOLD
            and field_similarity >= self._FIELD_SIMILARITY_THRESHOLD
        ):
            method = "field_similarity"
            score = 0.7 * statement_similarity + 0.3 * field_similarity
        else:
            return None

        return _MatchCandidate(
            predicted_index=predicted_index,
            expected_index=expected_index,
            method=method,
            score=min(score, 1.0),
            evidence_overlap=evidence_overlap,
            statement_similarity=statement_similarity,
            field_similarity=field_similarity,
        )

    def _evidence_span(
        self,
        section_text: str,
        evidence_quote: str,
        scope_quote: str | None,
    ) -> tuple[int, int]:
        search_start = 0
        if scope_quote is not None:
            scope_start = section_text.find(scope_quote)
            if scope_start == -1:
                raise EvaluationInputError("Evidence scope is absent from gold section")
            search_start = scope_start + len(scope_quote)
        start = section_text.find(evidence_quote, search_start)
        if start == -1:
            raise EvaluationInputError("Verified evidence is absent from gold section")
        return start, start + len(evidence_quote)

    def _span_overlap(
        self,
        left: tuple[int, int],
        right: tuple[int, int],
    ) -> float:
        overlap = max(0, min(left[1], right[1]) - max(left[0], right[0]))
        shorter_length = min(left[1] - left[0], right[1] - right[0])
        return self._ratio(overlap, shorter_length) or 0.0

    def _claim_field_similarity(
        self,
        prediction: GroundedClaim,
        expected: ProposedClaim,
    ) -> float:
        comparable_fields = [
            (prediction.metric, expected.metric),
            (prediction.value, expected.value),
            (prediction.period, expected.period),
        ]
        scores = [
            self._text_similarity(left, right)
            for left, right in comparable_fields
            if left is not None and right is not None
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def _text_similarity(self, left: str, right: str) -> float:
        left_tokens = set(re.findall(r"[a-z0-9]+", left.casefold()))
        right_tokens = set(re.findall(r"[a-z0-9]+", right.casefold()))
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

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
            scope_quote=claim.scope_quote,
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
