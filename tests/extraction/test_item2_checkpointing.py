from pathlib import Path

import pytest

from financial_knowledge_base.ai.llm import (
    LLMResponse,
    LLMResponseContentFilterError,
    LLMResponseTruncatedError,
)
from financial_knowledge_base.extraction.checkpoint_store import (
    Item2ChunkCheckpointStore,
)
from financial_knowledge_base.extraction.item2_chunker import Item2Chunker
from financial_knowledge_base.extraction.item2_extractor import Item2Extractor
from financial_knowledge_base.extraction.models import Item2ModelOutput, ProposedClaim
from financial_knowledge_base.ingestion.models import DocumentSection, ParsedDocument


class StubLLMClient:
    def __init__(self, results: list[LLMResponse[Item2ModelOutput] | Exception]):
        self._results = iter(results)
        self.calls = 0

    async def generate_structured_response(self, **_: object) -> LLMResponse:
        self.calls += 1
        result = next(self._results)
        if isinstance(result, Exception):
            raise result
        return result


def _response(
    response_id: str,
    claims: list[ProposedClaim] | None = None,
) -> LLMResponse[Item2ModelOutput]:
    return LLMResponse(
        output=Item2ModelOutput(claims=claims or []),
        model="test-model",
        raw_response='{"claims":[]}',
        response_id=response_id,
        input_tokens=10,
        output_tokens=5,
    )


def _document() -> ParsedDocument:
    text = f"Revenue\n\n{'A' * 700}\n\n{'B' * 700}"
    section = DocumentSection(
        section_id="part-i-item-2",
        part="PART I",
        item_number="2",
        heading="Management's Discussion and Analysis",
        order=1,
        character_start=0,
        character_end=len(text),
        text=text,
    )
    return ParsedDocument(
        parser_version="test-parser-v1",
        source_content_sha256="a" * 64,
        text_content_sha256="b" * 64,
        text=text,
        sections=[section],
    )


@pytest.mark.asyncio
async def test_extract_resumes_after_a_later_chunk_fails(tmp_path: Path) -> None:
    checkpoint_store = Item2ChunkCheckpointStore(tmp_path)
    first_client = StubLLMClient([_response("response-1"), RuntimeError("truncated")])
    extractor = Item2Extractor(
        first_client,
        chunker=Item2Chunker(maximum_characters=1_000),
        checkpoint_store=checkpoint_store,
    )

    with pytest.raises(RuntimeError, match="truncated"):
        await extractor.extract(_document())

    retry_client = StubLLMClient([_response("response-2")])
    retry_extractor = Item2Extractor(
        retry_client,
        chunker=Item2Chunker(maximum_characters=1_000),
        checkpoint_store=checkpoint_store,
    )
    proposal = await retry_extractor.extract(_document())

    assert first_client.calls == 2
    assert retry_client.calls == 1
    assert [response.response_id for response in proposal.responses] == [
        "response-1",
        "response-2",
    ]


@pytest.mark.asyncio
async def test_extract_splits_only_a_truncated_chunk(tmp_path: Path) -> None:
    client = StubLLMClient(
        [
            LLMResponseTruncatedError("structured output was truncated"),
            _response("response-child-1"),
            _response("response-child-2"),
        ]
    )
    extractor = Item2Extractor(
        client,
        checkpoint_store=Item2ChunkCheckpointStore(tmp_path),
    )

    proposal = await extractor.extract(_document())

    assert client.calls == 3
    assert len(proposal.responses) == 1
    assert proposal.responses[0].chunk_id == "item2-chunk-001"
    assert proposal.responses[0].proposed_claim_count == 0
    assert len(list(tmp_path.glob("item2-chunk-001.*.json"))) == 2


@pytest.mark.asyncio
async def test_extract_records_content_filter_as_coverage_gap(tmp_path: Path) -> None:
    def filtered(response_id: str) -> LLMResponseContentFilterError:
        return LLMResponseContentFilterError(
            "OpenAI filtered the response",
            model="test-model",
            response_id=response_id,
            raw_response='{"claims":[',
            input_tokens=100,
            output_tokens=0,
        )

    client = StubLLMClient(
        [
            filtered("filtered-parent"),
            _response("completed-child"),
            filtered("filtered-child"),
        ]
    )
    extractor = Item2Extractor(
        client,
        checkpoint_store=Item2ChunkCheckpointStore(tmp_path),
    )

    proposal = await extractor.extract(_document())

    assert proposal.claims == []
    assert proposal.responses[0].response_status == "content_filtered"
    assert proposal.responses[0].failure_reason == (
        "content_filter in item2-chunk-001.002"
    )
    assert proposal.responses[0].response_id is None
    assert client.calls == 3


@pytest.mark.asyncio
async def test_grounding_finds_heading_inherited_from_previous_chunk() -> None:
    heading = "Other Planned Uses of Capital"
    filler = "A" * 1_100
    evidence = "We will continue to invest in product infrastructure."
    text = f"{heading}\n\n{filler}\n\n{evidence}"
    section = DocumentSection(
        section_id="part-i-item-2",
        part="PART I",
        item_number="2",
        heading="Management's Discussion and Analysis",
        order=1,
        character_start=0,
        character_end=len(text),
        text=text,
    )
    document = ParsedDocument(
        parser_version="test-parser-v1",
        source_content_sha256="c" * 64,
        text_content_sha256="d" * 64,
        text=text,
        sections=[section],
    )
    claim = ProposedClaim(
        claim_type="management_outlook",
        statement=evidence,
        evidence_quote=evidence,
        scope_quote=heading,
    )
    client = StubLLMClient(
        [
            _response("response-1"),
            _response("response-2"),
            _response("response-3", [claim]),
        ]
    )
    extractor = Item2Extractor(
        client,
        chunker=Item2Chunker(maximum_characters=1_000),
    )

    proposal = await extractor.extract(document)

    assert proposal.claims[0].evidence_status == "verified"
    assert proposal.claims[0].scope_character_start == 0


@pytest.mark.asyncio
async def test_distinct_ungrounded_claims_are_not_deduplicated() -> None:
    document = _document()
    claims = [
        ProposedClaim(
            claim_type="financial_result",
            statement="First unsupported claim.",
            evidence_quote="Missing quote one.",
            scope_quote="Revenue",
        ),
        ProposedClaim(
            claim_type="financial_result",
            statement="Second unsupported claim.",
            evidence_quote="Missing quote two.",
            scope_quote="Revenue",
        ),
    ]
    client = StubLLMClient([_response("response-1", claims)])
    extractor = Item2Extractor(client)

    proposal = await extractor.extract(document)

    assert len(proposal.claims) == 2
    assert all(claim.evidence_status == "not_found" for claim in proposal.claims)
