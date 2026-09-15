from pathlib import Path

import pytest

from financial_knowledge_base.ai.llm import LLMResponse
from financial_knowledge_base.extraction.checkpoint_store import (
    Item2ChunkCheckpointStore,
)
from financial_knowledge_base.extraction.item2_chunker import Item2Chunker
from financial_knowledge_base.extraction.item2_extractor import Item2Extractor
from financial_knowledge_base.extraction.models import Item2ModelOutput
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


def _response(response_id: str) -> LLMResponse[Item2ModelOutput]:
    return LLMResponse(
        output=Item2ModelOutput(claims=[]),
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
