import json
from collections.abc import Iterable
from datetime import UTC, datetime

from financial_knowledge_base.ai.llm import LLMClient
from financial_knowledge_base.ingestion.models import DocumentSection, ParsedDocument

from .checkpoint_store import Item2ChunkCheckpointStore
from .item2_chunker import Item2Chunk, Item2Chunker
from .models import (
    GroundedClaim,
    Item2ChunkCheckpoint,
    Item2ExtractionProposal,
    Item2ExtractionResponse,
    Item2ModelOutput,
    ProposedClaim,
)
from .prompts import load_item2_prompt


class Item2ExtractionError(RuntimeError):
    """Raised when an Item 2 extraction cannot be performed."""


class Item2Extractor:
    """Extract and ground financial claims from Part I, Item 2."""

    SECTION_ID = "part-i-item-2"
    SCHEMA_VERSION = "item2-claims-v2"
    PROMPT_VERSION = "item2-extraction-v3"

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        chunker: Item2Chunker | None = None,
        checkpoint_store: Item2ChunkCheckpointStore | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._chunker = chunker or Item2Chunker()
        self._checkpoint_store = checkpoint_store
        self._prompt = load_item2_prompt(self.PROMPT_VERSION)

    async def extract(self, document: ParsedDocument) -> Item2ExtractionProposal:
        section = self._find_item_2(document)
        chunks = self._chunker.split(section)
        if not chunks:
            raise Item2ExtractionError("Item 2 section contains no extractable text")
        responses: list[Item2ExtractionResponse] = []
        claims: list[GroundedClaim] = []
        models: set[str] = set()

        for chunk in chunks:
            checkpoint = self._load_checkpoint(document, chunk)
            if checkpoint is None:
                checkpoint = await self._extract_chunk(document, section, chunk)
                if self._checkpoint_store is not None:
                    self._checkpoint_store.save(checkpoint=checkpoint, chunk=chunk)

            response_metadata = checkpoint.response
            output = checkpoint.output
            models.add(response_metadata.model)
            responses.append(response_metadata)
            claims.extend(
                self._ground_claim(section, chunk, claim)
                for claim in output.claims
            )

        if len(models) != 1:
            raise Item2ExtractionError(
                f"Expected one response model across all chunks; found {models}"
            )
        claims = self._deduplicate_claims(claims)

        return Item2ExtractionProposal(
            schema_version=self.SCHEMA_VERSION,
            prompt_version=self._prompt.version,
            prompt_sha256=self._prompt.sha256,
            generated_at=datetime.now(UTC),
            source_content_sha256=document.source_content_sha256,
            text_content_sha256=document.text_content_sha256,
            parser_version=document.parser_version,
            section_id=section.section_id,
            model=next(iter(models)),
            response_id=None,
            input_tokens=self._sum_usage(
                response.input_tokens for response in responses
            ),
            output_tokens=self._sum_usage(
                response.output_tokens for response in responses
            ),
            raw_response=json.dumps(
                [response.raw_response for response in responses],
                ensure_ascii=False,
            ),
            claims=claims,
            responses=responses,
        )

    def _load_checkpoint(
        self,
        document: ParsedDocument,
        chunk: Item2Chunk,
    ) -> Item2ChunkCheckpoint | None:
        if self._checkpoint_store is None:
            return None
        return self._checkpoint_store.load(
            chunk=chunk,
            prompt_version=self._prompt.version,
            prompt_sha256=self._prompt.sha256,
            source_content_sha256=document.source_content_sha256,
            text_content_sha256=document.text_content_sha256,
        )

    async def _extract_chunk(
        self,
        document: ParsedDocument,
        section: DocumentSection,
        chunk: Item2Chunk,
    ) -> Item2ChunkCheckpoint:
        try:
            response = await self._llm_client.generate_structured_response(
                system_prompt=self._prompt.text,
                user_prompt=self._user_prompt(section, chunk),
                output_model=Item2ModelOutput,
            )
        except Exception as error:
            raise Item2ExtractionError(
                f"Extraction failed for {chunk.chunk_id}: {error}"
            ) from error

        response_metadata = Item2ExtractionResponse(
            chunk_id=chunk.chunk_id,
            chunk_character_start=section.character_start + chunk.character_start,
            chunk_character_end=section.character_start + chunk.character_end,
            heading_context=list(chunk.heading_context),
            model=response.model,
            response_id=response.response_id,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            raw_response=response.raw_response,
            proposed_claim_count=len(response.output.claims),
        )
        return Item2ChunkCheckpoint(
            prompt_version=self._prompt.version,
            prompt_sha256=self._prompt.sha256,
            source_content_sha256=document.source_content_sha256,
            text_content_sha256=document.text_content_sha256,
            chunk_text_sha256=Item2ChunkCheckpointStore.chunk_text_sha256(chunk),
            response=response_metadata,
            output=response.output,
        )

    def _find_item_2(self, document: ParsedDocument) -> DocumentSection:
        matching_sections = [
            section
            for section in document.sections
            if section.section_id == self.SECTION_ID
        ]
        if len(matching_sections) != 1:
            raise Item2ExtractionError(
                f"Expected one {self.SECTION_ID} section; found "
                f"{len(matching_sections)}"
            )
        return matching_sections[0]

    def _user_prompt(self, section: DocumentSection, chunk: Item2Chunk) -> str:
        heading_context = " > ".join(chunk.heading_context) or "Item 2"
        return (
            f"Section ID: {section.section_id}\n"
            f"Chunk ID: {chunk.chunk_id}\n"
            f"Heading context: {heading_context}\n"
            "FILING SLICE START\n"
            f"{chunk.text}\n"
            "FILING SLICE END"
        )

    def _ground_claim(
        self,
        section: DocumentSection,
        chunk: Item2Chunk,
        claim: ProposedClaim,
    ) -> GroundedClaim:
        first_match = chunk.text.find(claim.evidence_quote)
        if first_match == -1:
            return GroundedClaim(
                **claim.model_dump(),
                evidence_status="not_found",
            )

        second_match = chunk.text.find(claim.evidence_quote, first_match + 1)
        if second_match != -1:
            return GroundedClaim(
                **claim.model_dump(),
                evidence_status="ambiguous",
            )

        evidence_start = section.character_start + chunk.character_start + first_match
        scope_start, scope_end = self._ground_scope(section, chunk, claim)
        if claim.scope_quote is not None and scope_start is None:
            return GroundedClaim(
                **claim.model_dump(),
                evidence_status="not_found",
            )
        global_match_count = section.text.count(claim.evidence_quote)
        if global_match_count > 1 and scope_start is None:
            return GroundedClaim(
                **claim.model_dump(),
                evidence_status="ambiguous",
            )
        return GroundedClaim(
            **claim.model_dump(),
            evidence_status="verified",
            evidence_character_start=evidence_start,
            evidence_character_end=evidence_start + len(claim.evidence_quote),
            scope_character_start=scope_start,
            scope_character_end=scope_end,
        )

    def _ground_scope(
        self,
        section: DocumentSection,
        chunk: Item2Chunk,
        claim: ProposedClaim,
    ) -> tuple[int | None, int | None]:
        if claim.scope_quote is None:
            return None, None
        scope_start_in_chunk = chunk.text.find(claim.scope_quote)
        if scope_start_in_chunk == -1:
            return None, None
        scope_start = (
            section.character_start + chunk.character_start + scope_start_in_chunk
        )
        return scope_start, scope_start + len(claim.scope_quote)

    def _deduplicate_claims(
        self,
        claims: list[GroundedClaim],
    ) -> list[GroundedClaim]:
        deduplicated: list[GroundedClaim] = []
        seen: set[tuple[int | None, int | None, str, str | None]] = set()
        for claim in claims:
            identity = (
                claim.evidence_character_start,
                claim.evidence_character_end,
                claim.claim_type,
                claim.scope_quote,
            )
            if identity in seen:
                continue
            seen.add(identity)
            deduplicated.append(claim)
        return deduplicated

    def _sum_usage(self, values: Iterable[int | None]) -> int | None:
        usage = list(values)
        if any(value is None for value in usage):
            return None
        return sum(value for value in usage if value is not None)
