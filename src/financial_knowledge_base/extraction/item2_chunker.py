import re
from dataclasses import dataclass

from financial_knowledge_base.ingestion.models import DocumentSection


@dataclass(frozen=True)
class Item2Chunk:
    """A bounded Item 2 text slice with inherited heading context."""

    chunk_id: str
    character_start: int
    character_end: int
    heading_context: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class _Paragraph:
    start: int
    end: int
    text: str


class Item2Chunker:
    """Split Item 2 at period boundaries and bounded paragraph groups."""

    _PERIOD_HEADING = re.compile(
        r"^(?:Three|Six|Nine) Months Ended .+ Compared with .+$",
        re.IGNORECASE,
    )
    _IGNORED_HEADINGS = frozenset({"PART I", "Item 2"})

    def __init__(self, *, maximum_characters: int = 12_000) -> None:
        if maximum_characters < 1_000:
            raise ValueError("maximum_characters must be at least 1000")
        self._maximum_characters = maximum_characters

    def split(self, section: DocumentSection) -> list[Item2Chunk]:
        paragraphs = self._paragraphs(section.text)
        if not paragraphs:
            return []

        chunks: list[Item2Chunk] = []
        chunk_paragraphs: list[_Paragraph] = []
        heading_context: list[str] = []
        chunk_headings: tuple[str, ...] = ()

        for paragraph in paragraphs:
            is_period_boundary = bool(self._PERIOD_HEADING.fullmatch(paragraph.text))
            if is_period_boundary and chunk_paragraphs:
                chunks.append(
                    self._build_chunk(
                        chunks,
                        chunk_paragraphs,
                        chunk_headings,
                        section.text,
                    )
                )
                chunk_paragraphs = []

            if self._would_exceed_limit(chunk_paragraphs, paragraph):
                chunks.append(
                    self._build_chunk(
                        chunks,
                        chunk_paragraphs,
                        chunk_headings,
                        section.text,
                    )
                )
                chunk_paragraphs = []

            heading = self._heading(paragraph.text)
            if heading is not None:
                if is_period_boundary:
                    heading_context = [heading]
                else:
                    heading_context = [*heading_context[-2:], heading]

            if not chunk_paragraphs:
                chunk_headings = tuple(heading_context)
            chunk_paragraphs.append(paragraph)

        if chunk_paragraphs:
            chunks.append(
                self._build_chunk(
                    chunks,
                    chunk_paragraphs,
                    chunk_headings,
                    section.text,
                )
            )
        return chunks

    def _paragraphs(self, text: str) -> list[_Paragraph]:
        return [
            _Paragraph(
                start=match.start(),
                end=match.end(),
                text=match.group().strip(),
            )
            for match in re.finditer(r"[^\n](?:.*?[^\n])?(?=\n{2,}|\Z)", text, re.DOTALL)
            if match.group().strip()
        ]

    def _would_exceed_limit(
        self,
        current: list[_Paragraph],
        candidate: _Paragraph,
    ) -> bool:
        if not current:
            return False
        return candidate.end - current[0].start > self._maximum_characters

    def _build_chunk(
        self,
        existing_chunks: list[Item2Chunk],
        paragraphs: list[_Paragraph],
        heading_context: tuple[str, ...],
        source_text: str,
    ) -> Item2Chunk:
        start = paragraphs[0].start
        end = paragraphs[-1].end
        return Item2Chunk(
            chunk_id=f"item2-chunk-{len(existing_chunks) + 1:03d}",
            character_start=start,
            character_end=end,
            heading_context=heading_context,
            text=source_text[start:end],
        )

    def _heading(self, paragraph: str) -> str | None:
        if paragraph in self._IGNORED_HEADINGS:
            return None
        if "\n" in paragraph or len(paragraph) > 120:
            return None
        if paragraph.endswith((".", ":", ";")):
            return None
        if self._PERIOD_HEADING.fullmatch(paragraph) or paragraph.isupper():
            return paragraph
        words = paragraph.split()
        if len(words) <= 8 and all(
            word[:1].isupper() or word.casefold() in {"and", "of", "the"}
            for word in words
        ):
            return paragraph
        return None
