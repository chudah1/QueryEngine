import hashlib
import re
import warnings
from dataclasses import dataclass
from typing import ClassVar

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from .models import DocumentSection, ParsedDocument


class FilingParseError(RuntimeError):
    """Raised when a filing cannot be converted into structured text."""


@dataclass(frozen=True)
class _TextBlock:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class _SectionMarker:
    block_index: int
    part: str | None
    item_number: str
    heading: str


class SECHTMLParser:
    """Convert SEC filing HTML into deterministic narrative sections."""

    VERSION = "sec-html-v1"

    _BLOCK_TAGS = ("div", "p", "li", "h1", "h2", "h3", "h4", "h5", "h6")
    _REMOVED_TAGS = ("script", "style", "noscript", "svg", "table")
    _REMOVED_XBRL_TAGS: ClassVar[set[str]] = {
        "ix:header",
        "ix:hidden",
        "ix:references",
        "ix:resources",
    }
    _PART_PATTERN = re.compile(r"^PART\s+([IVX]+)\b", re.IGNORECASE)
    _ITEM_PATTERN = re.compile(
        r"^Item\s+(\d+[A-Z]?)\.\s*(.*)$",
        re.IGNORECASE,
    )
    _PAGE_NUMBER_PATTERN = re.compile(r"^\d+$")

    def parse(self, content: bytes) -> ParsedDocument:
        if not content:
            raise FilingParseError("Cannot parse an empty filing")

        source_content_sha256 = hashlib.sha256(content).hexdigest()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(content, "lxml")

        if soup.body is None:
            raise FilingParseError("Filing does not contain an HTML body")

        self._remove_non_narrative_content(soup)
        block_texts = self._extract_block_texts(soup.body)
        document_text, blocks = self._build_document_text(block_texts)

        if not document_text:
            raise FilingParseError("Filing did not contain readable text")

        sections = self._build_sections(document_text, blocks)
        if not sections:
            raise FilingParseError("Filing did not contain recognizable item sections")

        return ParsedDocument(
            parser_version=self.VERSION,
            source_content_sha256=source_content_sha256,
            text_content_sha256=hashlib.sha256(
                document_text.encode("utf-8")
            ).hexdigest(),
            text=document_text,
            sections=sections,
        )

    def _remove_non_narrative_content(self, soup: BeautifulSoup) -> None:
        for element in soup.find_all(self._REMOVED_TAGS):
            element.decompose()

        for element in soup.find_all(
            lambda tag: (
                tag.name is not None and tag.name.lower() in self._REMOVED_XBRL_TAGS
            )
        ):
            element.decompose()

        for element in soup.find_all(style=True):
            style = str(element.get("style", "")).replace(" ", "").lower()
            if "display:none" in style or "visibility:hidden" in style:
                element.decompose()

    def _extract_block_texts(self, body: Tag) -> list[str]:
        block_names = set(self._BLOCK_TAGS)
        texts: list[str] = []

        for element in body.find_all(self._BLOCK_TAGS):
            has_direct_block_child = any(
                isinstance(child, Tag) and child.name in block_names
                for child in element.find_all(recursive=False)
            )
            if has_direct_block_child:
                continue

            text = " ".join(element.get_text(" ", strip=True).split())
            if not text or self._PAGE_NUMBER_PATTERN.fullmatch(text):
                continue
            if texts and texts[-1] == text:
                continue

            texts.append(text)

        return texts

    def _build_document_text(
        self,
        block_texts: list[str],
    ) -> tuple[str, list[_TextBlock]]:
        blocks: list[_TextBlock] = []
        cursor = 0

        for text in block_texts:
            start = cursor
            end = start + len(text)
            blocks.append(_TextBlock(text=text, start=start, end=end))
            cursor = end + 2

        return "\n\n".join(block_texts), blocks

    def _build_sections(
        self,
        document_text: str,
        blocks: list[_TextBlock],
    ) -> list[DocumentSection]:
        markers: list[_SectionMarker] = []
        current_part: str | None = None

        for block_index, block in enumerate(blocks):
            part_match = self._PART_PATTERN.match(block.text)
            if part_match and len(block.text) <= 160:
                current_part = part_match.group(1).upper()
                continue

            item_match = self._ITEM_PATTERN.match(block.text)
            if item_match is None or len(block.text) > 240:
                continue

            markers.append(
                _SectionMarker(
                    block_index=block_index,
                    part=current_part,
                    item_number=item_match.group(1).upper(),
                    heading=block.text,
                )
            )

        sections: list[DocumentSection] = []
        identifiers: dict[str, int] = {}

        for order, marker in enumerate(markers):
            start = blocks[marker.block_index].start
            next_start = (
                blocks[markers[order + 1].block_index].start
                if order + 1 < len(markers)
                else len(document_text)
            )
            section_text = document_text[start:next_start].rstrip()
            end = start + len(section_text)

            base_identifier = self._section_identifier(
                part=marker.part,
                item_number=marker.item_number,
            )
            occurrence = identifiers.get(base_identifier, 0) + 1
            identifiers[base_identifier] = occurrence
            section_id = (
                base_identifier
                if occurrence == 1
                else f"{base_identifier}-{occurrence}"
            )

            sections.append(
                DocumentSection(
                    section_id=section_id,
                    part=marker.part,
                    item_number=marker.item_number,
                    heading=marker.heading,
                    order=order,
                    character_start=start,
                    character_end=end,
                    text=section_text,
                )
            )

        return sections

    def _section_identifier(self, part: str | None, item_number: str) -> str:
        normalized_item = item_number.lower()
        if part is None:
            return f"item-{normalized_item}"
        return f"part-{part.lower()}-item-{normalized_item}"
