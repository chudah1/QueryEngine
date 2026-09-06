from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, HttpUrl


class FilingMetadata(BaseModel):
    """Metadata for a financial filing."""

    cik: str
    accession_number: str
    form_type: str
    filed_at: date
    reporting_period: date | None
    primary_document: str
    source_url: HttpUrl


class FilingArchiveMetadata(BaseModel):
    """Metadata stored beside an immutable raw filing."""

    filing: FilingMetadata
    content_sha256: str
    content_length_bytes: int
    downloaded_at: datetime


class StoredFiling(BaseModel):
    """Result of saving a raw filing to the local archive."""

    status: Literal["stored", "unchanged"]
    filing_path: Path
    metadata_path: Path
    content_sha256: str


class DocumentSection(BaseModel):
    """A stable, addressable section of normalized filing text."""

    section_id: str
    part: str | None
    item_number: str
    heading: str
    order: int
    character_start: int
    character_end: int
    text: str


class ParsedDocument(BaseModel):
    """Deterministic text representation derived from a raw filing."""

    parser_version: str
    source_content_sha256: str
    text_content_sha256: str
    text: str
    sections: list[DocumentSection]


class StoredParsedDocument(BaseModel):
    """Result of writing a parsed filing artifact."""

    status: Literal["stored", "unchanged", "regenerated"]
    document_path: Path
    parser_version: str
    section_count: int
    character_count: int


class IngestedFiling(BaseModel):
    """Raw and parsed artifacts produced by one ingestion run."""

    raw: StoredFiling
    parsed: StoredParsedDocument
