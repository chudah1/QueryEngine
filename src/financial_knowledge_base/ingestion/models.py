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
