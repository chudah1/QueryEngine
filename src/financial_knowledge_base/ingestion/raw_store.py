import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path

from .models import FilingArchiveMetadata, FilingMetadata, StoredFiling


class FilingArchiveError(RuntimeError):
    """Base error for invalid or conflicting raw filing state."""


class FilingArchiveConflictError(FilingArchiveError):
    """Raised when an accession already exists with different content."""


class FilingArchiveCorruptionError(FilingArchiveError):
    """Raised when an existing archive entry is incomplete or inconsistent."""


class RawFilingStore:
    """Store raw SEC filings without overwriting previously archived content."""

    _SAFE_ACCESSION = re.compile(r"^[0-9-]+$")

    def __init__(self, root: Path) -> None:
        self.root = root

    def save(self, filing: FilingMetadata, content: bytes) -> StoredFiling:
        if not content:
            raise ValueError("Cannot store an empty filing")
        if not filing.cik.isdigit():
            raise ValueError("CIK must contain only digits")
        if not self._SAFE_ACCESSION.fullmatch(filing.accession_number):
            raise ValueError("Accession number contains invalid characters")

        content_sha256 = hashlib.sha256(content).hexdigest()
        archive_directory = self.root / filing.cik / filing.accession_number
        filing_path = archive_directory / "filing.html"
        metadata_path = archive_directory / "metadata.json"

        if archive_directory.exists():
            return self._validate_existing(
                filing_path=filing_path,
                metadata_path=metadata_path,
                expected_sha256=content_sha256,
            )

        archive_directory.mkdir(parents=True, exist_ok=False)

        archive_metadata = FilingArchiveMetadata(
            filing=filing,
            content_sha256=content_sha256,
            content_length_bytes=len(content),
            downloaded_at=datetime.now(UTC),
        )

        filing_path.write_bytes(content)
        metadata_path.write_text(
            archive_metadata.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

        return StoredFiling(
            status="stored",
            filing_path=filing_path,
            metadata_path=metadata_path,
            content_sha256=content_sha256,
        )

    def _validate_existing(
        self,
        filing_path: Path,
        metadata_path: Path,
        expected_sha256: str,
    ) -> StoredFiling:
        if not filing_path.is_file() or not metadata_path.is_file():
            raise FilingArchiveCorruptionError(
                f"Incomplete filing archive at {filing_path.parent}"
            )

        try:
            metadata = FilingArchiveMetadata.model_validate_json(
                metadata_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise FilingArchiveCorruptionError(
                f"Invalid archive metadata at {metadata_path}"
            ) from error

        existing_sha256 = hashlib.sha256(filing_path.read_bytes()).hexdigest()
        if metadata.content_sha256 != existing_sha256:
            raise FilingArchiveCorruptionError(
                f"Stored hash does not match {filing_path}"
            )

        if existing_sha256 != expected_sha256:
            raise FilingArchiveConflictError(
                "Refusing to overwrite accession with different content: "
                f"{metadata.filing.accession_number}"
            )

        return StoredFiling(
            status="unchanged",
            filing_path=filing_path,
            metadata_path=metadata_path,
            content_sha256=existing_sha256,
        )
