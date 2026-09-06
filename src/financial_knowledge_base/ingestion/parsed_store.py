from pathlib import Path
from typing import Literal

from .models import ParsedDocument, StoredParsedDocument


class ParsedDocumentStore:
    """Store regenerable parsed documents beside their raw filings."""

    def save(
        self,
        filing_path: Path,
        document: ParsedDocument,
    ) -> StoredParsedDocument:
        document_path = filing_path.with_name("document.json")
        serialized = document.model_dump_json(indent=2) + "\n"
        status: Literal["stored", "unchanged", "regenerated"] = "stored"

        if document_path.exists():
            if document_path.read_text(encoding="utf-8") == serialized:
                status = "unchanged"
            else:
                status = "regenerated"

        if status != "unchanged":
            temporary_path = document_path.with_suffix(".json.tmp")
            temporary_path.write_text(serialized, encoding="utf-8")
            temporary_path.replace(document_path)

        return StoredParsedDocument(
            status=status,
            document_path=document_path,
            parser_version=document.parser_version,
            section_count=len(document.sections),
            character_count=len(document.text),
        )
