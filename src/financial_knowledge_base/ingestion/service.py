from .models import IngestedFiling
from .parsed_store import ParsedDocumentStore
from .parser import SECHTMLParser
from .raw_store import RawFilingStore
from .sec_client import SECClient


class FilingIngestor:
    """Coordinate SEC discovery, download, storage, and parsing."""

    def __init__(
        self,
        client: SECClient,
        raw_store: RawFilingStore,
        parser: SECHTMLParser,
        parsed_store: ParsedDocumentStore,
    ) -> None:
        self.client = client
        self.raw_store = raw_store
        self.parser = parser
        self.parsed_store = parsed_store

    async def ingest_latest(self, form_type: str) -> IngestedFiling | None:
        filing = await self.client.get_latest_filing(form_type)
        if filing is None:
            return None

        content = await self.client.download_filing(filing)
        stored_filing = self.raw_store.save(filing, content)
        document = self.parser.parse(content)
        stored_document = self.parsed_store.save(
            filing_path=stored_filing.filing_path,
            document=document,
        )

        return IngestedFiling(raw=stored_filing, parsed=stored_document)
