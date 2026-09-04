from .models import StoredFiling
from .raw_store import RawFilingStore
from .sec_client import SECClient


class FilingIngestor:
    """Coordinate SEC discovery, download, and immutable local storage."""

    def __init__(self, client: SECClient, store: RawFilingStore) -> None:
        self.client = client
        self.store = store

    async def ingest_latest(self, form_type: str) -> StoredFiling | None:
        filing = await self.client.get_latest_filing(form_type)
        if filing is None:
            return None

        content = await self.client.download_filing(filing)
        return self.store.save(filing, content)
