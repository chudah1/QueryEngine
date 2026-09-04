from datetime import date

import httpx
from pydantic import HttpUrl

from .models import FilingMetadata


class SECClient:
    BASE_URL = "https://data.sec.gov"

    def __init__(
        self,
        cik: str,
        user_agent: str,
        timeout_seconds: float = 10.0,
    ) -> None:
        if not cik.isdigit():
            raise ValueError("CIK must contain only digits")

        self.cik = cik.zfill(10)
        self.headers = {
            "User-Agent": user_agent,
        }
        self.timeout_seconds = timeout_seconds

    async def get_latest_filing(
        self,
        form_type: str,
    ) -> FilingMetadata | None:
        url = f"{self.BASE_URL}/submissions/CIK{self.cik}.json"

        async with httpx.AsyncClient(
            headers=self.headers,
            timeout=self.timeout_seconds,
        ) as client:
            response = await client.get(
                url,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            payload = response.json()

        recent = payload["filings"]["recent"]

        for index, current_form in enumerate(recent["form"]):
            if current_form != form_type:
                continue

            accession_number = recent["accessionNumber"][index]
            primary_document = recent["primaryDocument"][index]
            reporting_period = recent["reportDate"][index]

            archive_cik = str(int(self.cik))
            archive_accession = accession_number.replace("-", "")
            source_url = (
                "https://www.sec.gov/Archives/edgar/data/"
                f"{archive_cik}/{archive_accession}/{primary_document}"
            )

            return FilingMetadata(
                cik=self.cik,
                accession_number=accession_number,
                form_type=current_form,
                filed_at=date.fromisoformat(recent["filingDate"][index]),
                reporting_period=(
                    date.fromisoformat(reporting_period) if reporting_period else None
                ),
                primary_document=primary_document,
                source_url=HttpUrl(source_url),
            )

        return None

    async def download_filing(self, filing: FilingMetadata) -> bytes:
        """Download the official filing document from the SEC archive."""

        async with httpx.AsyncClient(
            headers=self.headers,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        ) as client:
            response = await client.get(
                str(filing.source_url),
                headers={"Accept": "text/html,application/xhtml+xml"},
            )
            response.raise_for_status()

        if not response.content:
            raise ValueError(
                f"SEC returned an empty filing for {filing.accession_number}"
            )

        return response.content
