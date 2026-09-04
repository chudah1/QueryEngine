import asyncio
from pathlib import Path
from typing import Annotated

import typer

from .ingestion.models import StoredFiling
from .ingestion.raw_store import RawFilingStore
from .ingestion.sec_client import SECClient
from .ingestion.service import FilingIngestor

app = typer.Typer(no_args_is_help=True)


@app.callback()
def application() -> None:
    """Maintain a versioned financial knowledge base."""


@app.command()
def ingest(
    cik: Annotated[str, typer.Option(help="SEC Central Index Key.")],
    user_agent: Annotated[
        str,
        typer.Option(
            envvar="SEC_USER_AGENT",
            help="Identifying SEC User-Agent, including a contact address.",
        ),
    ],
    form_type: Annotated[
        str,
        typer.Option("--form", help="SEC form type."),
    ] = "10-Q",
    raw_data_dir: Annotated[
        Path,
        typer.Option(help="Directory for immutable raw filings."),
    ] = Path("data/raw"),
) -> None:
    """Download and archive the latest matching SEC filing."""

    result = asyncio.run(
        _ingest_latest(
            cik=cik,
            form_type=form_type,
            user_agent=user_agent,
            raw_data_dir=raw_data_dir,
        )
    )

    if result is None:
        typer.echo(f"No {form_type} filing found for CIK {cik}.")
        raise typer.Exit(code=1)

    typer.echo(f"Status: {result.status}")
    typer.echo(f"Filing: {result.filing_path}")
    typer.echo(f"Metadata: {result.metadata_path}")
    typer.echo(f"SHA-256: {result.content_sha256}")


async def _ingest_latest(
    cik: str,
    form_type: str,
    user_agent: str,
    raw_data_dir: Path,
) -> StoredFiling | None:
    client = SECClient(cik=cik, user_agent=user_agent)
    store = RawFilingStore(raw_data_dir)
    ingestor = FilingIngestor(client=client, store=store)
    return await ingestor.ingest_latest(form_type)


def main() -> None:
    app()
