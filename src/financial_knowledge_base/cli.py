import asyncio
from pathlib import Path
from typing import Annotated

import typer

from .ai.openai_client import OpenAIClient
from .extraction.item2_extractor import Item2Extractor
from .extraction.proposal_store import ExtractionProposalStore
from .ingestion.models import IngestedFiling, ParsedDocument
from .ingestion.parsed_store import ParsedDocumentStore
from .ingestion.parser import SECHTMLParser
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
    """Download, archive, and parse the latest matching SEC filing."""

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

    typer.echo(f"Raw status: {result.raw.status}")
    typer.echo(f"Filing: {result.raw.filing_path}")
    typer.echo(f"Metadata: {result.raw.metadata_path}")
    typer.echo(f"SHA-256: {result.raw.content_sha256}")
    typer.echo(f"Parsed status: {result.parsed.status}")
    typer.echo(f"Document: {result.parsed.document_path}")
    typer.echo(f"Sections: {result.parsed.section_count}")
    typer.echo(f"Characters: {result.parsed.character_count}")


@app.command("parse")
def parse_filing(
    filing_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to an archived filing.html file.",
        ),
    ],
) -> None:
    """Regenerate document.json from an archived SEC filing."""

    parser = SECHTMLParser()
    parsed_store = ParsedDocumentStore()
    document = parser.parse(filing_path.read_bytes())
    result = parsed_store.save(filing_path=filing_path, document=document)

    typer.echo(f"Status: {result.status}")
    typer.echo(f"Document: {result.document_path}")
    typer.echo(f"Parser: {result.parser_version}")
    typer.echo(f"Sections: {result.section_count}")
    typer.echo(f"Characters: {result.character_count}")


@app.command("extract-item-2")
def extract_item_2(
    document_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to a parsed document.json file.",
        ),
    ],
    api_key: Annotated[
        str,
        typer.Option(
            envvar="OPENAI_API_KEY",
            help="OpenAI API key. Prefer the OPENAI_API_KEY environment variable.",
        ),
    ],
    model: Annotated[
        str,
        typer.Option(
            envvar="OPENAI_MODEL",
            help="OpenAI model that supports structured outputs.",
        ),
    ],
) -> None:
    """Propose grounded financial claims from Part I, Item 2."""

    document = ParsedDocument.model_validate_json(
        document_path.read_text(encoding="utf-8")
    )
    llm_client = OpenAIClient(api_key=api_key, model=model)
    extractor = Item2Extractor(llm_client)
    proposal = asyncio.run(extractor.extract(document))
    result = ExtractionProposalStore().save(
        document_path=document_path,
        proposal=proposal,
    )

    typer.echo(f"Status: {result.status}")
    typer.echo(f"Proposal: {result.proposal_path}")
    typer.echo(f"Claims: {result.claim_count}")
    typer.echo(f"Verified evidence: {result.verified_claim_count}")


async def _ingest_latest(
    cik: str,
    form_type: str,
    user_agent: str,
    raw_data_dir: Path,
) -> IngestedFiling | None:
    client = SECClient(cik=cik, user_agent=user_agent)
    raw_store = RawFilingStore(raw_data_dir)
    parser = SECHTMLParser()
    parsed_store = ParsedDocumentStore()
    ingestor = FilingIngestor(
        client=client,
        raw_store=raw_store,
        parser=parser,
        parsed_store=parsed_store,
    )
    return await ingestor.ingest_latest(form_type)


def main() -> None:
    app()
