import asyncio
from pathlib import Path
from typing import Annotated

import typer

from .ai.openai_client import OpenAIClient
from .evaluation.evaluator import Item2Evaluator
from .evaluation.models import GoldItem2Case, Item2Review
from .evaluation.review import Item2ReviewService
from .evaluation.store import EvaluationArtifactStore
from .extraction.checkpoint_store import Item2ChunkCheckpointStore
from .extraction.item2_extractor import Item2Extractor
from .extraction.models import Item2ExtractionProposal
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
    max_output_tokens: Annotated[
        int,
        typer.Option(
            envvar="OPENAI_MAX_OUTPUT_TOKENS",
            min=1_024,
            help="Maximum model output tokens for each Item 2 chunk.",
        ),
    ] = 16_000,
) -> None:
    """Propose grounded financial claims from Part I, Item 2."""

    document = ParsedDocument.model_validate_json(
        document_path.read_text(encoding="utf-8")
    )
    llm_client = OpenAIClient(
        api_key=api_key,
        model=model,
        max_output_tokens=max_output_tokens,
    )
    checkpoint_directory = (
        document_path.parent
        / "extractions"
        / "item-2"
        / Item2Extractor.PROMPT_VERSION
        / "checkpoints"
    )
    extractor = Item2Extractor(
        llm_client,
        checkpoint_store=Item2ChunkCheckpointStore(checkpoint_directory),
    )
    proposal = asyncio.run(extractor.extract(document))
    result = ExtractionProposalStore().save(
        document_path=document_path,
        proposal=proposal,
    )

    typer.echo(f"Status: {result.status}")
    typer.echo(f"Proposal: {result.proposal_path}")
    typer.echo(f"Claims: {result.claim_count}")
    typer.echo(f"Verified evidence: {result.verified_claim_count}")


@app.command("create-item-2-review")
def create_item_2_review(
    proposal_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to an Item 2 extraction proposal.",
        ),
    ],
    reviewer: Annotated[
        str,
        typer.Option(
            envvar="EVAL_REVIEWER",
            help="Name or identifier for the human reviewer.",
        ),
    ],
) -> None:
    """Create an editable human-review template for a proposal."""

    proposal = Item2ExtractionProposal.model_validate_json(
        proposal_path.read_text(encoding="utf-8")
    )
    review = Item2ReviewService().create_template(
        proposal=proposal,
        reviewer=reviewer,
    )
    result = EvaluationArtifactStore().save_review(
        proposal_path=proposal_path,
        review=review,
    )

    typer.echo(f"Status: {result.status}")
    typer.echo(f"Review: {result.path}")
    typer.echo(f"Claims to review: {len(review.claim_reviews)}")
    typer.echo(
        "Set each claim decision, add any missed_claims, then finalize the review."
    )


@app.command("finalize-item-2-review")
def finalize_item_2_review(
    proposal_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the reviewed proposal.",
        ),
    ],
    review_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the completed review JSON.",
        ),
    ],
    document_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the source document.json.",
        ),
    ],
    case_name: Annotated[
        str,
        typer.Option(help="Stable lowercase name for the evaluation case."),
    ],
    evaluation_directory: Annotated[
        Path,
        typer.Option(help="Directory for checked-in gold evaluation cases."),
    ] = Path("evals/item-2"),
) -> None:
    """Validate a completed review and create a reusable gold case."""

    proposal = Item2ExtractionProposal.model_validate_json(
        proposal_path.read_text(encoding="utf-8")
    )
    review = Item2Review.model_validate_json(review_path.read_text(encoding="utf-8"))
    document = ParsedDocument.model_validate_json(
        document_path.read_text(encoding="utf-8")
    )
    gold_case = Item2ReviewService().finalize(
        case_name=case_name,
        proposal=proposal,
        review=review,
        document=document,
    )
    result = EvaluationArtifactStore().save_gold_case(
        evaluation_directory=evaluation_directory,
        case_name=case_name,
        gold_case=gold_case,
    )

    typer.echo(f"Status: {result.status}")
    typer.echo(f"Gold case: {result.path}")
    typer.echo(f"Expected claims: {len(gold_case.expected_claims)}")
    typer.echo(f"Accepted: {gold_case.review_summary.accepted}")
    typer.echo(f"Rejected: {gold_case.review_summary.rejected}")
    typer.echo(f"Edited: {gold_case.review_summary.needs_edit}")
    typer.echo(f"Missed: {gold_case.review_summary.missed}")


@app.command("evaluate-item-2")
def evaluate_item_2(
    proposal_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to an Item 2 extraction proposal.",
        ),
    ],
    gold_case_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to a finalized Item 2 gold case.",
        ),
    ],
) -> None:
    """Evaluate an Item 2 proposal against human-labelled claims."""

    proposal = Item2ExtractionProposal.model_validate_json(
        proposal_path.read_text(encoding="utf-8")
    )
    gold_case = GoldItem2Case.model_validate_json(
        gold_case_path.read_text(encoding="utf-8")
    )
    report = Item2Evaluator().evaluate(proposal=proposal, gold_case=gold_case)
    result = EvaluationArtifactStore().save_report(
        proposal_path=proposal_path,
        case_name=gold_case.case_name,
        report=report,
    )

    typer.echo(f"Status: {result.status}")
    typer.echo(f"Report: {result.path}")
    typer.echo(f"Precision: {_format_metric(report.claim_precision)}")
    typer.echo(f"Recall: {_format_metric(report.claim_recall)}")
    typer.echo(f"F1: {_format_metric(report.claim_f1)}")
    typer.echo(f"Claim type accuracy: {_format_metric(report.claim_type_accuracy)}")
    typer.echo(f"Exact evidence matches: {report.exact_evidence_match_count}")
    typer.echo(f"Evidence-overlap matches: {report.evidence_overlap_match_count}")
    typer.echo(f"Field-similarity matches: {report.field_similarity_match_count}")
    typer.echo(
        f"Evidence verification: {_format_metric(report.evidence_verification_rate)}"
    )
    typer.echo(f"Ungrounded claims: {_format_metric(report.ungrounded_claim_rate)}")
    typer.echo(f"Unmatched claims: {_format_metric(report.unmatched_claim_rate)}")


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


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def main() -> None:
    app()
