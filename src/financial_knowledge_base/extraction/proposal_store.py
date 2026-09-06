import re
from pathlib import Path
from typing import Literal

from .models import Item2ExtractionProposal, StoredExtractionProposal


class ExtractionProposalConflictError(RuntimeError):
    """Raised when an immutable proposal path contains different content."""


class ExtractionProposalStore:
    """Store immutable, versioned Item 2 extraction proposals."""

    def save(
        self,
        *,
        document_path: Path,
        proposal: Item2ExtractionProposal,
    ) -> StoredExtractionProposal:
        proposal_path = self._proposal_path(document_path, proposal)
        serialized = proposal.model_dump_json(indent=2) + "\n"
        status: Literal["stored", "unchanged"] = "stored"

        proposal_path.parent.mkdir(parents=True, exist_ok=True)
        if proposal_path.exists():
            if proposal_path.read_text(encoding="utf-8") != serialized:
                raise ExtractionProposalConflictError(
                    f"Refusing to overwrite extraction proposal: {proposal_path}"
                )
            status = "unchanged"
        else:
            temporary_path = proposal_path.with_suffix(".json.tmp")
            temporary_path.write_text(serialized, encoding="utf-8")
            temporary_path.replace(proposal_path)

        verified_claim_count = sum(
            claim.evidence_status == "verified" for claim in proposal.claims
        )
        return StoredExtractionProposal(
            status=status,
            proposal_path=proposal_path,
            claim_count=len(proposal.claims),
            verified_claim_count=verified_claim_count,
        )

    def _proposal_path(
        self,
        document_path: Path,
        proposal: Item2ExtractionProposal,
    ) -> Path:
        identifier = proposal.response_id or proposal.generated_at.strftime(
            "%Y%m%dT%H%M%S.%fZ"
        )
        safe_identifier = re.sub(r"[^A-Za-z0-9._-]", "-", identifier)
        return (
            document_path.parent
            / "extractions"
            / "item-2"
            / proposal.prompt_version
            / f"{safe_identifier}.json"
        )
