from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .models import (
    GoldItem2Case,
    Item2EvaluationReport,
    Item2Review,
    StoredEvaluationArtifact,
)


class EvaluationArtifactConflictError(RuntimeError):
    """Raised when an immutable evaluation artifact would be overwritten."""


class EvaluationArtifactStore:
    """Store review templates, gold cases, and reports with stable paths."""

    def save_review(
        self,
        *,
        proposal_path: Path,
        review: Item2Review,
    ) -> StoredEvaluationArtifact:
        path = proposal_path.with_name(f"{proposal_path.stem}.review.json")
        return self._save_immutable(path, review)

    def save_gold_case(
        self,
        *,
        evaluation_directory: Path,
        case_name: str,
        gold_case: GoldItem2Case,
    ) -> StoredEvaluationArtifact:
        path = evaluation_directory / f"{case_name}.json"
        return self._save_immutable(path, gold_case)

    def save_report(
        self,
        *,
        proposal_path: Path,
        case_name: str,
        report: Item2EvaluationReport,
    ) -> StoredEvaluationArtifact:
        path = (
            proposal_path.parent
            / "evaluations"
            / case_name
            / report.schema_version
            / f"{proposal_path.stem}.json"
        )
        return self._save_immutable(path, report)

    def _save_immutable(
        self,
        path: Path,
        artifact: BaseModel,
    ) -> StoredEvaluationArtifact:
        serialized = artifact.model_dump_json(indent=2) + "\n"
        status: Literal["stored", "unchanged"] = "stored"
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            if path.read_text(encoding="utf-8") != serialized:
                raise EvaluationArtifactConflictError(
                    f"Refusing to overwrite evaluation artifact: {path}"
                )
            status = "unchanged"
        else:
            temporary_path = path.with_suffix(".json.tmp")
            temporary_path.write_text(serialized, encoding="utf-8")
            temporary_path.replace(path)

        return StoredEvaluationArtifact(status=status, path=path)
