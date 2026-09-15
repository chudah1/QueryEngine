import hashlib
import json
from pathlib import Path

from .item2_chunker import Item2Chunk
from .models import Item2ChunkCheckpoint


class Item2ChunkCheckpointConflictError(RuntimeError):
    """Raised when an immutable checkpoint path contains different content."""


class Item2ChunkCheckpointStore:
    """Persist successful chunk responses so an interrupted extraction can resume."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def load(
        self,
        *,
        chunk: Item2Chunk,
        prompt_version: str,
        prompt_sha256: str,
        source_content_sha256: str,
        text_content_sha256: str,
    ) -> Item2ChunkCheckpoint | None:
        path = self._path(
            chunk=chunk,
            prompt_version=prompt_version,
            prompt_sha256=prompt_sha256,
            source_content_sha256=source_content_sha256,
            text_content_sha256=text_content_sha256,
        )
        if not path.exists():
            return None
        return Item2ChunkCheckpoint.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    def save(
        self,
        *,
        checkpoint: Item2ChunkCheckpoint,
        chunk: Item2Chunk,
    ) -> Path:
        path = self._path(
            chunk=chunk,
            prompt_version=checkpoint.prompt_version,
            prompt_sha256=checkpoint.prompt_sha256,
            source_content_sha256=checkpoint.source_content_sha256,
            text_content_sha256=checkpoint.text_content_sha256,
        )
        serialized = checkpoint.model_dump_json(indent=2) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_text(encoding="utf-8") != serialized:
                raise Item2ChunkCheckpointConflictError(
                    f"Refusing to overwrite chunk checkpoint: {path}"
                )
            return path

        temporary_path = path.with_suffix(".json.tmp")
        temporary_path.write_text(serialized, encoding="utf-8")
        temporary_path.replace(path)
        return path

    def _path(
        self,
        *,
        chunk: Item2Chunk,
        prompt_version: str,
        prompt_sha256: str,
        source_content_sha256: str,
        text_content_sha256: str,
    ) -> Path:
        identity = json.dumps(
            {
                "chunk_id": chunk.chunk_id,
                "chunk_start": chunk.character_start,
                "chunk_end": chunk.character_end,
                "chunk_text_sha256": self.chunk_text_sha256(chunk),
                "prompt_version": prompt_version,
                "prompt_sha256": prompt_sha256,
                "source_content_sha256": source_content_sha256,
                "text_content_sha256": text_content_sha256,
            },
            sort_keys=True,
        )
        identity_sha256 = hashlib.sha256(identity.encode()).hexdigest()
        return self._directory / f"{chunk.chunk_id}-{identity_sha256[:16]}.json"

    @staticmethod
    def chunk_text_sha256(chunk: Item2Chunk) -> str:
        return hashlib.sha256(chunk.text.encode()).hexdigest()
