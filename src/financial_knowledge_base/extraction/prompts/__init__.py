import hashlib
from dataclasses import dataclass
from importlib.resources import files


@dataclass(frozen=True)
class PromptDefinition:
    """Immutable prompt text and the metadata needed to reproduce a run."""

    version: str
    text: str
    sha256: str


def load_item2_prompt(version: str) -> PromptDefinition:
    """Load a versioned Item 2 prompt bundled with the application."""

    prompt_path = files(__package__).joinpath(f"{version}.txt")
    if not prompt_path.is_file():
        raise ValueError(f"Unknown Item 2 prompt version: {version}")

    text = prompt_path.read_text(encoding="utf-8")
    return PromptDefinition(
        version=version,
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
