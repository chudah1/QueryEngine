from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel


class LLMResponseError(RuntimeError):
    """Raised when a provider does not return a usable structured response."""


@dataclass(frozen=True)
class LLMResponse[OutputType: BaseModel]:
    """Provider-neutral structured response and execution metadata."""

    output: OutputType
    model: str
    raw_response: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_id: str | None = None


class LLMClient(Protocol):
    """Capability required by services that need structured model output."""

    async def generate_structured_response[OutputType: BaseModel](
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_model: type[OutputType],
    ) -> LLMResponse[OutputType]: ...
