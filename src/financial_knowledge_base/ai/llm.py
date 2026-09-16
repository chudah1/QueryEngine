from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel


class LLMResponseError(RuntimeError):
    """Raised when a provider does not return a usable structured response."""


class LLMResponseTruncatedError(LLMResponseError):
    """Raised when a provider stops before completing structured output."""


class LLMResponseContentFilterError(LLMResponseError):
    """Raised when a provider content filter stops structured output."""

    def __init__(
        self,
        message: str,
        *,
        model: str,
        response_id: str | None,
        raw_response: str,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> None:
        super().__init__(message)
        self.model = model
        self.response_id = response_id
        self.raw_response = raw_response
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


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
