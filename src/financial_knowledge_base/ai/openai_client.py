from typing import cast

from openai import AsyncOpenAI
from openai.types.responses import Response
from openai.types.shared import ReasoningEffort
from pydantic import BaseModel, ValidationError

from .llm import (
    LLMResponse,
    LLMResponseContentFilterError,
    LLMResponseError,
    LLMResponseTruncatedError,
)


class OpenAIClient:
    """OpenAI implementation of the structured-output LLM capability."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_output_tokens: int = 16_000,
        reasoning_effort: str = "low",
    ) -> None:
        supported_efforts = {
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }
        if reasoning_effort not in supported_efforts:
            raise ValueError(
                "reasoning_effort must be one of "
                f"{', '.join(sorted(supported_efforts))}"
            )
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._reasoning_effort = cast(ReasoningEffort, reasoning_effort)

    async def generate_structured_response[OutputType: BaseModel](
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_model: type[OutputType],
    ) -> LLMResponse[OutputType]:
        raw_response = await self._client.responses.with_raw_response.parse(
            model=self._model,
            instructions=system_prompt,
            input=user_prompt,
            text_format=output_model,
            store=False,
            max_output_tokens=self._max_output_tokens,
            reasoning={"effort": self._reasoning_effort},
        )
        response = Response.model_validate_json(raw_response.text)
        output_text = response.output_text
        diagnostics = self._diagnostics(response, output_text)

        if (
            response.status == "incomplete"
            and response.incomplete_details is not None
            and response.incomplete_details.reason == "content_filter"
        ):
            usage = response.usage
            raise LLMResponseContentFilterError(
                f"OpenAI filtered the structured response ({diagnostics})",
                model=response.model,
                response_id=response.id,
                raw_response=output_text,
                input_tokens=usage.input_tokens if usage is not None else None,
                output_tokens=usage.output_tokens if usage is not None else None,
            )
        if response.status == "incomplete":
            raise LLMResponseTruncatedError(
                f"OpenAI returned an incomplete response ({diagnostics})"
            )
        if response.status != "completed":
            raise LLMResponseError(
                f"OpenAI did not complete the response ({diagnostics})"
            )
        try:
            output = output_model.model_validate_json(output_text)
        except ValidationError as error:
            raise LLMResponseError(
                f"OpenAI returned invalid structured output ({diagnostics})"
            ) from error

        usage = response.usage
        return LLMResponse(
            output=output,
            model=response.model,
            raw_response=response.output_text,
            input_tokens=usage.input_tokens if usage is not None else None,
            output_tokens=usage.output_tokens if usage is not None else None,
            response_id=response.id,
        )

    def _diagnostics(self, response: Response, output_text: str) -> str:
        reason = (
            response.incomplete_details.reason
            if response.incomplete_details is not None
            else None
        )
        usage = response.usage
        output_tokens = usage.output_tokens if usage is not None else None
        reasoning_tokens = (
            usage.output_tokens_details.reasoning_tokens
            if usage is not None
            else None
        )
        return (
            f"response_id={response.id}, status={response.status}, reason={reason}, "
            f"output_chars={len(output_text)}, output_tokens={output_tokens}, "
            f"reasoning_tokens={reasoning_tokens}"
        )
