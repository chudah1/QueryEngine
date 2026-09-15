from openai import AsyncOpenAI
from pydantic import BaseModel

from .llm import LLMResponse, LLMResponseError


class OpenAIClient:
    """OpenAI implementation of the structured-output LLM capability."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_output_tokens: int = 16_000,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._max_output_tokens = max_output_tokens

    async def generate_structured_response[OutputType: BaseModel](
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_model: type[OutputType],
    ) -> LLMResponse[OutputType]:
        response = await self._client.responses.parse(
            model=self._model,
            instructions=system_prompt,
            input=user_prompt,
            text_format=output_model,
            store=False,
            max_output_tokens=self._max_output_tokens,
        )

        output = response.output_parsed
        if output is None:
            raise LLMResponseError(
                "OpenAI returned no parsed output "
                f"(response_id={response.id}, status={response.status})"
            )

        usage = response.usage
        return LLMResponse(
            output=output,
            model=response.model,
            raw_response=response.output_text,
            input_tokens=usage.input_tokens if usage is not None else None,
            output_tokens=usage.output_tokens if usage is not None else None,
            response_id=response.id,
        )
