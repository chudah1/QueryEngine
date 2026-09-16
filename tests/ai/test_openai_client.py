import json
from types import SimpleNamespace

import pytest

from financial_knowledge_base.ai.llm import (
    LLMResponseContentFilterError,
    LLMResponseTruncatedError,
)
from financial_knowledge_base.ai.openai_client import OpenAIClient
from financial_knowledge_base.extraction.models import Item2ModelOutput


def _raw_response(
    *,
    output_text: str,
    status: str = "completed",
    incomplete_reason: str | None = None,
    output_tokens: int = 20,
    reasoning_tokens: int = 10,
) -> SimpleNamespace:
    body = {
        "id": "response-1",
        "created_at": 0,
        "model": "test-model",
        "object": "response",
        "output": [
            {
                "id": "message-1",
                "type": "message",
                "status": status,
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": output_text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": status,
        "incomplete_details": (
            {"reason": incomplete_reason} if incomplete_reason else None
        ),
        "usage": {
            "input_tokens": 50,
            "input_tokens_details": {
                "cache_write_tokens": 0,
                "cached_tokens": 0,
            },
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
            "total_tokens": 50 + output_tokens,
        },
    }
    return SimpleNamespace(text=json.dumps(body))


@pytest.mark.asyncio
async def test_structured_response_uses_configured_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = OpenAIClient(
        api_key="test-key",
        model="test-model",
        reasoning_effort="low",
    )
    request: dict[str, object] = {}

    async def parse(**kwargs: object) -> SimpleNamespace:
        request.update(kwargs)
        return _raw_response(output_text='{"claims":[]}')

    async def automatic_parse(**_: object) -> None:
        raise AssertionError("automatic parser should not be used")

    monkeypatch.setattr(client._client.responses, "parse", automatic_parse)
    monkeypatch.setattr(client._client.responses.with_raw_response, "parse", parse)

    await client.generate_structured_response(
        system_prompt="Extract claims.",
        user_prompt="A filing slice.",
        output_model=Item2ModelOutput,
    )

    assert request["reasoning"] == {"effort": "low"}


@pytest.mark.asyncio
async def test_incomplete_response_reports_provider_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = OpenAIClient(api_key="test-key", model="test-model")

    async def parse(**_: object) -> SimpleNamespace:
        return _raw_response(
            output_text='{"claims":[',
            status="incomplete",
            incomplete_reason="max_output_tokens",
            output_tokens=16_000,
            reasoning_tokens=15_900,
        )

    async def automatic_parse(**_: object) -> None:
        raise AssertionError("automatic parser should not be used")

    monkeypatch.setattr(client._client.responses, "parse", automatic_parse)
    monkeypatch.setattr(client._client.responses.with_raw_response, "parse", parse)

    with pytest.raises(
        LLMResponseTruncatedError,
        match=(
            "status=incomplete.*reason=max_output_tokens.*"
            "output_tokens=16000.*reasoning_tokens=15900"
        ),
    ):
        await client.generate_structured_response(
            system_prompt="Extract claims.",
            user_prompt="A filing slice.",
            output_model=Item2ModelOutput,
        )


@pytest.mark.asyncio
async def test_content_filter_preserves_partial_response_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = OpenAIClient(api_key="test-key", model="test-model")

    async def parse(**_: object) -> SimpleNamespace:
        return _raw_response(
            output_text='{"claims":[',
            status="incomplete",
            incomplete_reason="content_filter",
            output_tokens=0,
            reasoning_tokens=0,
        )

    monkeypatch.setattr(client._client.responses.with_raw_response, "parse", parse)

    with pytest.raises(LLMResponseContentFilterError) as raised:
        await client.generate_structured_response(
            system_prompt="Extract claims.",
            user_prompt="A filing slice.",
            output_model=Item2ModelOutput,
        )

    assert raised.value.response_id == "response-1"
    assert raised.value.raw_response == '{"claims":['
    assert raised.value.model == "test-model"
