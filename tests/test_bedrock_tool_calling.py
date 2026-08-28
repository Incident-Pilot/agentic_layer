"""Standalone, credential-gated verification that Kimi K2.5 via Bedrock's
Mantle gateway actually supports OpenAI-style function calling.

Not run as part of the normal suite -- skipped unless AWS_BEARER_TOKEN_BEDROCK
is set, since it makes a real network call to a real (billed) endpoint. This
exists because a plain chat completion succeeding says nothing about tool
calling, which is the one thing the whole investigation pipeline depends on
(agents/tool_loop.py drives every graph node through tools=[...]). If this
test ever starts failing, do not wire --llm bedrock in further before
finding out why -- see the task notes this test was added for.

Run explicitly with: pytest tests/test_bedrock_tool_calling.py -v
"""

import os

import pytest
from openai import AsyncOpenAI

from incident_pilot_agent import config

requires_bedrock_credentials = pytest.mark.skipif(
    not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"),
    reason="requires a live AWS_BEARER_TOKEN_BEDROCK -- makes a real call to the Bedrock Mantle gateway",
)

_GET_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}


@requires_bedrock_credentials
async def test_bedrock_kimi_supports_tool_calling():
    client = AsyncOpenAI(api_key=config.BEDROCK_API_KEY, base_url=config.BEDROCK_BASE_URL)

    response = await client.chat.completions.create(
        model=config.BEDROCK_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Paris?"},
        ],
        tools=[_GET_WEATHER_TOOL],
        max_tokens=512,
    )

    choice = response.choices[0]
    tool_calls = choice.message.tool_calls or []

    assert tool_calls, (
        f"expected a tool_calls entry invoking get_weather, got none -- "
        f"finish_reason={choice.finish_reason!r} content={choice.message.content!r}"
    )
    assert tool_calls[0].function.name == "get_weather"
    assert "Paris" in tool_calls[0].function.arguments
