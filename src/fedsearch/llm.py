"""PydanticAI model factory for any OpenAI-compatible endpoint (vLLM, Ollama, llama.cpp, LM Studio)."""

from __future__ import annotations

from pydantic_ai import NativeOutput, PromptedOutput, ToolOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from fedsearch.config import Settings


def make_model(settings: Settings) -> OpenAIChatModel:
    provider = OpenAIProvider(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    return OpenAIChatModel(settings.llm_model, provider=provider)


def model_settings(settings: Settings, max_tokens: int = 1024) -> ModelSettings:
    return ModelSettings(
        temperature=settings.llm_temperature,
        max_tokens=max_tokens,
        # Qwen3.x: skip the <think> block for low-latency structured outputs (honoured by vLLM chat templates).
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def output_spec(settings: Settings, schema: type):
    """How to obtain schema-validated output from the server.

    native   -> response_format=json_schema; vLLM enforces it with guided decoding (recommended)
    tool     -> function calling (vLLM needs --enable-auto-tool-choice --tool-call-parser hermes)
    prompted -> schema in the prompt, validated + retried client-side
    """
    return {"native": NativeOutput, "tool": ToolOutput, "prompted": PromptedOutput}[settings.llm_output_mode](schema)
