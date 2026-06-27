"""Runtime settings, overridable via env vars prefixed with ``FEDSEARCH_`` or a ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FEDSEARCH_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    index_dir: Path = Path("artifacts/index")
    sources_config: Path = Path("configs/sources.yaml")

    # Streaming
    bus: Literal["kafka", "memory"] = "memory"
    kafka_bootstrap: str = "localhost:9092"
    kafka_topic_prefix: str = "fedsearch.raw"
    kafka_group_id: str = "fedsearch-indexer"

    # Embeddings: "hashing" works fully offline; "fastembed" uses a real ONNX model.
    embedder: Literal["hashing", "fastembed"] = "hashing"
    fastembed_model: str = "BAAI/bge-small-en-v1.5"

    # LLM (any OpenAI-compatible server: vLLM in prod, Ollama/llama.cpp locally).
    # planner/answerer = "rules"/"extractive" disables the LLM entirely.
    planner: Literal["llm", "rules"] = "rules"
    answerer: Literal["llm", "extractive"] = "extractive"
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = "EMPTY"
    llm_model: str = "fedsearch-qwen3.8-27b"
    # "native" = JSON-schema response_format (vLLM guided decoding); "tool" = tool calling;
    # "prompted" = schema in the prompt, for servers without either.
    llm_output_mode: Literal["native", "tool", "prompted"] = "native"
    llm_temperature: float = 0.0

    top_k: int = 10

    # Aggregation agent (model-agnostic). A bare model id uses the OpenAI-compatible endpoint below
    # (OpenCode Go by default; also vLLM, Ollama, OpenRouter, ...). A "provider:model" id such as
    # "anthropic:claude-sonnet-5-5" or "openai:gpt-5.6" is handed to PydanticAI's own provider registry.
    agent_model: str = "deepseek-v4.1-flash"
    agent_base_url: str = "https://opencode.ai/inference/go/openai/v1"
    agent_api_key: str = ""
    agent_user_agent: str = "fedsearch-agent/0.1"
    catalog_config: Path = Path("configs/catalog.yaml")


@lru_cache
def get_settings() -> Settings:
    # Also export .env to os.environ: configs/*.yaml expand ${FEDSEARCH_*} from the process environment.
    load_dotenv(override=False)
    return Settings()
