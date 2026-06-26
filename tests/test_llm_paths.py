"""LLM code paths with PydanticAI FunctionModel standing in for vLLM."""

import json

from pydantic_ai import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from fedsearch.config import Settings
from fedsearch.schemas import Document, SearchHit
from fedsearch.search.answer import LLMAnswerer
from fedsearch.search.planner import LLMPlanner


def _respond(payload: dict, info: AgentInfo) -> ModelResponse:
    if info.output_tools:  # tool output mode
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])
    return ModelResponse(parts=[TextPart(json.dumps(payload))])


async def test_llm_planner_returns_validated_plan():
    plan = {"intent": "troubleshoot", "sources": ["docs", "support"],
            "entities": [{"type": "error_code", "value": "E-SSO-112"}], "keywords": ["saml"],
            "rewritten_query": "fix E-SSO-112"}
    for mode in ("native", "tool", "prompted"):
        planner = LLMPlanner(Settings(llm_output_mode=mode), model=FunctionModel(lambda m, i: _respond(plan, i)))
        out = await planner.plan("how to fix E-SSO-112")
        assert out.intent == "troubleshoot" and out.entities[0].value == "E-SSO-112"


async def test_answerer_retries_on_hallucinated_citation():
    doc = Document(id="docs:pages:x", source="docs", stream="pages", title="Runbook", text="Rotate the export token.")
    hits = [SearchHit(doc=doc, score=1.0)]
    calls = []

    def model(messages, info):
        calls.append(1)
        bad = {"answer": "Reboot it.", "citations": [{"doc_id": "docs:pages:x", "quote": "reboot the server"}]}
        good = {"answer": "Rotate the export token [docs:pages:x].",
                "citations": [{"doc_id": "docs:pages:x", "quote": "Rotate the export token"}]}
        return _respond(bad if len(calls) == 1 else good, info)

    ans = await LLMAnswerer(Settings(), model=FunctionModel(model)).answer("fix?", hits)
    assert len(calls) == 2  # first answer rejected by the citation validator, model re-asked
    assert ans.citations[0].quote == "Rotate the export token"
