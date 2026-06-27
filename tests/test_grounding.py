from pydantic_ai.messages import ModelRequest, ToolReturnPart, UserPromptPart

from fedsearch.agent.grounding import answer_numbers, ungrounded


def _msgs(result: dict, question: str = "q") -> list:
    return [ModelRequest(parts=[UserPromptPart(question)]),
            ModelRequest(parts=[ToolReturnPart("run_sql", result, tool_call_id="1")])]


RESULT = {"columns": ["name", "overdue_usd", "share"], "rows": [["Cyberdyne", 74667, 0.8667], ["Oscorp", 3333.33, 0.04]],
          "row_count": 2, "systems": ["Billing"], "sql": "SELECT ... LIMIT 200", "elapsed_ms": 12}


def test_ids_and_dates_are_not_numbers():
    assert [m.text for m in answer_numbers("CUST-0007 hit E-SSO-112 on TCK-1090 (v2.1)")] == []


def test_grounded_with_display_rounding():
    text = "Cyberdyne owes **$74,667** (about $74.7k, 87% of the total); Oscorp $3,333. 2 customers."
    assert ungrounded(text, _msgs(RESULT)) == []


def test_flags_model_arithmetic():
    # 74,667 + 3,333.33 = 78,000 was never returned by a tool.
    assert ungrounded("Together they owe $78,000 across 2 customers.", _msgs(RESULT)) == ["$78,000"]


def test_numbers_from_question_are_allowed():
    assert ungrounded("Top 5 shown.", _msgs(RESULT, "show the top 5")) == []
