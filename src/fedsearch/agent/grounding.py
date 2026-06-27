"""Numeric grounding: every number in an answer must come from a tool result (or the question).

The agent is told never to do arithmetic itself; this check enforces it. A number counts as grounded when
some value returned by a tool matches it up to the precision the answer displays ("$74.7k" matches
74,667; "33%" matches 0.333 or 33.3). Identifiers such as CUST-0007, E-SSO-112 or TCK-1090 are not numbers.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pydantic_ai.messages import ModelMessage, ToolReturnPart, UserPromptPart

# A number in prose: not glued to a letter, hyphen, dot or underscore (that would be an id or a version).
_ANSWER_NUM = re.compile(
    r"(?<![\w.\-/#])(?P<neg>[-−])?\$?(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?P<dec>\.\d+)?"
    r"(?:\s?(?P<suffix>%|[kK]\b|[mM]\b|bn\b|B\b))?(?![\w\-])"
)
# Any number inside tool output text (dates, snippets, ids are all fair game here).
_SOURCE_NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")

_SCALE = {"k": 1e3, "K": 1e3, "m": 1e6, "M": 1e6, "bn": 1e9, "B": 1e9}


@dataclass(frozen=True)
class Mention:
    text: str
    value: float
    tolerance: float
    percent: bool


def answer_numbers(text: str) -> list[Mention]:
    out = []
    for m in _ANSWER_NUM.finditer(text):
        digits = m["num"].replace(",", "")
        decimals = len(m["dec"]) - 1 if m["dec"] else 0
        value = float(digits + (m["dec"] or ""))
        scale = _SCALE.get(m["suffix"] or "", 1.0)
        sign = -1.0 if m["neg"] else 1.0
        out.append(Mention(m.group(0).strip(), sign * value * scale, 0.5 * 10 ** -decimals * scale + 1e-9,
                           m["suffix"] == "%"))
    return out


def _walk(obj: Any) -> Iterable[float]:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            yield float(obj)
    elif isinstance(obj, str):
        for s in _SOURCE_NUM.findall(obj):
            yield float(s.replace(",", ""))
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)


def source_values(messages: list[ModelMessage]) -> set[float]:
    """Numbers the model was given: tool results and what the user typed."""
    vals: set[float] = set()
    for msg in messages:
        for part in msg.parts:
            if isinstance(part, ToolReturnPart):
                content = part.content
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except ValueError:
                        pass
                vals.update(_walk(content))
            elif isinstance(part, UserPromptPart):
                vals.update(_walk(part.content if isinstance(part.content, str) else str(part.content)))
    return vals


def _matches(m: Mention, sources: set[float]) -> bool:
    candidates = [abs(m.value)]
    if m.percent:
        candidates.append(abs(m.value) / 100)  # "33%" may come from a 0.333 ratio
    for c in candidates:
        tol = m.tolerance / 100 if (m.percent and c != abs(m.value)) else m.tolerance
        if any(abs(abs(v) - c) <= tol for v in sources):
            return True
    return False


def ungrounded(text: str, messages: list[ModelMessage], extra: Iterable[float] = ()) -> list[str]:
    sources = source_values(messages) | set(extra)
    seen, bad = set(), []
    for m in answer_numbers(text):
        if m.text not in seen and not _matches(m, sources):
            seen.add(m.text)
            bad.append(m.text)
    return bad
