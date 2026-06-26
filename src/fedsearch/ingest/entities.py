"""Entity linking: the glue that lets results be cross-referenced across systems.

Every document is tagged with canonical entity keys (``customer:CUST-0007``, ``ticket:TCK-1042``,
``error_code:E-SSO-112``, ``product:sso``). Aliases learned from one system (a CRM row says
``Globex`` == ``globex.com`` == ``CUST-0001``) are used to link mentions in every other system.
"""

from __future__ import annotations

import difflib
import re
from collections import defaultdict

ID_PATTERNS: dict[str, re.Pattern[str]] = {
    "ticket": re.compile(r"\bTCK-\d{3,6}\b", re.IGNORECASE),
    "error_code": re.compile(r"\bE-[A-Z]{2,4}-\d{3}\b", re.IGNORECASE),
    "customer": re.compile(r"\bCUST-\d{4}\b", re.IGNORECASE),
}
EMAIL = re.compile(r"\b[\w.+-]+@([\w-]+\.[\w.-]+)\b")


def key(etype: str, value: str) -> str:
    return f"{etype}:{value}"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


class EntityRegistry:
    """Alias -> canonical key, plus canonical key -> display name."""

    def __init__(self) -> None:
        self.aliases: dict[str, dict[str, str]] = defaultdict(dict)  # etype -> norm(alias) -> canonical id
        self.names: dict[str, str] = {}  # canonical key -> display name

    def add(self, etype: str, canonical: str, *aliases: str, display: str | None = None) -> None:
        self.aliases[etype][norm(canonical)] = canonical
        for a in aliases:
            if a:
                self.aliases[etype][norm(a)] = canonical
        if display:
            self.names[key(etype, canonical)] = display

    def resolve(self, etype: str, surface: str, fuzzy: bool = True) -> str | None:
        """Map a surface form (``globx``, ``SSO``, ``tck-1042``) to a canonical key."""
        table = self.aliases.get(etype, {})
        n = norm(surface)
        if n in table:
            return key(etype, table[n])
        if etype in ID_PATTERNS and ID_PATTERNS[etype].fullmatch(surface.strip()):
            return key(etype, surface.strip().upper())
        if fuzzy and table:
            match = difflib.get_close_matches(n, list(table), n=1, cutoff=0.8)
            if match:
                return key(etype, table[match[0]])
        return None

    def extract(self, text: str, types: tuple[str, ...] = ("customer", "product")) -> set[str]:
        """Find entity mentions in free text: IDs by regex, names by gazetteer, customers by email domain."""
        found: set[str] = set()
        for etype, pat in ID_PATTERNS.items():
            for m in pat.finditer(text):
                found.add(key(etype, m.group(0).upper()))
        for m in EMAIL.finditer(text):
            if (k := self.resolve("customer", m.group(1), fuzzy=False)) is not None:
                found.add(k)
        padded = f" {norm(text)} "
        for etype in types:
            for alias, canonical in self.aliases.get(etype, {}).items():
                if len(alias) >= 3 and f" {alias} " in padded:
                    found.add(key(etype, canonical))
        return found

    def to_dict(self) -> dict:
        return {"aliases": {k: dict(v) for k, v in self.aliases.items()}, "names": self.names}

    @classmethod
    def from_dict(cls, d: dict) -> EntityRegistry:
        r = cls()
        for etype, table in d.get("aliases", {}).items():
            r.aliases[etype].update(table)
        r.names.update(d.get("names", {}))
        return r
