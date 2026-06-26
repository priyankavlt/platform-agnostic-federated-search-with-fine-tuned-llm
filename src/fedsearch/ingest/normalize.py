"""Turn raw connector records into source-agnostic :class:`Document` objects, driven by ``sources.yaml``.

Per-stream spec keys used here::

    primary_key: id
    title: "{subject}"                 # str.format over the record
    text: "{description}"
    url: "https://support.example.com/tickets/{id}"
    entity_fields: {customer: customer_id, product: product}   # fields that already hold canonical ids
    aliases: {type: customer, id: id, fields: [name, domain], display: name}  # teach the registry
    extract_entities: true             # also run regex/gazetteer linking over title+text
"""

from __future__ import annotations

from typing import Any

from fedsearch.ingest.entities import EntityRegistry, key
from fedsearch.schemas import Document, Record


class _SafeDict(dict):
    def __missing__(self, k: str) -> str:
        return ""


def _fmt(template: str | None, data: dict[str, Any]) -> str:
    if not template:
        return ""
    return template.format_map(_SafeDict({k: ("" if v is None else v) for k, v in data.items()}))


class Normalizer:
    def __init__(self, sources_config: dict[str, Any], registry: EntityRegistry):
        self.cfg = sources_config["sources"]
        self.registry = registry
        for etype, entries in sources_config.get("gazetteer", {}).items():
            for canonical, spec in entries.items():
                self.registry.add(etype, canonical, spec.get("name", ""), *spec.get("aliases", []),
                                  display=spec.get("name", canonical))

    def spec(self, source: str, stream: str) -> dict[str, Any]:
        return self.cfg[source]["streams"][stream]

    def learn_aliases(self, rec: Record) -> None:
        a = self.spec(rec.source, rec.stream).get("aliases")
        if a:
            canonical = str(rec.data[a["id"]])
            self.registry.add(a["type"], canonical, *(str(rec.data.get(f, "")) for f in a.get("fields", [])),
                              display=str(rec.data.get(a.get("display", a["id"]), canonical)))

    def to_document(self, rec: Record) -> Document:
        spec = self.spec(rec.source, rec.stream)
        d = rec.data
        pk = str(d[spec.get("primary_key", "id")])
        title = _fmt(spec.get("title"), d) or pk
        text = _fmt(spec.get("text"), d) or " ".join(str(v) for v in d.values() if isinstance(v, str))
        ents = {key(etype, str(d[f])) for etype, f in spec.get("entity_fields", {}).items() if d.get(f)}
        return Document(
            id=f"{rec.source}:{rec.stream}:{pk}",
            source=rec.source,
            stream=rec.stream,
            title=title,
            text=text,
            url=_fmt(spec.get("url"), d) or None,
            metadata={k: v for k, v in d.items() if k in spec.get("metadata_fields", [])},
            entities=sorted(ents),
        )

    def link(self, doc: Document) -> Document:
        """Add entities found in free text. Run after aliases from all sources are known."""
        if self.spec(doc.source, doc.stream).get("extract_entities", True):
            doc.entities = sorted(set(doc.entities) | self.registry.extract(f"{doc.title}\n{doc.text}"))
        return doc
