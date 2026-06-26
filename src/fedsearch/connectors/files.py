"""Local or mounted files: Markdown/text documents and JSONL/JSON exports."""

from __future__ import annotations

import glob
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fedsearch.connectors.base import Connector


class FilesConnector(Connector):
    kind = "files"

    def read_stream(self, stream: str, spec: dict[str, Any], state: dict[str, Any]) -> Iterator[dict[str, Any]]:
        root = Path(spec.get("root", "."))
        fmt = spec.get("format", "text")
        for name in sorted(glob.glob(str(root / spec["glob"]), recursive=True)):
            path = Path(name)
            if fmt == "jsonl":
                with path.open() as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line)
            elif fmt == "json":
                data = json.loads(path.read_text())
                yield from (data if isinstance(data, list) else [data])
            else:
                body = path.read_text()
                first = body.lstrip().splitlines()[0] if body.strip() else path.stem
                yield {
                    "id": path.stem,
                    "path": str(path.relative_to(root)),
                    "title": first.lstrip("# ").strip(),
                    "body": body,
                }
