"""Run any of Airbyte's 350+ source connectors in-process via PyAirbyte (``pip install fedsearch[airbyte]``).

Example ``sources.yaml`` entry::

    github:
      connector: airbyte
      airbyte_source: source-github
      config: {repositories: ["org/repo"], credentials: {personal_access_token: "${GITHUB_TOKEN}"}}
      streams:
        issues: {primary_key: id, title: "{title}", text: "{body}"}

For production you can instead run the Airbyte platform with its Kafka destination writing to the same
``fedsearch.raw.<source>.<stream>`` topics; the indexer consumes them identically.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from fedsearch.connectors.base import Connector


class AirbyteConnector(Connector):
    kind = "airbyte"

    def __init__(self, source: str, config: dict[str, Any]):
        super().__init__(source, config)
        try:
            import airbyte as ab
        except ImportError as e:  # pragma: no cover
            raise ImportError("Install the airbyte extra: uv sync --extra airbyte") from e
        self._source = ab.get_source(config["airbyte_source"], config=config.get("config", {}), install_if_missing=True)
        self._source.check()
        self._source.select_streams(list(self.streams))
        self._result = None

    def read_stream(self, stream: str, spec: dict[str, Any], state: dict[str, Any]) -> Iterator[dict[str, Any]]:
        if self._result is None:
            self._result = self._source.read()  # PyAirbyte manages incremental state in its local cache
        for rec in self._result[stream]:
            yield dict(rec)
