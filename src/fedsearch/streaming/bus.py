"""Message bus between connectors (producers) and the indexer (consumer).

``KafkaBus`` is the production path. ``MemoryBus`` has the same semantics (topics, keys, ordered
per-topic delivery, consumer offsets) and lets the whole pipeline run in tests without a broker.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator

from fedsearch.schemas import Record


def topic_for(prefix: str, source: str, stream: str) -> str:
    return f"{prefix}.{source}.{stream}"


class Bus(ABC):
    def __init__(self, prefix: str):
        self.prefix = prefix

    def publish_record(self, rec: Record, key: str | None) -> None:
        self.publish(topic_for(self.prefix, rec.source, rec.stream), key, rec.model_dump_json().encode())

    def consume_records(self, idle_timeout: float = 5.0) -> Iterator[Record]:
        for value in self.consume(rf"^{re.escape(self.prefix)}\..+", idle_timeout):
            yield Record.model_validate_json(value)

    @abstractmethod
    def publish(self, topic: str, key: str | None, value: bytes) -> None: ...

    @abstractmethod
    def flush(self) -> None: ...

    @abstractmethod
    def consume(self, topic_pattern: str, idle_timeout: float) -> Iterator[bytes]:
        """Yield values from all matching topics; stop after ``idle_timeout`` seconds with no messages."""


class MemoryBus(Bus):
    def __init__(self, prefix: str):
        super().__init__(prefix)
        self.topics: dict[str, list[bytes]] = defaultdict(list)
        self.offsets: dict[str, int] = defaultdict(int)

    def publish(self, topic: str, key: str | None, value: bytes) -> None:
        self.topics[topic].append(value)

    def flush(self) -> None:
        pass

    def consume(self, topic_pattern: str, idle_timeout: float) -> Iterator[bytes]:
        pat = re.compile(topic_pattern)
        for topic in sorted(self.topics):
            if pat.match(topic):
                msgs = self.topics[topic]
                while self.offsets[topic] < len(msgs):
                    yield msgs[self.offsets[topic]]
                    self.offsets[topic] += 1


class KafkaBus(Bus):
    def __init__(self, prefix: str, bootstrap: str, group_id: str):
        super().__init__(prefix)
        from confluent_kafka import Producer

        self.bootstrap = bootstrap
        self.group_id = group_id
        self._producer = Producer({
            "bootstrap.servers": bootstrap,
            "enable.idempotence": True,
            "compression.type": "zstd",
            "linger.ms": 20,
        })

    def publish(self, topic: str, key: str | None, value: bytes) -> None:
        while True:
            try:
                self._producer.produce(topic, key=key.encode() if key else None, value=value)
                break
            except BufferError:
                self._producer.poll(0.5)
        self._producer.poll(0)

    def flush(self) -> None:
        self._producer.flush(30)

    def consume(self, topic_pattern: str, idle_timeout: float) -> Iterator[bytes]:
        import time

        from confluent_kafka import Consumer, KafkaError

        consumer = Consumer({
            "bootstrap.servers": self.bootstrap,
            "group.id": self.group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "topic.metadata.refresh.interval.ms": 2000,
        })
        consumer.subscribe([topic_pattern])  # leading ^ => regex subscription
        last = time.monotonic()
        try:
            while time.monotonic() - last < idle_timeout:
                msg = consumer.poll(0.5)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    raise RuntimeError(msg.error())
                last = time.monotonic()
                yield msg.value()
                consumer.commit(msg, asynchronous=True)
        finally:
            consumer.close()


def make_bus(settings) -> Bus:
    if settings.bus == "kafka":
        return KafkaBus(settings.kafka_topic_prefix, settings.kafka_bootstrap, settings.kafka_group_id)
    return MemoryBus(settings.kafka_topic_prefix)


def encode_key(rec: Record, primary_key: str | None) -> str | None:
    if primary_key and primary_key in rec.data:
        return f"{rec.source}:{rec.stream}:{rec.data[primary_key]}"
    return None


__all__ = ["Bus", "KafkaBus", "MemoryBus", "encode_key", "make_bus", "topic_for"]
