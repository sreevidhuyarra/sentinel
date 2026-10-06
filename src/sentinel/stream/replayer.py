"""Replay recorded flows into net.flows at a fixed rate, in time order (report section 3.1).

Keyed by source IP, so one host's flows stay ordered within a partition.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Any, Protocol

import polars as pl

from sentinel.common.logging import get_logger
from sentinel.stream import messages
from sentinel.stream.metrics import REPLAYED

log = get_logger(__name__)


class Producer(Protocol):
    def produce(self, topic: str, value: bytes, key: bytes | None = None) -> None: ...
    def poll(self, timeout: float) -> int: ...
    def flush(self, timeout: float) -> int: ...


def select_flows(
    frame: pl.DataFrame, split: str | None, days: list[str] | None, limit: int | None
) -> pl.DataFrame:
    if split:
        frame = frame.filter(pl.col("split") == split)
    if days:
        frame = frame.filter(pl.col("day").is_in([d.lower() for d in days]))
    frame = frame.sort("ts")
    return frame.head(limit) if limit else frame


def replay(
    producer: Producer,
    rows: Iterable[dict[str, Any]],
    inputs: list[str],
    topic: str,
    rate: float,
    batch: int = 100,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Publish `rows` at about `rate` flows/s (0 = as fast as possible); returns the count."""
    n, t0 = 0, clock()
    for row in rows:
        producer.produce(topic, messages.encode(row, inputs), key=str(row.get("src_ip")).encode())
        n += 1
        if n % batch == 0:
            producer.poll(0)
            REPLAYED.inc(batch)
            if rate > 0:
                ahead = n / rate - (clock() - t0)
                if ahead > 0:
                    sleep(ahead)
            if n % (batch * 50) == 0:
                log.info("replayed %d flows (%.0f/s)", n, n / max(clock() - t0, 1e-9))
    REPLAYED.inc(n % batch)
    producer.flush(30)
    return n
