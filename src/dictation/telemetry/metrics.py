"""Local routing metrics for dictation runtime decisions."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class RoutingMetrics:
    """Writes per-utterance routing events to a local JSONL file."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (Path.home() / ".wisper-genie" / "routing_metrics.jsonl")
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def record(self, event: dict) -> None:
        payload = {"timestamp": time.time(), **event}
        line = json.dumps(payload, sort_keys=True)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def read_recent(self, limit: int = 50) -> list[dict]:
        if limit <= 0:
            return []
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").splitlines()[-limit:]
        events: list[dict] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events
