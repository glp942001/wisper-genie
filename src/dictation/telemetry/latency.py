"""Per-stage latency tracking and budget enforcement."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator


class LatencyTracker:
    """Tracks per-stage latency and warns when budgets are exceeded."""

    def __init__(
        self,
        budgets: dict[str, float] | None = None,
        total_budget_ms: float = 800.0,
    ) -> None:
        """
        Args:
            budgets: Stage name -> budget in ms. E.g. {"vad": 100, "asr": 250}.
            total_budget_ms: Total end-to-end budget in ms.
        """
        self._budgets = budgets or {}
        self._total_budget_ms = total_budget_ms
        self._timings: dict[str, float] = {}  # stage -> ms
        self._pipeline_start: float | None = None

    def start_pipeline(self) -> None:
        """Mark the start of a pipeline run."""
        self._timings.clear()
        self._pipeline_start = time.perf_counter()

    @contextmanager
    def track(self, stage: str) -> Generator[None, None, None]:
        """Context manager to time a pipeline stage.

        Usage:
            with tracker.track("asr"):
                result = asr.transcribe(audio)
        """
        if self._pipeline_start is None:
            print(f"[Latency] WARNING: track('{stage}') called before start_pipeline()")
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._timings[stage] = elapsed_ms

            budget = self._budgets.get(stage)
            if budget is not None and elapsed_ms > budget:
                print(
                    f"[Latency] WARNING: {stage} took {elapsed_ms:.1f}ms "
                    f"(budget: {budget:.0f}ms)"
                )

    def finish_pipeline(self) -> float:
        """Mark the end of a pipeline run. Returns total elapsed ms."""
        if self._pipeline_start is None:
            return 0.0
        total_ms = (time.perf_counter() - self._pipeline_start) * 1000
        self._timings["total"] = total_ms

        if total_ms > self._total_budget_ms:
            print(
                f"[Latency] WARNING: total pipeline took {total_ms:.1f}ms "
                f"(budget: {self._total_budget_ms:.0f}ms)"
            )
        return total_ms

    def summary(self) -> str:
        """Return a formatted summary of all stage timings."""
        lines = ["Pipeline latency breakdown:"]
        for stage, ms in self._timings.items():
            budget = self._budgets.get(stage)
            if budget is not None:
                status = "OK" if ms <= budget else "OVER"
                lines.append(f"  {stage:20s} {ms:7.1f}ms / {budget:.0f}ms [{status}]")
            else:
                lines.append(f"  {stage:20s} {ms:7.1f}ms")
        return "\n".join(lines)

    @property
    def timings(self) -> dict[str, float]:
        """Return a copy of the current timings."""
        return dict(self._timings)
