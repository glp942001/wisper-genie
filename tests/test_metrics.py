"""Tests for local routing metrics."""

from __future__ import annotations

import json

from dictation.telemetry.metrics import RoutingMetrics


def test_record_appends_jsonl_and_read_recent_returns_latest(tmp_path) -> None:
    path = tmp_path / "routing_metrics.jsonl"
    metrics = RoutingMetrics(path)

    metrics.record({"route": "direct", "app": "Slack"})
    metrics.record({"route": "clipboard", "app": "Mail"})

    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    second = json.loads(lines[1])

    assert len(lines) == 2
    assert first["route"] == "direct"
    assert second["route"] == "clipboard"
    assert "timestamp" in first
    assert "timestamp" in second
    assert metrics.read_recent(limit=1) == [second]


def test_read_recent_ignores_blank_and_invalid_lines(tmp_path) -> None:
    path = tmp_path / "routing_metrics.jsonl"
    path.write_text(
        '\n{"timestamp": 1, "route": "direct"}\nnot-json\n{"timestamp": 2, "route": "clipboard"}\n',
        encoding="utf-8",
    )
    metrics = RoutingMetrics(path)

    recent = metrics.read_recent(limit=10)

    assert recent == [
        {"timestamp": 1, "route": "direct"},
        {"timestamp": 2, "route": "clipboard"},
    ]


def test_read_recent_handles_missing_file_and_non_positive_limit(tmp_path) -> None:
    metrics = RoutingMetrics(tmp_path / "routing_metrics.jsonl")

    assert metrics.read_recent() == []

    metrics.record({"route": "direct"})

    assert metrics.read_recent(limit=0) == []
