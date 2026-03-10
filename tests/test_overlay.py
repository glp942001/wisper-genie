"""Tests for the ghost preview overlay helpers."""

from __future__ import annotations

from dictation.output.overlay import GhostPreviewOverlay, _truncate_preview_text


def test_truncate_preview_text_compacts_whitespace() -> None:
    result = _truncate_preview_text("hello   there\nteam", max_chars=40)
    assert result == "hello there team"


def test_truncate_preview_text_adds_ellipsis_when_needed() -> None:
    result = _truncate_preview_text("abcdefghijklmnopqrstuvwxyz", max_chars=10)
    assert result == "abcdefghi…"


def test_disabled_overlay_is_noop() -> None:
    overlay = GhostPreviewOverlay(enabled=False)
    overlay.show_status("Recording")
    overlay.show_preview("hello world")
    overlay.hide()
