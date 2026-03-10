"""Tests for the short-lived screen context cache."""

from __future__ import annotations

from unittest.mock import patch

from dictation.context.cache import ScreenContextCache
from dictation.context.screen import FocusedTextDetails


def test_cache_reuses_recent_snapshot() -> None:
    details = FocusedTextDetails(
        app_name="Slack",
        full_text="hello there",
        field_text="hello there",
        selected_text="",
        selected_range=(11, 0),
    )
    cache = ScreenContextCache(ttl_ms=500)

    with patch("dictation.context.cache.get_focused_text_details", return_value=details) as mock_get:
        first = cache.get(include_full_text=True)
        second = cache.get(include_full_text=True)

    assert first == details
    assert second == details
    mock_get.assert_called_once()


def test_prefetch_full_text_populates_light_snapshot() -> None:
    details = FocusedTextDetails(
        app_name="Mail",
        full_text="hello world",
        field_text="hello world",
        selected_text="world",
        selected_range=(6, 5),
    )
    cache = ScreenContextCache(ttl_ms=500)

    with patch("dictation.context.cache.get_focused_text_details", return_value=details):
        cache.prefetch(include_full_text=True)

    light = cache.get(include_full_text=False)
    assert light.app_name == "Mail"
    assert light.field_text == "hello world"
    assert light.selected_text == "world"
    assert light.full_text == ""
