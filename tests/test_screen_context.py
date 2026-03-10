"""Tests for screen context helpers."""

from unittest.mock import MagicMock, patch

import pytest

from dictation.context.screen import (
    FocusedTextDetails,
    _build_context_window,
    build_replacement_text,
    get_screen_context,
)


class TestContextWindow:
    def test_uses_tail_when_no_selection(self):
        text = "a" * 250
        result = _build_context_window(text, selected_range=None, selected_text="", max_chars=200)
        assert result == text[-200:]

    def test_prefers_text_around_selection(self):
        text = "0123456789" * 40
        result = _build_context_window(text, selected_range=(150, 5), selected_text="", max_chars=60)
        assert "456789" in result
        assert len(result) <= 60

    def test_falls_back_to_selected_text_when_value_missing(self):
        result = _build_context_window("", selected_range=(0, 5), selected_text="hello world", max_chars=5)
        assert result == "world"


class TestReplacementPlanning:
    def test_replaces_selected_text_first(self):
        new_text, caret = build_replacement_text(
            "hello world",
            "world",
            "team",
            selected_range=(6, 5),
            selected_text="world",
        )
        assert new_text == "hello team"
        assert caret == (10, 0)

    def test_replaces_unique_case_insensitive_match(self):
        new_text, caret = build_replacement_text(
            "Hello Ollama team",
            "ollama",
            "Whisper",
        )
        assert new_text == "Hello Whisper team"
        assert caret == (13, 0)

    def test_rejects_ambiguous_match(self):
        with pytest.raises(ValueError, match="appears multiple times"):
            build_replacement_text("hello hello", "hello", "hi")

    def test_rejects_missing_match(self):
        with pytest.raises(ValueError, match="was not found"):
            build_replacement_text("hello world", "team", "crew")


class TestScreenContext:
    @patch("dictation.context.screen.get_focused_text_details")
    def test_returns_dict_with_expected_keys(self, mock_details):
        mock_details.return_value = FocusedTextDetails(
            app_name="Slack",
            full_text="",
            field_text="hi there",
            selected_text="",
            selected_range=None,
        )
        ctx = get_screen_context()
        assert ctx == {
            "app_name": "Slack",
            "field_text": "hi there",
            "selected_text": "",
        }

    @patch("dictation.context.screen.AppKit")
    def test_graceful_on_app_name_failure(self, mock_appkit):
        mock_appkit.NSWorkspace.sharedWorkspace.side_effect = Exception("no workspace")
        ctx = get_screen_context()
        assert ctx["app_name"] == ""
        assert ctx["field_text"] == ""
        assert ctx["selected_text"] == ""

    @patch("dictation.context.screen.AppKit")
    @patch("dictation.context.screen._get_focused_element")
    @patch("dictation.context.screen._read_text_attribute")
    @patch("dictation.context.screen._read_selected_range")
    def test_skip_list_apps_return_selection_but_not_field_text(
        self,
        mock_selected_range,
        mock_read_text,
        mock_get_element,
        mock_appkit,
    ):
        mock_app = MagicMock()
        mock_app.localizedName.return_value = "Cursor"
        mock_appkit.NSWorkspace.sharedWorkspace.return_value.frontmostApplication.return_value = mock_app
        mock_get_element.return_value = object()
        mock_read_text.side_effect = ["selected snippet", "full text should be ignored"]
        mock_selected_range.return_value = (0, 8)

        ctx = get_screen_context()

        assert ctx["app_name"] == "Cursor"
        assert ctx["field_text"] == ""
        assert ctx["selected_text"] == "selected snippet"
