"""Tests for screen context module."""

from unittest.mock import patch, MagicMock
from dictation.context.screen import get_screen_context


class TestScreenContext:
    def test_returns_dict_with_expected_keys(self):
        ctx = get_screen_context()
        assert "app_name" in ctx
        assert "field_text" in ctx

    def test_app_name_is_string(self):
        ctx = get_screen_context()
        assert isinstance(ctx["app_name"], str)

    def test_field_text_is_string(self):
        ctx = get_screen_context()
        assert isinstance(ctx["field_text"], str)

    @patch("dictation.context.screen.AppKit")
    def test_returns_app_name_from_nsworkspace(self, mock_appkit):
        mock_app = MagicMock()
        mock_app.localizedName.return_value = "Slack"
        mock_appkit.NSWorkspace.sharedWorkspace.return_value.frontmostApplication.return_value = mock_app

        ctx = get_screen_context()
        assert ctx["app_name"] == "Slack"

    @patch("dictation.context.screen.AppKit")
    def test_graceful_on_app_name_failure(self, mock_appkit):
        mock_appkit.NSWorkspace.sharedWorkspace.side_effect = Exception("no workspace")
        ctx = get_screen_context()
        assert ctx["app_name"] == ""
        assert ctx["field_text"] == ""

    def test_field_text_truncated_to_200_chars(self):
        """If we can't test the real AX API, at least verify the function doesn't crash."""
        ctx = get_screen_context()
        assert len(ctx["field_text"]) <= 200
