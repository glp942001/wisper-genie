"""Tests for text injector."""

from unittest.mock import patch, MagicMock

from dictation.output.injector import ClipboardInjector


class TestClipboardInjector:
    def test_empty_text_does_nothing(self):
        injector = ClipboardInjector()
        # Should not raise
        injector.inject("")

    @patch("dictation.output.injector.Quartz")
    @patch("dictation.output.injector.AppKit")
    def test_inject_sets_clipboard_and_pastes(self, mock_appkit, mock_quartz):
        mock_pasteboard = MagicMock()
        mock_appkit.NSPasteboard.generalPasteboard.return_value = mock_pasteboard

        injector = ClipboardInjector(paste_delay_ms=0)
        injector._pasteboard = mock_pasteboard
        injector.inject("hello world")

        # clearContents called twice: once in _set_clipboard (before writing)
        # and once in finally block (to clear sensitive text after paste)
        assert mock_pasteboard.clearContents.call_count == 2
        mock_pasteboard.setString_forType_.assert_called_once()
        assert mock_quartz.CGEventPost.call_count == 2

    @patch("dictation.output.injector.Quartz")
    def test_paste_creates_cmd_v_events(self, mock_quartz):
        mock_quartz.CGEventCreateKeyboardEvent.return_value = MagicMock()

        injector = ClipboardInjector()
        injector._paste()

        calls = mock_quartz.CGEventCreateKeyboardEvent.call_args_list
        assert len(calls) == 2
        assert calls[0][0][1] == 9  # keycode v
        assert calls[0][0][2] is True  # key down
        assert calls[1][0][1] == 9  # keycode v
        assert calls[1][0][2] is False  # key up

    @patch("dictation.output.injector.Quartz")
    def test_paste_handles_null_event(self, mock_quartz):
        """If CGEvent returns None (permissions issue), should not crash."""
        mock_quartz.CGEventCreateKeyboardEvent.return_value = None

        injector = ClipboardInjector()
        injector._paste()  # Should not raise

        mock_quartz.CGEventPost.assert_not_called()
