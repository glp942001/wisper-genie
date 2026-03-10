"""Tests for text injector."""

from unittest.mock import patch, MagicMock

from dictation.output.injector import ClipboardInjector


class TestClipboardInjector:
    def test_empty_text_does_nothing(self):
        injector = ClipboardInjector()
        assert injector.inject("") == "skipped"

    @patch("dictation.output.injector.insert_text_into_focused_element")
    def test_inject_prefers_direct_insertion_when_available(self, mock_insert):
        mock_insert.return_value = (True, "")

        injector = ClipboardInjector(prefer_direct_insertion=True)
        route = injector.inject("hello world", prefer_direct=True)

        assert route == "direct"
        mock_insert.assert_called_once()

    @patch("dictation.output.injector.insert_text_into_focused_element")
    @patch("dictation.output.injector.AppKit")
    def test_inject_uses_typing_fallback_when_paste_events_fail(self, mock_appkit, mock_insert):
        mock_insert.return_value = (False, "unsupported")
        mock_pasteboard = MagicMock()
        mock_appkit.NSPasteboard.generalPasteboard.return_value = mock_pasteboard

        injector = ClipboardInjector(prefer_direct_insertion=True, paste_delay_ms=0)
        injector._pasteboard = mock_pasteboard
        with patch.object(injector, "_type_text", return_value=True) as mock_type:
            with patch.object(injector, "_paste", return_value=False) as mock_paste:
                route = injector.inject("hello world")

        assert route == "typing"
        mock_paste.assert_called_once()
        mock_type.assert_called_once_with("hello world")

    @patch("dictation.output.injector.Quartz")
    @patch("dictation.output.injector.AppKit")
    @patch("dictation.output.injector.insert_text_into_focused_element")
    def test_inject_sets_clipboard_and_pastes(self, mock_insert, mock_appkit, mock_quartz):
        mock_insert.return_value = (False, "unsupported")
        mock_pasteboard = MagicMock()
        mock_appkit.NSPasteboard.generalPasteboard.return_value = mock_pasteboard

        injector = ClipboardInjector(paste_delay_ms=0)
        injector._pasteboard = mock_pasteboard
        with patch.object(injector, "_type_text", return_value=False) as mock_type:
            route = injector.inject("hello world")

        # Clipboard stays populated after paste so slow targets still have time
        # to read it. It is cleared on the next recording start.
        assert route == "clipboard"
        mock_type.assert_not_called()
        assert mock_pasteboard.clearContents.call_count == 1
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
        assert injector._paste() is False

        mock_quartz.CGEventPost.assert_not_called()

    @patch("dictation.output.injector.Quartz")
    def test_type_text_posts_unicode_events(self, mock_quartz):
        mock_quartz.CGEventCreateKeyboardEvent.side_effect = [MagicMock(), MagicMock()]

        injector = ClipboardInjector()
        assert injector._type_text("hello") is True
        assert mock_quartz.CGEventKeyboardSetUnicodeString.call_count == 2
        assert mock_quartz.CGEventPost.call_count == 2
