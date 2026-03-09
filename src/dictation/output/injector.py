"""Text injection via clipboard + Cmd+V (macOS) using native APIs."""

from __future__ import annotations

import time

import AppKit
import Quartz


class ClipboardInjector:
    """Injects text into the active application via NSPasteboard + CGEvent paste.

    Uses native macOS APIs — no subprocess overhead.
    """

    def __init__(self, paste_delay_ms: int = 50) -> None:
        self._paste_delay = paste_delay_ms / 1000.0
        self._pasteboard = AppKit.NSPasteboard.generalPasteboard()

    def inject(self, text: str) -> None:
        """Inject text into the active application."""
        if not text:
            return

        self._set_clipboard(text)
        time.sleep(self._paste_delay)
        self._paste()
        # Do NOT clear clipboard here. The target app (especially browsers)
        # may not have read the clipboard yet. Clipboard is cleared on
        # the next call to inject() or clear_clipboard().

    def clear_clipboard(self) -> None:
        """Clear the clipboard. Called before next recording starts."""
        self._pasteboard.clearContents()

    def _set_clipboard(self, text: str) -> None:
        """Set clipboard using NSPasteboard (native, no subprocess)."""
        self._pasteboard.clearContents()
        self._pasteboard.setString_forType_(text, AppKit.NSPasteboardTypeString)

    def _paste(self) -> None:
        """Simulate Cmd+V keystroke using CGEvent."""
        v_keycode = 9  # 'v' on macOS

        try:
            event_down = Quartz.CGEventCreateKeyboardEvent(None, v_keycode, True)
            if event_down is None:
                print("[Injector] ERROR: Could not create key event (check accessibility permissions)")
                return

            Quartz.CGEventSetFlags(event_down, Quartz.kCGEventFlagMaskCommand)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_down)

            event_up = Quartz.CGEventCreateKeyboardEvent(None, v_keycode, False)
            if event_up is None:
                print("[Injector] ERROR: Could not create key-up event")
                return

            Quartz.CGEventSetFlags(event_up, Quartz.kCGEventFlagMaskCommand)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_up)
        except Exception as exc:
            print(f"[Injector] ERROR posting paste event: {exc}")
