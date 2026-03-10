"""Text injection via clipboard + Cmd+V (macOS) using native APIs."""

from __future__ import annotations

import threading
import time

import AppKit
import Quartz


class ClipboardInjector:
    """Injects text into the active application via NSPasteboard + CGEvent paste.

    Uses native macOS APIs — no subprocess overhead.
    """

    def __init__(self, paste_delay_ms: int = 50, restore_delay_ms: int | None = None) -> None:
        self._paste_delay = paste_delay_ms / 1000.0
        self._restore_delay = (
            self._paste_delay if restore_delay_ms is None else restore_delay_ms / 1000.0
        )
        self._pasteboard = AppKit.NSPasteboard.generalPasteboard()
        self._lock = threading.Lock()
        self._version = 0

    def inject(self, text: str) -> None:
        """Inject text into the active application."""
        if not text:
            return

        with self._lock:
            previous_text = self._get_clipboard_text()
            self._version += 1
            version = self._version
            self._set_clipboard(text)

        time.sleep(self._paste_delay)
        self._paste()
        self._restore_clipboard(previous_text, version)

    def clear_clipboard(self) -> None:
        """Clear the clipboard and cancel any pending restore."""
        with self._lock:
            self._version += 1
            self._pasteboard.clearContents()

    def _get_clipboard_text(self) -> str | None:
        value = self._pasteboard.stringForType_(AppKit.NSPasteboardTypeString)
        return value if isinstance(value, str) else None

    def _set_clipboard(self, text: str) -> None:
        """Set clipboard using NSPasteboard (native, no subprocess)."""
        self._pasteboard.clearContents()
        self._pasteboard.setString_forType_(text, AppKit.NSPasteboardTypeString)

    def _restore_clipboard(self, previous_text: str | None, version: int) -> None:
        """Restore the user's previous clipboard after the paste event lands."""

        def restore() -> None:
            if self._restore_delay > 0:
                time.sleep(self._restore_delay)

            with self._lock:
                if version != self._version:
                    return
                self._pasteboard.clearContents()
                if previous_text:
                    self._pasteboard.setString_forType_(
                        previous_text,
                        AppKit.NSPasteboardTypeString,
                    )

        if self._restore_delay <= 0:
            restore()
            return

        threading.Thread(target=restore, daemon=True).start()

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
