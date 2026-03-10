"""Text injection via clipboard + Cmd+V (macOS) using native APIs."""

from __future__ import annotations

import time

import AppKit
import Quartz

from dictation.context.screen import (
    FocusedTextDetails,
    get_focused_text_details,
    insert_text_into_focused_element,
)


class ClipboardInjector:
    """Injects text into the active application via NSPasteboard + CGEvent paste.

    Uses native macOS APIs — no subprocess overhead.
    """

    def __init__(
        self,
        paste_delay_ms: int = 50,
        prefer_direct_insertion: bool = True,
        prefer_typing_fallback: bool = True,
        typing_max_chars: int = 220,
    ) -> None:
        self._paste_delay = paste_delay_ms / 1000.0
        self._prefer_direct_insertion = prefer_direct_insertion
        self._prefer_typing_fallback = prefer_typing_fallback
        self._typing_max_chars = typing_max_chars
        self._pasteboard = AppKit.NSPasteboard.generalPasteboard()

    def inject(
        self,
        text: str,
        *,
        prefer_direct: bool = True,
        context_details: FocusedTextDetails | None = None,
    ) -> str:
        """Inject text into the active application.

        Returns the insertion route used: ``direct``, ``clipboard``, or ``skipped``.
        """
        if not text:
            return "skipped"

        self._activate_target_app(context_details)
        direct_details = self._refresh_target_details(context_details)
        if prefer_direct and self._prefer_direct_insertion:
            ok, _message = insert_text_into_focused_element(
                text,
                details=direct_details,
            )
            if ok:
                return "direct"

        self._activate_target_app(context_details)

        self._set_clipboard(text)
        time.sleep(self._paste_delay)
        pasted = self._paste()
        if pasted:
            return "clipboard"
        if self._prefer_typing_fallback and len(text) <= self._typing_max_chars:
            self._activate_target_app(context_details)
            if self._type_text(text):
                return "typing"
        return "clipboard"

    def clear_clipboard(self) -> None:
        """Clear the clipboard. Called before the next recording starts."""
        self._pasteboard.clearContents()

    def _set_clipboard(self, text: str) -> None:
        """Set clipboard using NSPasteboard (native, no subprocess)."""
        self._pasteboard.clearContents()
        self._pasteboard.setString_forType_(text, AppKit.NSPasteboardTypeString)

    def _paste(self) -> bool:
        """Simulate Cmd+V keystroke using CGEvent."""
        v_keycode = 9  # 'v' on macOS

        try:
            event_down = Quartz.CGEventCreateKeyboardEvent(None, v_keycode, True)
            if event_down is None:
                print("[Injector] ERROR: Could not create key event (check accessibility permissions)")
                return False

            Quartz.CGEventSetFlags(event_down, Quartz.kCGEventFlagMaskCommand)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_down)

            event_up = Quartz.CGEventCreateKeyboardEvent(None, v_keycode, False)
            if event_up is None:
                print("[Injector] ERROR: Could not create key-up event")
                return False

            Quartz.CGEventSetFlags(event_up, Quartz.kCGEventFlagMaskCommand)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_up)
            return True
        except Exception as exc:
            print(f"[Injector] ERROR posting paste event: {exc}")
            return False

    def _type_text(self, text: str) -> bool:
        """Type Unicode text directly into the active app as a fallback."""
        try:
            event_down = Quartz.CGEventCreateKeyboardEvent(None, 0, True)
            event_up = Quartz.CGEventCreateKeyboardEvent(None, 0, False)
            if event_down is None or event_up is None:
                return False

            Quartz.CGEventKeyboardSetUnicodeString(event_down, len(text), text)
            Quartz.CGEventKeyboardSetUnicodeString(event_up, len(text), text)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_down)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_up)
            return True
        except Exception as exc:
            print(f"[Injector] ERROR posting text event: {exc}")
            return False

    def _activate_target_app(self, context_details: FocusedTextDetails | None) -> None:
        """Re-activate the original target app before event-based insertion."""
        app_pid = getattr(context_details, "app_pid", None)
        if not app_pid:
            return
        try:
            target_app = AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(app_pid)
            if target_app is None:
                return
            options = getattr(AppKit, "NSApplicationActivateIgnoringOtherApps", 1 << 1)
            target_app.activateWithOptions_(options)
            deadline = time.monotonic() + 0.18
            while time.monotonic() < deadline:
                workspace = AppKit.NSWorkspace.sharedWorkspace()
                front_app = workspace.frontmostApplication()
                if front_app is not None and int(front_app.processIdentifier()) == app_pid:
                    time.sleep(0.02)
                    return
                time.sleep(0.01)
        except Exception:
            return

    def _refresh_target_details(
        self,
        context_details: FocusedTextDetails | None,
    ) -> FocusedTextDetails | None:
        """Refresh focused-element metadata after app reactivation when possible."""
        if context_details is None:
            return None
        app_pid = getattr(context_details, "app_pid", None)
        if app_pid is None:
            return context_details
        try:
            refreshed = get_focused_text_details(include_full_text=True)
        except Exception:
            return context_details
        if refreshed.app_pid != app_pid:
            return context_details
        return refreshed
