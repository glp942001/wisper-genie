"""Transient ghost preview overlay for dictation feedback."""

from __future__ import annotations

import threading
import time

import AppKit
import Foundation


def _truncate_preview_text(text: str, max_chars: int) -> str:
    text = " ".join(text.split()).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


class GhostPreviewOverlay:
    """Shows a lightweight floating preview above the current workspace."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        preview_ms: int = 1200,
        status_ms: int = 500,
        max_chars: int = 120,
    ) -> None:
        self._enabled = enabled
        self._preview_seconds = preview_ms / 1000.0
        self._status_seconds = status_ms / 1000.0
        self._max_chars = max_chars
        self._lock = threading.Lock()
        self._version = 0
        self._window = None
        self._label = None

    def show_status(self, text: str) -> None:
        self._show(text, duration=self._status_seconds)

    def show_preview(self, text: str) -> None:
        self._show(text, duration=self._preview_seconds)

    def hide(self) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._version += 1
        self._run_on_main(self._hide_window)

    def _show(self, text: str, *, duration: float) -> None:
        if not self._enabled:
            return

        preview_text = _truncate_preview_text(text, self._max_chars)
        with self._lock:
            self._version += 1
            version = self._version

        self._run_on_main(lambda: self._show_window(preview_text))

        def hide_later() -> None:
            time.sleep(duration)
            with self._lock:
                if version != self._version:
                    return
            self._run_on_main(self._hide_window)

        threading.Thread(target=hide_later, daemon=True).start()

    def _run_on_main(self, callback) -> None:
        try:
            Foundation.NSOperationQueue.mainQueue().addOperationWithBlock_(callback)
        except Exception:
            try:
                callback()
            except Exception:
                return

    def _ensure_window(self) -> None:
        if self._window is not None and self._label is not None:
            return

        app = AppKit.NSApplication.sharedApplication()
        app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)

        screen = AppKit.NSScreen.mainScreen()
        if screen is None:
            raise RuntimeError("No screen available for overlay.")

        screen_frame = screen.visibleFrame()
        width = min(720.0, screen_frame.size.width - 80.0)
        height = 74.0
        x = screen_frame.origin.x + (screen_frame.size.width - width) / 2.0
        y = screen_frame.origin.y + screen_frame.size.height - 140.0
        frame = Foundation.NSMakeRect(x, y, width, height)

        style = AppKit.NSWindowStyleMaskBorderless
        if hasattr(AppKit, "NSWindowStyleMaskNonactivatingPanel"):
            style |= AppKit.NSWindowStyleMaskNonactivatingPanel
        window = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            style,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        window.setLevel_(AppKit.NSFloatingWindowLevel)
        window.setOpaque_(False)
        window.setBackgroundColor_(AppKit.NSColor.clearColor())
        window.setHasShadow_(True)
        if hasattr(window, "setFloatingPanel_"):
            window.setFloatingPanel_(True)
        window.setIgnoresMouseEvents_(True)
        if hasattr(window, "setHidesOnDeactivate_"):
            window.setHidesOnDeactivate_(False)
        window.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        content = AppKit.NSView.alloc().initWithFrame_(frame)
        content.setWantsLayer_(True)
        layer = content.layer()
        layer.setCornerRadius_(18.0)
        layer.setMasksToBounds_(True)
        layer.setBackgroundColor_(AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.08, 0.9).CGColor())

        label_frame = Foundation.NSMakeRect(24.0, 18.0, width - 48.0, height - 36.0)
        label = AppKit.NSTextField.alloc().initWithFrame_(label_frame)
        label.setBezeled_(False)
        label.setEditable_(False)
        label.setDrawsBackground_(False)
        label.setSelectable_(False)
        label.setFont_(AppKit.NSFont.systemFontOfSize_weight_(22.0, AppKit.NSFontWeightMedium))
        label.setTextColor_(AppKit.NSColor.whiteColor())
        label.setAlignment_(AppKit.NSTextAlignmentCenter)
        label.setLineBreakMode_(AppKit.NSLineBreakByTruncatingTail)
        label.setMaximumNumberOfLines_(2)

        content.addSubview_(label)
        window.setContentView_(content)
        self._window = window
        self._label = label

    def _show_window(self, text: str) -> None:
        try:
            self._ensure_window()
            self._label.setStringValue_(text)
            self._window.orderFrontRegardless()
        except Exception:
            return

    def _hide_window(self) -> None:
        try:
            if self._window is not None:
                self._window.orderOut_(None)
        except Exception:
            return
