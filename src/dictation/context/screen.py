"""Screen context module for macOS dictation app.

Reads the active (frontmost) application name and the focused text field
content via macOS native APIs. Fully local — no network calls.
"""

from __future__ import annotations

import AppKit
from ApplicationServices import (
    AXUIElementCreateSystemWide,
    AXUIElementCopyAttributeValue,
    kAXFocusedApplicationAttribute,
    kAXFocusedUIElementAttribute,
    kAXValueAttribute,
)

_warned_once = False

# Apps where reading the text field is harmful (terminal buffers, code editors
# with massive files). We still detect the app name for tone, but skip field text.
_SKIP_FIELD_TEXT_APPS = {
    "terminal", "iterm", "iterm2", "warp", "alacritty", "kitty", "hyper",
    "code", "visual studio code", "cursor", "windsurf", "xcode",
    "intellij idea", "pycharm", "webstorm", "neovim", "vim",
}


def _warn_once(msg: str) -> None:
    """Print a warning only on the first failure."""
    global _warned_once
    if not _warned_once:
        print(f"[screen-context] {msg}")
        _warned_once = True


def _get_app_name() -> str:
    """Return the localised name of the frontmost application."""
    workspace = AppKit.NSWorkspace.sharedWorkspace()
    front_app = workspace.frontmostApplication()
    name = front_app.localizedName()
    return str(name) if name else ""


def _get_field_text() -> str:
    """Return the text value of the currently focused UI element.

    Uses the macOS Accessibility (AX) API to walk from the system-wide
    element down to the focused text field and read its value.
    The result is truncated to the last 200 characters.
    """
    # 1. System-wide accessibility element
    system_wide = AXUIElementCreateSystemWide()

    # 2. Focused application
    err, focused_app = AXUIElementCopyAttributeValue(
        system_wide, kAXFocusedApplicationAttribute, None
    )
    if err != 0 or focused_app is None:
        _warn_once(
            "Could not get focused application via Accessibility API. "
            "Ensure this app has Accessibility permissions in "
            "System Settings > Privacy & Security > Accessibility."
        )
        return ""

    # 3. Focused UI element
    err, focused_element = AXUIElementCopyAttributeValue(
        focused_app, kAXFocusedUIElementAttribute, None
    )
    if err != 0 or focused_element is None:
        return ""

    # 4. Value of the focused element
    err, value = AXUIElementCopyAttributeValue(
        focused_element, kAXValueAttribute, None
    )
    if err != 0 or value is None:
        return ""

    text = str(value)

    # 5. Truncate to last 200 characters
    return text[-200:]


def get_screen_context() -> dict:
    """Return a dict with the frontmost app name and focused field text.

    Returns
    -------
    dict
        ``{"app_name": str, "field_text": str}``
        Both values default to ``""`` on any failure (e.g. accessibility
        permissions not granted, no focused element, etc.).
    """
    try:
        app_name = _get_app_name()
    except Exception:
        app_name = ""

    # Skip field text for terminals/editors — their buffers are noise
    if any(skip in app_name.lower() for skip in _SKIP_FIELD_TEXT_APPS):
        field_text = ""
    else:
        try:
            field_text = _get_field_text()
        except Exception:
            _warn_once(
                "Could not read focused field text. "
                "Check Accessibility permissions in "
                "System Settings > Privacy & Security > Accessibility."
            )
            field_text = ""

    return {"app_name": app_name, "field_text": field_text}
