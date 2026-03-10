"""Screen context module for macOS dictation app.

Reads the active (frontmost) application name and focused text context via
macOS Accessibility APIs. Fully local — no network calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import AppKit
from ApplicationServices import (
    AXUIElementCopyAttributeValue,
    AXUIElementCreateSystemWide,
    AXUIElementSetAttributeValue,
    AXValueCreate,
    AXValueGetValue,
    CFRangeMake,
    kAXFocusedApplicationAttribute,
    kAXFocusedUIElementAttribute,
    kAXSelectedTextAttribute,
    kAXSelectedTextRangeAttribute,
    kAXValueAttribute,
    kAXValueCFRangeType,
)

_warned_once = False
_MAX_CONTEXT_CHARS = 200
_SKIP_FIELD_TEXT_APPS = {
    "terminal", "iterm", "iterm2", "warp", "alacritty", "kitty", "hyper",
    "code", "visual studio code", "cursor", "windsurf", "xcode",
    "intellij idea", "pycharm", "webstorm", "neovim", "vim",
}
_CONTEXT_READ_DISALLOWED_APPS = _SKIP_FIELD_TEXT_APPS


@dataclass(frozen=True)
class FocusedTextDetails:
    app_name: str
    full_text: str
    field_text: str
    selected_text: str
    selected_range: tuple[int, int] | None
    focused_element: object | None = None
    app_pid: int | None = None


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


def _copy_attribute(element: object, attribute: str) -> object | None:
    err, value = AXUIElementCopyAttributeValue(element, attribute, None)
    if err != 0 or value is None:
        return None
    return value


def _get_focused_element() -> object | None:
    """Return the currently focused accessibility element."""
    system_wide = AXUIElementCreateSystemWide()
    focused_app = _copy_attribute(system_wide, kAXFocusedApplicationAttribute)
    if focused_app is None:
        _warn_once(
            "Could not get focused application via Accessibility API. "
            "Ensure this app has Accessibility permissions in "
            "System Settings > Privacy & Security > Accessibility."
        )
        return None

    focused_element = _copy_attribute(focused_app, kAXFocusedUIElementAttribute)
    if focused_element is None:
        return None
    return focused_element


def _read_text_attribute(element: object, attribute: str) -> str:
    value = _copy_attribute(element, attribute)
    return str(value) if isinstance(value, str) else ""


def _read_selected_range(element: object) -> tuple[int, int] | None:
    value = _copy_attribute(element, kAXSelectedTextRangeAttribute)
    if value is None:
        return None
    try:
        ok, raw_range = AXValueGetValue(value, kAXValueCFRangeType, None)
    except Exception:
        return None
    if not ok or raw_range is None:
        return None
    location, length = raw_range
    if location < 0 or length < 0:
        return None
    return int(location), int(length)


def _build_context_window(
    full_text: str,
    selected_range: tuple[int, int] | None,
    selected_text: str,
    max_chars: int = _MAX_CONTEXT_CHARS,
) -> str:
    """Build the snippet closest to the insertion point for cleanup context."""
    if not full_text:
        return selected_text[-max_chars:]

    if not selected_range:
        return full_text[-max_chars:]

    location, length = selected_range
    text_length = len(full_text)
    if location > text_length:
        return full_text[-max_chars:]

    anchor = min(text_length, max(0, location + length))
    before_budget = int(max_chars * 0.7)
    after_budget = max_chars - before_budget
    start = max(0, anchor - before_budget)
    end = min(text_length, anchor + after_budget)
    return full_text[start:end]


def _count_case_insensitive_matches(text: str, needle: str) -> list[tuple[int, int]]:
    if not text or not needle:
        return []

    matches: list[tuple[int, int]] = []
    cursor = 0
    haystack_lower = text.lower()
    needle_lower = needle.lower()
    needle_len = len(needle)
    while True:
        idx = haystack_lower.find(needle_lower, cursor)
        if idx < 0:
            return matches
        matches.append((idx, idx + needle_len))
        cursor = idx + needle_len


def build_replacement_text(
    full_text: str,
    find_text: str,
    replacement: str,
    *,
    selected_range: tuple[int, int] | None = None,
    selected_text: str = "",
) -> tuple[str, tuple[int, int]]:
    """Build a safe text replacement against the current focused value."""
    find_text = find_text.strip()
    if not full_text:
        raise ValueError("Focused field text is unavailable.")
    if not find_text:
        raise ValueError("Replace target is empty.")

    if selected_range and selected_text and selected_text.strip().lower() == find_text.lower():
        start = selected_range[0]
        end = min(len(full_text), start + selected_range[1])
        new_text = full_text[:start] + replacement + full_text[end:]
        return new_text, (start + len(replacement), 0)

    exact_matches = _count_case_insensitive_matches(full_text, find_text)
    if not exact_matches:
        raise ValueError(f"'{find_text}' was not found in the focused field.")
    if len(exact_matches) > 1:
        raise ValueError(
            f"'{find_text}' appears multiple times. Select the exact text first, then try again."
        )

    start, end = exact_matches[0]
    new_text = full_text[:start] + replacement + full_text[end:]
    return new_text, (start + len(replacement), 0)


def build_insertion_text(
    full_text: str,
    insertion_text: str,
    *,
    selected_range: tuple[int, int] | None = None,
) -> tuple[str, tuple[int, int]]:
    """Build a direct text insertion against the current focused value."""
    if not insertion_text:
        raise ValueError("Insertion text is empty.")
    if full_text is None:
        raise ValueError("Focused field text is unavailable.")

    if not selected_range:
        start = end = len(full_text)
    else:
        start = max(0, selected_range[0])
        end = min(len(full_text), start + selected_range[1])

    new_text = full_text[:start] + insertion_text + full_text[end:]
    return new_text, (start + len(insertion_text), 0)


def get_focused_text_details(
    *,
    context_chars: int = _MAX_CONTEXT_CHARS,
    include_full_text: bool = False,
) -> FocusedTextDetails:
    """Return the active app, focused field text, and selection metadata."""
    try:
        app_name = _get_app_name()
    except Exception:
        app_name = ""

    focused_element = _get_focused_element()
    if focused_element is None:
        return FocusedTextDetails(
            app_name=app_name,
            full_text="",
            field_text="",
            selected_text="",
            selected_range=None,
            focused_element=None,
            app_pid=None,
        )

    try:
        workspace = AppKit.NSWorkspace.sharedWorkspace()
        front_app = workspace.frontmostApplication()
        app_pid = int(front_app.processIdentifier()) if front_app is not None else None
    except Exception:
        app_pid = None

    selected_text = _read_text_attribute(focused_element, kAXSelectedTextAttribute)
    selected_range = _read_selected_range(focused_element)

    skip_context_read = any(skip in app_name.lower() for skip in _CONTEXT_READ_DISALLOWED_APPS)
    if skip_context_read and not include_full_text:
        return FocusedTextDetails(
            app_name=app_name,
            full_text="",
            field_text="",
            selected_text=selected_text[:context_chars],
            selected_range=selected_range,
            focused_element=focused_element if include_full_text else None,
            app_pid=app_pid,
        )

    try:
        full_text = _read_text_attribute(focused_element, kAXValueAttribute)
    except Exception:
        _warn_once(
            "Could not read focused field text. "
            "Check Accessibility permissions in "
            "System Settings > Privacy & Security > Accessibility."
        )
        full_text = ""

    field_text = _build_context_window(
        full_text,
        selected_range=selected_range,
        selected_text=selected_text,
        max_chars=context_chars,
    ).strip()

    return FocusedTextDetails(
        app_name=app_name,
        full_text=full_text if include_full_text else "",
        field_text=field_text,
        selected_text=selected_text[:context_chars],
        selected_range=selected_range,
        focused_element=focused_element if include_full_text else None,
        app_pid=app_pid,
    )


def get_screen_context(*, context_chars: int = _MAX_CONTEXT_CHARS) -> dict:
    """Return a dict with the frontmost app name and focused field text."""
    details = get_focused_text_details(context_chars=context_chars)
    return {
        "app_name": details.app_name,
        "field_text": details.field_text,
        "selected_text": details.selected_text,
    }


def replace_text_in_focused_element(find_text: str, replacement: str) -> tuple[bool, str]:
    """Safely replace text in the focused field when the target is unambiguous."""
    details = get_focused_text_details(include_full_text=True)
    app_name = details.app_name.lower()
    selection_matches = (
        details.selected_range is not None
        and details.selected_text.strip()
        and details.selected_text.strip().lower() == find_text.strip().lower()
    )

    if any(skip in app_name for skip in _SKIP_FIELD_TEXT_APPS) and not selection_matches:
        return False, (
            "Replace is only supported in editors when the exact text is selected first."
        )

    focused_element = _get_focused_element()
    if focused_element is None:
        return False, "No focused text field is available."

    try:
        new_text, caret_range = build_replacement_text(
            details.full_text,
            find_text,
            replacement,
            selected_range=details.selected_range,
            selected_text=details.selected_text,
        )
    except ValueError as exc:
        return False, str(exc)

    err = AXUIElementSetAttributeValue(focused_element, kAXValueAttribute, new_text)
    if err != 0:
        return False, "Focused field does not support safe text replacement."

    try:
        caret_value = AXValueCreate(kAXValueCFRangeType, CFRangeMake(*caret_range))
        AXUIElementSetAttributeValue(
            focused_element,
            kAXSelectedTextRangeAttribute,
            caret_value,
        )
    except Exception:
        pass

    return True, ""


def insert_text_into_focused_element(
    text: str,
    *,
    details: FocusedTextDetails | None = None,
) -> tuple[bool, str]:
    """Prefer direct AX text insertion over clipboard paste when supported."""
    details = details or get_focused_text_details(include_full_text=True)
    focused_element = details.focused_element or _get_focused_element()
    if focused_element is None:
        return False, "No focused text field is available."

    try:
        new_text, caret_range = build_insertion_text(
            details.full_text,
            text,
            selected_range=details.selected_range,
        )
    except ValueError as exc:
        return False, str(exc)

    err = AXUIElementSetAttributeValue(focused_element, kAXValueAttribute, new_text)
    if err != 0:
        return False, "Focused field does not support direct text insertion."

    try:
        caret_value = AXValueCreate(kAXValueCFRangeType, CFRangeMake(*caret_range))
        AXUIElementSetAttributeValue(
            focused_element,
            kAXSelectedTextRangeAttribute,
            caret_value,
        )
    except Exception:
        pass

    return True, ""
