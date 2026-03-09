from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import AppKit
import Quartz


@dataclass
class CommandResult:
    action: str
    args: dict = field(default_factory=dict)


# Pre-compiled command patterns — must match the FULL utterance
_COMMANDS: list[tuple[re.Pattern, str, dict]] = [
    (re.compile(r"^(?:delete that|delete last|scratch that)$", re.IGNORECASE), "delete_last", {}),
    (re.compile(r"^(?:undo that|undo)$", re.IGNORECASE), "undo", {}),
    (re.compile(r"^select all$", re.IGNORECASE), "select_all", {}),
]
_REPLACE_PATTERN = re.compile(
    r"^replace\s+(.+?)\s+with\s+(.+)$", re.IGNORECASE
)


def detect_command(text: str) -> CommandResult | None:
    text = text.strip()
    if not text:
        return None
    for pattern, action, args in _COMMANDS:
        if pattern.match(text):
            return CommandResult(action=action, args=dict(args))
    m = _REPLACE_PATTERN.match(text)
    if m:
        return CommandResult(action="replace", args={"find": m.group(1), "replacement": m.group(2)})
    return None


def _simulate_keystroke(keycode: int, flags: int = 0) -> None:
    event_down = Quartz.CGEventCreateKeyboardEvent(None, keycode, True)
    if event_down is None:
        return
    if flags:
        Quartz.CGEventSetFlags(event_down, flags)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_down)

    event_up = Quartz.CGEventCreateKeyboardEvent(None, keycode, False)
    if event_up is None:
        return
    if flags:
        Quartz.CGEventSetFlags(event_up, flags)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_up)


def execute_command(cmd: CommandResult) -> bool:
    try:
        cmd_flag = Quartz.kCGEventFlagMaskCommand
        if cmd.action in ("delete_last", "undo"):
            _simulate_keystroke(6, cmd_flag)  # Cmd+Z
            return True
        elif cmd.action == "select_all":
            _simulate_keystroke(0, cmd_flag)  # Cmd+A
            return True
        elif cmd.action == "replace":
            # Select all, paste replacement
            _simulate_keystroke(0, cmd_flag)  # Cmd+A
            time.sleep(0.05)
            pb = AppKit.NSPasteboard.generalPasteboard()
            pb.clearContents()
            pb.setString_forType_(cmd.args["replacement"], AppKit.NSPasteboardTypeString)
            time.sleep(0.05)
            _simulate_keystroke(9, cmd_flag)  # Cmd+V
            time.sleep(0.05)
            pb.clearContents()
            return True
        return False
    except Exception as exc:
        print(f"[Commands] ERROR executing {cmd.action}: {exc}")
        return False
