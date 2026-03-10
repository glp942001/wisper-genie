from __future__ import annotations

import re
from dataclasses import dataclass, field

import Quartz

from dictation.context.screen import replace_text_in_focused_element


@dataclass
class CommandResult:
    action: str
    args: dict = field(default_factory=dict)


_COMMANDS: list[tuple[re.Pattern, str, dict]] = [
    (re.compile(r"^(?:delete that|delete last|scratch that)$", re.IGNORECASE), "delete_last", {}),
    (re.compile(r"^(?:undo that|undo)$", re.IGNORECASE), "undo", {}),
    (re.compile(r"^select all$", re.IGNORECASE), "select_all", {}),
]
_REPLACE_PATTERN = re.compile(r"^replace\s+(.+?)\s+with\s+(.+)$", re.IGNORECASE)


def detect_command(text: str) -> CommandResult | None:
    text = text.strip()
    if not text:
        return None

    clean = re.sub(r"[.,!?;:]+$", "", text).strip()
    if not clean:
        return None

    for pattern, action, args in _COMMANDS:
        if pattern.match(clean):
            return CommandResult(action=action, args=dict(args))

    match = _REPLACE_PATTERN.match(clean)
    if match:
        return CommandResult(
            action="replace",
            args={"find": match.group(1), "replacement": match.group(2)},
        )
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
        if cmd.action == "select_all":
            _simulate_keystroke(0, cmd_flag)  # Cmd+A
            return True
        if cmd.action == "replace":
            ok, message = replace_text_in_focused_element(
                cmd.args["find"],
                cmd.args["replacement"],
            )
            if not ok and message:
                print(f"[Commands] {message}")
            return ok
        return False
    except Exception as exc:
        print(f"[Commands] ERROR executing {cmd.action}: {exc}")
        return False
