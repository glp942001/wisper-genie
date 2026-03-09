"""CLI subcommands for Wisper Genie."""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path

PLIST_NAME = "com.wisper-genie.dictation"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{PLIST_NAME}.plist"
WRAPPER_PATH = Path.home() / ".local" / "bin" / "dictation"


def _create_plist() -> None:
    """Create a LaunchAgent plist to start dictation on login."""
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{PLIST_NAME}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{WRAPPER_PATH}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
    <key>StandardOutPath</key>
    <string>{Path.home() / '.wisper-genie' / 'dictation.log'}</string>
    <key>StandardErrorPath</key>
    <string>{Path.home() / '.wisper-genie' / 'dictation.err.log'}</string>
</dict>
</plist>
"""
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.write_text(plist_content)
    subprocess.run(["launchctl", "load", str(PLIST_PATH)], check=False)
    print(f"Autostart enabled. Dictation will launch on login.")
    print(f"  Plist: {PLIST_PATH}")


def _remove_plist() -> None:
    """Remove the LaunchAgent plist."""
    if PLIST_PATH.exists():
        subprocess.run(["launchctl", "unload", str(PLIST_PATH)], check=False)
        PLIST_PATH.unlink()
        print("Autostart disabled.")
    else:
        print("Autostart was not enabled.")


def autostart(args: list[str]) -> None:
    """Handle the autostart subcommand."""
    if "--remove" in args:
        _remove_plist()
    else:
        _create_plist()


def main() -> None:
    """Entry point for CLI subcommands."""
    if len(sys.argv) < 2:
        print("Usage: python -m dictation.cli <subcommand>")
        sys.exit(1)

    subcmd = sys.argv[1]
    if subcmd == "autostart":
        autostart(sys.argv[2:])
    else:
        print(f"Unknown subcommand: {subcmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
