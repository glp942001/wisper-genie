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


def uninstall() -> None:
    """Remove Wisper Genie completely from the system."""
    import shutil

    genie_home = Path.home() / ".wisper-genie"
    wrapper = Path.home() / ".local" / "bin" / "dictation"
    zshrc = Path.home() / ".zshrc"

    print("Wisper Genie — Uninstaller")
    print("=" * 40)
    print()
    print("This will remove:")
    print(f"  {genie_home}/")
    print(f"  {wrapper}")
    if PLIST_PATH.exists():
        print(f"  {PLIST_PATH}")
    print()

    try:
        answer = input("Are you sure? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return

    if answer not in ("y", "yes"):
        print("Cancelled.")
        return

    # 1. Remove autostart if active
    if PLIST_PATH.exists():
        subprocess.run(["launchctl", "unload", str(PLIST_PATH)], check=False)
        PLIST_PATH.unlink()
        print("  Removed autostart plist.")

    # 2. Remove the install directory (~/.wisper-genie)
    if genie_home.exists():
        shutil.rmtree(genie_home)
        print(f"  Removed {genie_home}/")

    # 3. Remove the wrapper script
    if wrapper.exists():
        wrapper.unlink()
        print(f"  Removed {wrapper}")

    # 4. Clean up PATH entry from .zshrc
    if zshrc.exists():
        lines = zshrc.read_text().splitlines()
        new_lines = [
            line for line in lines
            if "Added by Wisper Genie" not in line
            and not (line.strip().startswith("export PATH") and ".local/bin" in line and "Wisper" not in line and line == 'export PATH="$HOME/.local/bin:$PATH"')
        ]
        # Also remove the specific line added by installer
        new_lines = [
            line for line in new_lines
            if line.strip() != 'export PATH="$HOME/.local/bin:$PATH"'
            or "Wisper" not in "".join(lines)  # only remove if we added it
        ]
        zshrc.write_text("\n".join(new_lines) + "\n")
        print("  Cleaned up ~/.zshrc")

    print()
    print("Wisper Genie has been uninstalled.")
    print("Note: Ollama and its models were left in place (shared system tool).")
    print("  To remove Ollama: brew uninstall --cask ollama")
    print("  To remove the model: ollama rm qwen3.5:2b")


def main() -> None:
    """Entry point for CLI subcommands."""
    if len(sys.argv) < 2:
        print("Usage: python -m dictation.cli <subcommand>")
        sys.exit(1)

    subcmd = sys.argv[1]
    if subcmd == "autostart":
        autostart(sys.argv[2:])
    elif subcmd == "uninstall":
        uninstall()
    else:
        print(f"Unknown subcommand: {subcmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
