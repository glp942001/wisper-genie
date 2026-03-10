"""CLI subcommands for Wisper Genie."""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path

from dictation.telemetry.metrics import RoutingMetrics

PLIST_NAME = "com.wisper-genie.dictation"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{PLIST_NAME}.plist"
WRAPPER_PATH = Path.home() / ".local" / "bin" / "wisper-genie"


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
    wrapper = WRAPPER_PATH
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


def show_metrics() -> None:
    """Print a compact summary of recent local routing metrics."""
    metrics = RoutingMetrics()
    events = metrics.read_recent(limit=200)
    if not events:
        print(f"No routing metrics found at {metrics.path}")
        return

    dictations = [event for event in events if event.get("event") == "dictation"]
    if not dictations:
        print(f"No dictation events found in {metrics.path}")
        return

    avg_total = sum(event.get("timings", {}).get("total", 0.0) for event in dictations) / len(dictations)
    direct_inserts = sum(1 for event in dictations if event.get("injection_route") == "direct")
    cleanup_skips = sum(1 for event in dictations if not event.get("cleanup_used", True))
    live_count = sum(1 for event in dictations if event.get("trigger_mode") == "live")
    avg_conf = sum(event.get("asr_confidence", 0.0) for event in dictations) / len(dictations)

    print(f"Metrics file: {metrics.path}")
    print(f"Recent dictations: {len(dictations)}")
    print(f"Average total latency: {avg_total:.1f}ms")
    print(f"Average ASR confidence: {avg_conf:.2f}")
    print(f"Direct insertions: {direct_inserts}/{len(dictations)}")
    print(f"Cleanup skips: {cleanup_skips}/{len(dictations)}")
    print(f"Live mode dictations: {live_count}/{len(dictations)}")


def main() -> None:
    """Entry point for CLI subcommands."""
    if len(sys.argv) < 2:
        print("Usage: python -m dictation.cli <autostart|metrics|uninstall>")
        sys.exit(1)

    subcmd = sys.argv[1]
    if subcmd == "autostart":
        autostart(sys.argv[2:])
    elif subcmd == "metrics":
        show_metrics()
    elif subcmd == "uninstall":
        uninstall()
    else:
        print(f"Unknown subcommand: {subcmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
