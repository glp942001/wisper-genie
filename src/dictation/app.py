"""Main entry point — pipeline orchestrator with push-to-talk."""

from __future__ import annotations

import io
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Config loading
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

# --- Verbose flag ---
VERBOSE = "--verbose" in sys.argv or "--debug" in sys.argv or "-v" in sys.argv


def _log(msg: str) -> None:
    """Print only in verbose mode."""
    if VERBOSE:
        print(msg)


def _progress(step: str, done: bool = False) -> None:
    """Print a compact progress line."""
    if done:
        print(f"  ✔ {step}")
    else:
        print(f"  ⏳ {step}...", end="", flush=True)


def _progress_done() -> None:
    print(" ✔")


def load_config(path: Path | None = None) -> dict:
    """Load configuration from TOML file."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config" / "default.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)


def main() -> None:
    """Run the dictation pipeline."""
    import AppKit
    from pynput import keyboard

    from dictation.asr.whisper_cpp import WhisperCppAdapter
    from dictation.audio.capture import MicCapture
    from dictation.cleanup.ollama import OllamaCleanup
    from dictation.commands.handler import detect_command, execute_command
    from dictation.context.dictionary import load_dictionary
    from dictation.output.injector import ClipboardInjector
    from dictation.telemetry.latency import LatencyTracker
    from dictation.transcript.buffer import TranscriptBuffer

    cfg = load_config()

    # Resolve model path relative to project root
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / cfg["asr"]["model_path"]

    # Suppress whisper.cpp verbose output unless --verbose
    if not VERBOSE:
        # Redirect C-level stderr during model load to suppress whisper logs
        _stderr_fd = os.dup(2)
        _devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull, 2)

    # --- Startup banner ---
    print()
    print("  🧞 Wisper Genie")
    print("  ─────────────────────")
    print()

    # --- Initialize components ---
    _progress("Loading speech model")
    asr = WhisperCppAdapter(
        model_path=model_path,
        language=cfg["asr"]["language"],
    )
    asr.load()
    asr.transcribe(np.zeros(16000, dtype=np.int16))
    _progress_done()

    _progress("Initializing microphone")
    mic = MicCapture(
        sample_rate=cfg["audio"]["sample_rate"],
        channels=cfg["audio"]["channels"],
        frame_duration_ms=cfg["vad"]["frame_size_ms"],
        device=cfg["audio"].get("device"),
        dtype=cfg["audio"]["dtype"],
    )
    # Keep mic stream running permanently — never open/close per recording.
    # This eliminates CoreAudio cold-start issues entirely.
    mic.start()
    _progress_done()

    _progress("Loading dictionary")
    transcript_buf = TranscriptBuffer()
    dictionary = load_dictionary()
    _progress_done()

    _progress("Warming up AI cleanup")
    cleanup = OllamaCleanup(
        model=cfg["cleanup"]["model"],
        base_url=cfg["cleanup"]["base_url"],
        timeout_ms=cfg["cleanup"]["timeout_ms"],
    )
    warmup_ok = cleanup.warmup()
    _progress_done()
    if not warmup_ok:
        print("  ⚠ Cleanup LLM unavailable — dictation works but without AI formatting.")

    # Restore stderr if we suppressed it
    if not VERBOSE:
        os.dup2(_stderr_fd, 2)
        os.close(_devnull)
        os.close(_stderr_fd)

    # Pre-load audio cues
    sound_start = AppKit.NSSound.alloc().initWithContentsOfFile_byReference_(
        "/System/Library/Sounds/Purr.aiff", True
    )
    sound_stop = AppKit.NSSound.alloc().initWithContentsOfFile_byReference_(
        "/System/Library/Sounds/Purr.aiff", True
    )
    sound_error = AppKit.NSSound.alloc().initWithContentsOfFile_byReference_(
        "/System/Library/Sounds/Basso.aiff", True
    )

    def play_sound(sound: AppKit.NSSound) -> None:
        """Play a sound without blocking the calling thread."""
        def _play() -> None:
            sound.stop()
            sound.play()
        threading.Thread(target=_play, daemon=True).start()

    injector = ClipboardInjector(
        paste_delay_ms=cfg["output"]["paste_delay_ms"],
    )

    latency = LatencyTracker(
        budgets={
            "asr": cfg["latency"]["asr_budget_ms"],
            "cleanup": cfg["latency"]["cleanup_budget_ms"],
            "injection": cfg["latency"]["injection_budget_ms"],
        },
        total_budget_ms=cfg["latency"]["total_budget_ms"],
    )

    # --- State (all access protected by a single lock) ---
    lock = threading.Lock()
    pressed_keys: set = set()
    recording = False
    recorded_frames: list[np.ndarray] = []
    utterance_id = 0
    recording_event = threading.Event()
    shutdown_event = threading.Event()

    def process_utterance(frames: list[np.ndarray], uid: int) -> None:
        """Process recorded audio through the full pipeline."""
        try:
            if not frames:
                print("  ⚠ No audio captured. Try holding the key a bit longer.")
                return

            audio = np.concatenate(frames).ravel()

            # Smart trim: only remove the first 300ms if it contains
            # the push-to-talk start sound (high energy burst).
            trim_samples = int(0.3 * cfg["audio"]["sample_rate"])
            if len(audio) > trim_samples * 2:
                head_energy = np.abs(audio[:trim_samples].astype(np.float32)).mean()
                body_energy = np.abs(audio[trim_samples:trim_samples * 3].astype(np.float32)).mean()
                if head_energy > body_energy * 2.5 and head_energy > 500:
                    audio = audio[trim_samples:]

            # Audio preprocessing: noise gate + gain normalization
            audio_f = audio.astype(np.float32)
            noise_floor = np.percentile(np.abs(audio_f), 10)
            gate_threshold = max(noise_floor * 3, 50.0)
            audio_f[np.abs(audio_f) < gate_threshold] = 0.0
            peak = np.max(np.abs(audio_f))
            if peak > 0:
                target_peak = 28000.0
                audio_f = audio_f * (target_peak / peak)
            audio = audio_f.clip(-32767, 32767).astype(np.int16)

            duration_s = len(audio) / cfg["audio"]["sample_rate"]
            _log(f"\n[Pipeline] Processing {duration_s:.1f}s of audio...")

            latency.start_pipeline()

            whisper_prompt = dictionary["whisper_hint"] if dictionary["whisper_hint"] else None

            # ASR
            with latency.track("asr"):
                raw_text = asr.transcribe(audio, initial_prompt=whisper_prompt)

            if not raw_text.strip():
                print("  ⚠ No speech detected. Make sure your mic is working.")
                return

            _log(f"[ASR] Raw: {raw_text}")

            # Early cancellation check
            with lock:
                if uid != utterance_id:
                    _log("[Pipeline] Cancelled — new recording started.")
                    return

            # Transcript normalization
            with latency.track("normalize"):
                normalized, has_backtrack = transcript_buf.add(raw_text)

            if not normalized:
                print("  ⚠ Nothing to transcribe.")
                return

            _log(f"[Normalize] {normalized}" + (" [BACKTRACK]" if has_backtrack else ""))

            # Check for voice commands
            cmd = detect_command(normalized)
            if cmd is not None:
                _log(f"[Command] Detected: {cmd.action}")
                with latency.track("command"):
                    execute_command(cmd)
                total = latency.finish_pipeline()
                _log(f"[Pipeline] Command executed in {total:.0f}ms")
                return

            cleanup_context = {
                "dictionary_hint": dictionary["prompt_hint"],
                "has_backtrack": has_backtrack,
            }

            # LLM cleanup
            with latency.track("cleanup"):
                cleaned = cleanup.cleanup(normalized, context=cleanup_context)

            _log(f"[Cleanup] {cleaned}")

            # Final cancellation check
            with lock:
                if uid != utterance_id:
                    _log("[Pipeline] Cancelled before injection.")
                    return

            # Injection
            with latency.track("injection"):
                injector.inject(cleaned)

            total = latency.finish_pipeline()
            if VERBOSE:
                print(latency.summary())
                _log(f"[Pipeline] Done in {total:.0f}ms")
            else:
                print(f"  ✔ \"{cleaned}\" ({total:.0f}ms)")

        except Exception as exc:
            print(f"[Pipeline] ERROR: {type(exc).__name__}: {exc}")

    # --- Hotkey handler (push-to-talk) ---
    hotkey_keys = set()
    _ALT_VARIANTS = {"alt_r", "alt_gr", "alt_l", "alt"}
    for k in cfg["hotkey"]["keys"]:
        key_name = k.strip("<>")
        try:
            hotkey_keys.add(keyboard.Key[key_name])
        except (KeyError, AttributeError):
            try:
                hotkey_keys.add(keyboard.KeyCode.from_char(k))
            except Exception:
                print(f"  ⚠ Could not parse hotkey '{k}', skipping.")

    if not hotkey_keys:
        print("  ✘ No valid hotkey configured.")
        sys.exit(1)

    def _is_hotkey_match(pressed: set) -> bool:
        """Check if hotkey is pressed, accounting for alt key variants."""
        if hotkey_keys.issubset(pressed):
            return True
        pressed_names = set()
        for k in pressed:
            if hasattr(k, 'name'):
                pressed_names.add(k.name)
        configured_names = set()
        for k in hotkey_keys:
            if hasattr(k, 'name'):
                configured_names.add(k.name)
        for cfg_name in configured_names:
            if cfg_name in _ALT_VARIANTS:
                if pressed_names & _ALT_VARIANTS:
                    return True
        return False

    def on_press(key: keyboard.Key | keyboard.KeyCode) -> None:
        nonlocal recording, recorded_frames, utterance_id

        with lock:
            pressed_keys.add(key)
            if not _is_hotkey_match(pressed_keys):
                return
            if recording:
                return
            utterance_id += 1
            recording = True
            recorded_frames = []

        play_sound(sound_start)
        # Mic stream is always running — just start collecting frames
        recording_event.set()
        print("  🎙️  Recording...")

    def on_release(key: keyboard.Key | keyboard.KeyCode) -> None:
        nonlocal recording, recorded_frames

        with lock:
            pressed_keys.discard(key)
            # Also discard all alt variants — macOS reports modifier keys
            # inconsistently (press=alt_r, release=alt), leaving phantom keys.
            if hasattr(key, 'name') and key.name in _ALT_VARIANTS:
                pressed_keys.difference_update({
                    k for k in list(pressed_keys)
                    if hasattr(k, 'name') and k.name in _ALT_VARIANTS
                })
            if not recording:
                return
            # Stop recording — clear pressed_keys to prevent any phantom buildup
            recording = False
            pressed_keys.clear()
            frames = list(recorded_frames)
            recorded_frames = []
            uid = utterance_id

        # Mic stream stays running — just stop collecting frames
        recording_event.clear()
        play_sound(sound_stop)
        print("  ⏳ Processing...")
        threading.Thread(target=process_utterance, args=(frames, uid), daemon=True).start()

    # --- Audio capture loop ---
    # Mic stream is always running. This loop drains frames continuously.
    # When recording=True, frames are stored. When False, frames are
    # discarded immediately to keep the queue empty for the next recording.
    def capture_loop() -> None:
        nonlocal recording
        while not shutdown_event.is_set():
            try:
                with lock:
                    is_rec = recording
                if is_rec:
                    # Recording: read frames one at a time and store them
                    frame = mic.read(timeout=0.03)
                    if frame is not None:
                        with lock:
                            if recording:
                                recorded_frames.append(frame)
                else:
                    # Not recording: drain ALL queued frames to keep queue empty.
                    # This ensures when recording starts, the queue has only
                    # fresh frames — not stale silence from seconds ago.
                    drained = mic.read_all()
                    if not drained:
                        time.sleep(0.03)  # nothing to drain, avoid busy loop
            except Exception as exc:
                _log(f"[Capture] ERROR: {type(exc).__name__}: {exc}")
                time.sleep(0.1)

    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()

    # --- Ready ---
    print()
    print("  ✔ Ready!")
    print()
    print("  Hold Right Option key to dictate. Ctrl+C to quit.")
    print()

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            pass

    # --- Shutdown ---
    print("\n  Shutting down...")
    shutdown_event.set()
    mic.stop()
    cleanup.close()
    asr.unload()
    capture_thread.join(timeout=2)
    print("  Bye.")


if __name__ == "__main__":
    main()
