"""Main entry point — pipeline orchestrator with push-to-talk.

Audio architecture (Wispr Flow-style):
  - Mic stream runs permanently from startup
  - A rolling buffer always holds the last ~500ms of audio (pre-buffer)
  - On key press: pre-buffer is captured + new frames are collected
  - On key release: VAD trims silence, audio sent to ASR
  - No stream open/close per recording — instant, reliable
"""

from __future__ import annotations

import collections
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
    if VERBOSE:
        print(msg)


def _progress(step: str, done: bool = False) -> None:
    if done:
        print(f"  ✔ {step}")
    else:
        print(f"  ⏳ {step}...", end="", flush=True)


def _progress_done() -> None:
    print(" ✔")


def load_config(path: Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config" / "default.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)


def main() -> None:
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
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / cfg["asr"]["model_path"]
    sample_rate = cfg["audio"]["sample_rate"]
    frame_ms = cfg["vad"]["frame_size_ms"]
    frame_size = int(sample_rate * frame_ms / 1000)

    # Suppress whisper.cpp verbose output unless --verbose
    if not VERBOSE:
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
    asr = WhisperCppAdapter(model_path=model_path, language=cfg["asr"]["language"])
    asr.load()
    asr.transcribe(np.zeros(16000, dtype=np.int16))
    _progress_done()

    _progress("Initializing microphone")
    mic = MicCapture(
        sample_rate=sample_rate,
        channels=cfg["audio"]["channels"],
        frame_duration_ms=frame_ms,
        device=cfg["audio"].get("device"),
        dtype=cfg["audio"]["dtype"],
    )
    mic.start()  # Permanent — never closed until shutdown
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

    if not VERBOSE:
        os.dup2(_stderr_fd, 2)
        os.close(_devnull)
        os.close(_stderr_fd)

    # Audio cues
    sound_start = AppKit.NSSound.alloc().initWithContentsOfFile_byReference_(
        "/System/Library/Sounds/Purr.aiff", True)
    sound_stop = AppKit.NSSound.alloc().initWithContentsOfFile_byReference_(
        "/System/Library/Sounds/Purr.aiff", True)
    sound_error = AppKit.NSSound.alloc().initWithContentsOfFile_byReference_(
        "/System/Library/Sounds/Basso.aiff", True)

    def play_sound(sound: AppKit.NSSound) -> None:
        def _play() -> None:
            sound.stop()
            sound.play()
        threading.Thread(target=_play, daemon=True).start()

    injector = ClipboardInjector(paste_delay_ms=cfg["output"]["paste_delay_ms"])
    latency = LatencyTracker(
        budgets={
            "asr": cfg["latency"]["asr_budget_ms"],
            "cleanup": cfg["latency"]["cleanup_budget_ms"],
            "injection": cfg["latency"]["injection_budget_ms"],
        },
        total_budget_ms=cfg["latency"]["total_budget_ms"],
    )

    # =====================================================================
    # Rolling audio buffer (Wispr Flow-style)
    # =====================================================================
    # Always holds the last ~500ms of audio from the mic, even when not
    # recording. When the user presses the hotkey, these pre-buffered
    # frames are included — capturing speech that started just before
    # the keypress. This eliminates the "missed first word" problem.
    PRE_BUFFER_MS = 500
    pre_buffer_size = int(PRE_BUFFER_MS / frame_ms)  # ~17 frames at 30ms
    pre_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=pre_buffer_size)


    # --- Shared state ---
    lock = threading.Lock()
    pressed_keys: set = set()
    recording = False
    recorded_frames: list[np.ndarray] = []
    utterance_id = 0
    shutdown_event = threading.Event()

    # =====================================================================
    # Audio capture loop — always running, feeds rolling buffer + recording
    # =====================================================================
    def capture_loop() -> None:
        while not shutdown_event.is_set():
            try:
                frame = mic.read(timeout=0.05)
                if frame is None:
                    continue
                with lock:
                    if recording:
                        recorded_frames.append(frame)
                    else:
                        pre_buffer.append(frame)
            except Exception as exc:
                _log(f"[Capture] ERROR: {type(exc).__name__}: {exc}")
                time.sleep(0.05)

    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()

    # =====================================================================
    # Pipeline
    # =====================================================================
    def process_utterance(frames: list[np.ndarray], uid: int) -> None:
        try:
            if not frames:
                print("  ⚠ No audio captured. Try holding the key a bit longer.")
                return

            audio = np.concatenate(frames).ravel()

            # VAD-style silence trimming: find where speech starts/ends
            # by looking at energy in 30ms windows
            energy_per_frame = []
            for i in range(0, len(audio) - frame_size, frame_size):
                chunk = audio[i:i + frame_size].astype(np.float32)
                energy_per_frame.append(np.abs(chunk).mean())

            if energy_per_frame:
                threshold = max(np.percentile(energy_per_frame, 20) * 3, 100)
                # Find first frame above threshold
                start_idx = 0
                for i, e in enumerate(energy_per_frame):
                    if e > threshold:
                        start_idx = max(0, i - 2)  # keep 2 frames before speech
                        break
                # Find last frame above threshold
                end_idx = len(energy_per_frame)
                for i in range(len(energy_per_frame) - 1, -1, -1):
                    if energy_per_frame[i] > threshold:
                        end_idx = min(len(energy_per_frame), i + 3)  # keep 3 after
                        break
                audio = audio[start_idx * frame_size:end_idx * frame_size]

            if len(audio) < sample_rate * 0.1:  # less than 100ms
                print("  ⚠ Recording too short.")
                return

            # Gain normalization
            audio_f = audio.astype(np.float32)
            peak = np.max(np.abs(audio_f))
            if peak > 0:
                audio_f = audio_f * (28000.0 / peak)
            audio = audio_f.clip(-32767, 32767).astype(np.int16)

            duration_s = len(audio) / sample_rate
            _log(f"[Pipeline] Processing {duration_s:.1f}s of audio...")

            latency.start_pipeline()

            whisper_prompt = dictionary["whisper_hint"] if dictionary["whisper_hint"] else None

            with latency.track("asr"):
                raw_text = asr.transcribe(audio, initial_prompt=whisper_prompt)

            if not raw_text.strip():
                print("  ⚠ No speech detected. Make sure your mic is working.")
                return

            _log(f"[ASR] Raw: {raw_text}")

            with lock:
                if uid != utterance_id:
                    _log("[Pipeline] Cancelled — new recording started.")
                    return

            with latency.track("normalize"):
                normalized, has_backtrack = transcript_buf.add(raw_text)

            if not normalized:
                print("  ⚠ Nothing to transcribe.")
                return

            _log(f"[Normalize] {normalized}" + (" [BACKTRACK]" if has_backtrack else ""))

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

            with latency.track("cleanup"):
                cleaned = cleanup.cleanup(normalized, context=cleanup_context)

            _log(f"[Cleanup] {cleaned}")

            with lock:
                if uid != utterance_id:
                    _log("[Pipeline] Cancelled before injection.")
                    return

            with latency.track("injection"):
                injector.inject(cleaned)

            total = latency.finish_pipeline()
            if VERBOSE:
                print(latency.summary())
                _log(f"[Pipeline] Done in {total:.0f}ms")
            else:
                print(f"  ✔ \"{cleaned}\" ({total:.0f}ms)")

        except Exception as exc:
            print(f"  ⚠ Error: {type(exc).__name__}: {exc}")

    # =====================================================================
    # Hotkey handler
    # =====================================================================
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
        if hotkey_keys.issubset(pressed):
            return True
        pressed_names = {getattr(k, 'name', None) for k in pressed} - {None}
        configured_names = {getattr(k, 'name', None) for k in hotkey_keys} - {None}
        for cfg_name in configured_names:
            if cfg_name in _ALT_VARIANTS and pressed_names & _ALT_VARIANTS:
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
            # Grab the pre-buffer — this captures ~500ms of audio from
            # BEFORE the keypress, so the first word isn't cut off.
            recorded_frames = list(pre_buffer)
            pre_buffer.clear()

        play_sound(sound_start)
        print("  🎙️  Recording...")

    def on_release(key: keyboard.Key | keyboard.KeyCode) -> None:
        nonlocal recording, recorded_frames

        with lock:
            pressed_keys.discard(key)
            if hasattr(key, 'name') and key.name in _ALT_VARIANTS:
                pressed_keys.difference_update({
                    k for k in list(pressed_keys)
                    if hasattr(k, 'name') and k.name in _ALT_VARIANTS
                })
            if not recording:
                return
            recording = False
            pressed_keys.clear()
            frames = list(recorded_frames)
            recorded_frames = []
            uid = utterance_id

        play_sound(sound_stop)
        print("  ⏳ Processing...")
        threading.Thread(target=process_utterance, args=(frames, uid), daemon=True).start()

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

    print("\n  Shutting down...")
    shutdown_event.set()
    mic.stop()
    cleanup.close()
    asr.unload()
    capture_thread.join(timeout=2)
    print("  Bye.")


if __name__ == "__main__":
    main()
