"""Main entry point — pipeline orchestrator with push-to-talk."""

from __future__ import annotations

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

    print("=== Dictation Pipeline ===")
    print(f"Hotkey: {' + '.join(cfg['hotkey']['keys'])}")
    print()

    # --- Initialize components ---
    print("Loading ASR model...")
    asr = WhisperCppAdapter(
        model_path=model_path,
        language=cfg["asr"]["language"],
    )
    asr.load()
    # Warmup: dummy transcription to eliminate cold-start GPU penalty
    asr.transcribe(np.zeros(16000, dtype=np.int16))
    print("ASR model loaded and warm.")

    mic = MicCapture(
        sample_rate=cfg["audio"]["sample_rate"],
        channels=cfg["audio"]["channels"],
        frame_duration_ms=cfg["vad"]["frame_size_ms"],
        device=cfg["audio"].get("device"),
        dtype=cfg["audio"]["dtype"],
    )

    transcript_buf = TranscriptBuffer()

    # Load personal dictionary
    dictionary = load_dictionary()
    if dictionary["terms"]:
        print(f"Personal dictionary: {len(dictionary['terms'])} terms loaded.")
    else:
        print("No personal dictionary found. Add terms to config/dictionary.toml.")

    cleanup = OllamaCleanup(
        model=cfg["cleanup"]["model"],
        base_url=cfg["cleanup"]["base_url"],
        timeout_ms=cfg["cleanup"]["timeout_ms"],
    )

    print("Warming up cleanup LLM...")
    warmup_ok = cleanup.warmup()
    if warmup_ok:
        print("Cleanup LLM warm.")
    else:
        print("WARNING: Cleanup LLM unavailable. Dictation will work but without text cleanup.")

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
    # Event signals capture_loop to wake up when recording starts/stops
    recording_event = threading.Event()
    # Shutdown signal
    shutdown_event = threading.Event()

    def process_utterance(frames: list[np.ndarray], uid: int) -> None:
        """Process recorded audio through the full pipeline."""
        try:
            if not frames:
                return

            audio = np.concatenate(frames).ravel()

            # Smart trim: only remove the first 300ms if it contains
            # the push-to-talk start sound (high energy burst).
            # If the beginning is quiet (speech or silence), keep it.
            trim_samples = int(0.3 * cfg["audio"]["sample_rate"])  # 4800 @ 16kHz
            if len(audio) > trim_samples * 2:
                head_energy = np.abs(audio[:trim_samples].astype(np.float32)).mean()
                body_energy = np.abs(audio[trim_samples:trim_samples * 3].astype(np.float32)).mean()
                # Trim only if the head is significantly louder than the body
                # (i.e., a sound effect burst, not normal speech)
                if head_energy > body_energy * 2.5 and head_energy > 500:
                    audio = audio[trim_samples:]

            duration_s = len(audio) / cfg["audio"]["sample_rate"]
            print(f"\n[Pipeline] Processing {duration_s:.1f}s of audio...")

            latency.start_pipeline()

            # Build Whisper initial prompt from dictionary terms ONLY.
            # Do NOT include recent utterances — Whisper hallucinates from them
            # when audio is unclear, producing the previous text instead of new speech.
            whisper_prompt = dictionary["whisper_hint"] if dictionary["whisper_hint"] else None

            # ASR
            with latency.track("asr"):
                raw_text = asr.transcribe(audio, initial_prompt=whisper_prompt)

            if not raw_text.strip():
                print("[Pipeline] No speech detected.")
                return

            print(f"[ASR] Raw: {raw_text}")

            # Early cancellation check — don't waste LLM call if stale
            with lock:
                if uid != utterance_id:
                    print("[Pipeline] Cancelled after ASR — new recording started.")
                    return

            # Transcript normalization (fillers, dictation commands)
            with latency.track("normalize"):
                normalized, has_backtrack = transcript_buf.add(raw_text)

            if not normalized:
                print("[Pipeline] Empty after normalization.")
                return

            print(f"[Normalize] {normalized}" + (" [BACKTRACK]" if has_backtrack else ""))

            # Check for voice commands (delete that, undo, etc.)
            cmd = detect_command(normalized)
            if cmd is not None:
                print(f"[Command] Detected: {cmd.action}")
                with latency.track("command"):
                    execute_command(cmd)
                total = latency.finish_pipeline()
                print(f"[Pipeline] Command executed in {total:.0f}ms")
                return

            cleanup_context = {
                "dictionary_hint": dictionary["prompt_hint"],
                "has_backtrack": has_backtrack,
            }

            # LLM cleanup
            with latency.track("cleanup"):
                cleaned = cleanup.cleanup(normalized, context=cleanup_context)

            print(f"[Cleanup] {cleaned}")

            # Final cancellation check before injecting
            with lock:
                if uid != utterance_id:
                    print("[Pipeline] Cancelled before injection — new recording started.")
                    return

            # Injection
            with latency.track("injection"):
                injector.inject(cleaned)

            total = latency.finish_pipeline()
            print(latency.summary())
            print(f"[Pipeline] Done in {total:.0f}ms")

        except Exception as exc:
            print(f"[Pipeline] ERROR: {type(exc).__name__}: {exc}")

    # --- Hotkey handler (push-to-talk) ---
    hotkey_keys = set()
    for k in cfg["hotkey"]["keys"]:
        try:
            hotkey_keys.add(keyboard.Key[k.strip("<>")])
        except (KeyError, AttributeError):
            try:
                hotkey_keys.add(keyboard.KeyCode.from_char(k))
            except Exception:
                print(f"WARNING: Could not parse hotkey '{k}', skipping.")

    if not hotkey_keys:
        print("ERROR: No valid hotkey configured. Check [hotkey] keys in config/default.toml.")
        sys.exit(1)

    def on_press(key: keyboard.Key | keyboard.KeyCode) -> None:
        nonlocal recording, recorded_frames, utterance_id

        with lock:
            pressed_keys.add(key)
            if not hotkey_keys.issubset(pressed_keys):
                return
            if recording:
                return
            utterance_id += 1
            recording = True
            recorded_frames = []

        play_sound(sound_start)
        mic.start()
        recording_event.set()
        print("\n[Recording] Started — speak now...")

    def on_release(key: keyboard.Key | keyboard.KeyCode) -> None:
        nonlocal recording, recorded_frames

        with lock:
            pressed_keys.discard(key)
            if not recording or hotkey_keys.issubset(pressed_keys):
                return
            recording = False
            frames = list(recorded_frames)
            recorded_frames = []
            uid = utterance_id

        mic.stop()
        recording_event.clear()
        play_sound(sound_stop)
        print("[Recording] Stopped")
        threading.Thread(target=process_utterance, args=(frames, uid), daemon=True).start()

    # --- Audio capture loop (event-driven) ---
    def capture_loop() -> None:
        """Continuously read mic frames when recording."""
        nonlocal recording
        consecutive_errors = 0
        while not shutdown_event.is_set():
            # Wait for recording to start (or shutdown)
            recording_event.wait(timeout=0.1)
            if shutdown_event.is_set():
                break

            try:
                with lock:
                    is_rec = recording
                if is_rec:
                    frame = mic.read(timeout=0.05)
                    if frame is not None:
                        with lock:
                            if recording:  # Double-check still recording
                                recorded_frames.append(frame)
                        consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                print(f"[Capture] ERROR: {type(exc).__name__}: {exc}")
                with lock:
                    if recording:
                        recording = False
                        recording_event.clear()
                play_sound(sound_error)
                print("[Capture] Recording lost — mic error. Release and try again.")
                if consecutive_errors >= 3:
                    print("[Capture] Multiple consecutive errors. Check microphone connection.")
                    consecutive_errors = 0

    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()

    # --- Start ---
    print("\nReady! Hold the hotkey to dictate. Press Ctrl+C to quit.")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            pass

    # --- Shutdown ---
    print("\nShutting down...")
    shutdown_event.set()
    recording_event.set()  # Wake capture loop so it can exit
    mic.stop()
    cleanup.close()
    asr.unload()
    capture_thread.join(timeout=2)
    print("Bye.")


if __name__ == "__main__":
    main()
