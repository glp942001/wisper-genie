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


def _build_whisper_prompt(
    recent_utterances: list[str],
    dictionary_hint: str,
    max_chars: int = 224,
) -> str | None:
    """Build a short initial prompt for Whisper.

    Keep recent dictated context plus dictionary terms, but bias toward
    preserving the vocabulary hint because it has the biggest ASR impact.
    """
    recent_text = " ".join(u.strip() for u in recent_utterances if u.strip()).strip()
    dictionary_hint = dictionary_hint.strip()

    if not recent_text and not dictionary_hint:
        return None

    if dictionary_hint and recent_text:
        remaining = max_chars - len(dictionary_hint) - 2
        if remaining > 0:
            recent_text = recent_text[-remaining:].lstrip()
        else:
            recent_text = ""

    parts = [part for part in (recent_text, dictionary_hint) if part]
    prompt = ". ".join(parts).strip()
    return prompt[:max_chars] if prompt else None


def _trim_audio_energy(
    audio: np.ndarray,
    frame_size: int,
    *,
    pre_speech_frames: int = 2,
    post_speech_frames: int = 3,
) -> np.ndarray:
    """Trim leading/trailing silence with a conservative energy heuristic."""
    if len(audio) < frame_size:
        return audio

    energy_per_frame = []
    for i in range(0, len(audio) - frame_size + 1, frame_size):
        chunk = audio[i:i + frame_size].astype(np.float32)
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        energy_per_frame.append(rms)

    if not energy_per_frame:
        return audio

    noise_floor = float(np.percentile(energy_per_frame, 20))
    speech_peak = float(np.percentile(energy_per_frame, 95))
    threshold = max(noise_floor * 2.0, noise_floor + (speech_peak - noise_floor) * 0.2, 80.0)
    speech_frames = [idx for idx, energy in enumerate(energy_per_frame) if energy >= threshold]
    if not speech_frames:
        return audio

    start_idx = max(0, speech_frames[0] - pre_speech_frames)
    end_idx = min(len(energy_per_frame), speech_frames[-1] + post_speech_frames + 1)
    return audio[start_idx * frame_size:end_idx * frame_size]


def _normalize_audio(
    audio: np.ndarray,
    *,
    target_peak: float = 28000.0,
    max_gain: float = 4.0,
    min_rms: float = 250.0,
) -> np.ndarray:
    """Normalize speech conservatively without boosting low-energy noise."""
    if audio.size == 0:
        return audio

    audio_f = audio.astype(np.float32)
    peak = float(np.max(np.abs(audio_f)))
    if peak <= 0:
        return audio.astype(np.int16)

    rms = float(np.sqrt(np.mean(np.square(audio_f))))
    if rms < min_rms:
        return audio.astype(np.int16)

    gain = min(target_peak / peak, max_gain)
    if gain <= 1.0:
        return audio.astype(np.int16)

    normalized = np.clip(audio_f * gain, -32767, 32767)
    return normalized.astype(np.int16)


def main() -> None:
    import AppKit
    from pynput import keyboard

    from dictation.asr.whisper_cpp import WhisperCppAdapter
    from dictation.audio.capture import MicCapture
    from dictation.cleanup.ollama import OllamaCleanup
    from dictation.commands.handler import detect_command, execute_command
    from dictation.context.dictionary import load_dictionary
    from dictation.context.screen import get_screen_context
    from dictation.output.injector import ClipboardInjector
    from dictation.telemetry.latency import LatencyTracker
    from dictation.transcript.buffer import TranscriptBuffer

    cfg = load_config()
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / cfg["asr"]["model_path"]
    sample_rate = cfg["audio"]["sample_rate"]
    frame_ms = cfg["vad"]["frame_size_ms"]
    frame_size = int(sample_rate * frame_ms / 1000)
    pre_speech_frames = cfg["vad"].get("pre_speech_frames", 2)
    post_speech_frames = cfg["vad"].get("post_speech_frames", 3)
    normalize_target_peak = cfg["audio"].get("normalize_target_peak", 28000)
    normalize_max_gain = cfg["audio"].get("normalize_max_gain", 4.0)
    normalize_min_rms = cfg["audio"].get("normalize_min_rms", 250)
    capture_restart_timeout_s = cfg["audio"].get("capture_restart_timeout_ms", 2000) / 1000.0

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
    def play_sound(sound: AppKit.NSSound) -> None:
        def _play() -> None:
            sound.stop()
            sound.play()
        threading.Thread(target=_play, daemon=True).start()

    injector = ClipboardInjector(paste_delay_ms=cfg["output"]["paste_delay_ms"])
    latency_budgets = {
        "asr": cfg["latency"]["asr_budget_ms"],
        "cleanup": cfg["latency"]["cleanup_budget_ms"],
        "injection": cfg["latency"]["injection_budget_ms"],
    }
    total_latency_budget_ms = cfg["latency"]["total_budget_ms"]

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
        consecutive_timeouts = 0
        last_restart_at = 0.0
        while not shutdown_event.is_set():
            try:
                frame = mic.read(timeout=0.05)
                if frame is None:
                    consecutive_timeouts += 1
                    now = time.monotonic()
                    if consecutive_timeouts * 0.05 >= capture_restart_timeout_s and now - last_restart_at >= 1.0:
                        print("  ⚠ Microphone stream stalled — restarting input.")
                        mic.restart()
                        consecutive_timeouts = 0
                        last_restart_at = now
                    continue
                consecutive_timeouts = 0
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
            latency = LatencyTracker(
                budgets=latency_budgets,
                total_budget_ms=total_latency_budget_ms,
            )
            if not frames:
                print("  ⚠ No audio captured. Try holding the key a bit longer.")
                return

            audio = np.concatenate(frames).ravel()
            audio = _trim_audio_energy(
                audio,
                frame_size=frame_size,
                pre_speech_frames=pre_speech_frames,
                post_speech_frames=post_speech_frames,
            )

            if len(audio) < sample_rate * 0.1:  # less than 100ms
                print("  ⚠ Recording too short.")
                return

            audio = _normalize_audio(
                audio,
                target_peak=normalize_target_peak,
                max_gain=normalize_max_gain,
                min_rms=normalize_min_rms,
            )

            duration_s = len(audio) / sample_rate
            _log(f"[Pipeline] Processing {duration_s:.1f}s of audio...")

            latency.start_pipeline()

            recent_utterances = transcript_buf.get_last(3)
            whisper_prompt = _build_whisper_prompt(
                recent_utterances=recent_utterances,
                dictionary_hint=dictionary["whisper_hint"],
            )

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
                    executed = execute_command(cmd)
                total = latency.finish_pipeline()
                if executed:
                    _log(f"[Pipeline] Command executed in {total:.0f}ms")
                else:
                    print("  ⚠ Command could not be applied safely.")
                return

            with latency.track("context"):
                screen_context = get_screen_context()

            with lock:
                if uid != utterance_id:
                    _log("[Pipeline] Cancelled before cleanup.")
                    return

            cleanup_context = {
                "app_name": screen_context.get("app_name", ""),
                "field_text": screen_context.get("field_text", ""),
                "recent_utterances": recent_utterances,
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

            transcript_buf.commit(cleaned)

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
