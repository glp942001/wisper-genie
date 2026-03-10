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
import re
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
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_PHRASE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'._/-]*")


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


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}


def _phrase_tokens(text: str) -> set[str]:
    return {token.lower() for token in _PHRASE_RE.findall(text) if len(token) >= 3}


def _context_tokens(
    *,
    field_text: str,
    selected_text: str,
    recent_utterances: list[str],
    dictionary_terms: list[str],
) -> set[str]:
    tokens = set()
    for chunk in [field_text, selected_text, " ".join(recent_utterances), " ".join(dictionary_terms)]:
        tokens.update(_tokenize(chunk))
    return tokens


def _compact_prompt_fragment(text: str, *, max_words: int = 10, max_chars: int = 96) -> str:
    words = [word for word in _PHRASE_RE.findall(text) if word]
    if not words:
        return ""
    fragment = " ".join(words[-max_words:])
    return fragment[-max_chars:].lstrip()


def _build_whisper_prompt_variants(
    *,
    recent_utterances: list[str],
    dictionary_hint: str,
    field_text: str,
    selected_text: str,
    max_hypotheses: int,
) -> list[tuple[str, str | None]]:
    prompted_variants: list[tuple[str, str | None]] = []
    seen_prompts: set[str] = set()

    def add(source: str, prompt: str | None) -> None:
        normalized = (prompt or "").strip()
        if not normalized:
            return
        key = normalized
        if key in seen_prompts:
            return
        seen_prompts.add(key)
        prompted_variants.append((source, normalized or None))

    add(
        "prompted_full",
        _build_whisper_prompt(
            recent_utterances=recent_utterances,
            dictionary_hint=dictionary_hint,
        ),
    )

    focus_fragment = _compact_prompt_fragment(selected_text or field_text)
    if focus_fragment:
        add(
            "prompted_focus",
            _build_whisper_prompt(
                recent_utterances=[focus_fragment],
                dictionary_hint=dictionary_hint,
                max_chars=180,
            ),
        )

    recent_prompt = _build_whisper_prompt(
        recent_utterances=recent_utterances,
        dictionary_hint="",
        max_chars=160,
    )
    add("prompted_recent", recent_prompt)

    add(
        "prompted_dict",
        _build_whisper_prompt(
            recent_utterances=[],
            dictionary_hint=dictionary_hint,
            max_chars=180,
        ),
    )
    if max_hypotheses <= 1:
        return [("unprompted", None)]
    return prompted_variants[: max_hypotheses - 1] + [("unprompted", None)]


def _select_best_candidate(
    candidates: list,
    *,
    field_text: str,
    selected_text: str,
    recent_utterances: list[str],
    dictionary_terms: list[str],
):
    context_terms = _context_tokens(
        field_text=field_text,
        selected_text=selected_text,
        recent_utterances=recent_utterances,
        dictionary_terms=dictionary_terms,
    )
    selected_terms = _phrase_tokens(selected_text)
    field_terms = _phrase_tokens(field_text)
    history_terms = _phrase_tokens(" ".join(recent_utterances))
    dict_terms = _phrase_tokens(" ".join(dictionary_terms))

    def score(candidate) -> float:
        candidate_tokens = _tokenize(candidate.text)
        candidate_phrases = _phrase_tokens(candidate.text)
        overlap = 0.0
        if candidate_tokens and context_terms:
            overlap = len(candidate_tokens & context_terms) / len(candidate_tokens)
        selected_overlap = 0.0
        if candidate_phrases and selected_terms:
            selected_overlap = len(candidate_phrases & selected_terms) / len(candidate_phrases)
        field_overlap = 0.0
        if candidate_phrases and field_terms:
            field_overlap = len(candidate_phrases & field_terms) / len(candidate_phrases)
        history_overlap = 0.0
        if candidate_phrases and history_terms:
            history_overlap = len(candidate_phrases & history_terms) / len(candidate_phrases)
        dictionary_overlap = 0.0
        if candidate_phrases and dict_terms:
            dictionary_overlap = len(candidate_phrases & dict_terms) / len(candidate_phrases)
        source_bonus = 0.02 if candidate.source.startswith("prompted") else 0.0
        return (
            candidate.confidence * 0.70
            + overlap * 0.10
            + selected_overlap * 0.10
            + field_overlap * 0.04
            + history_overlap * 0.03
            + dictionary_overlap * 0.03
            + source_bonus
        )

    non_empty = [candidate for candidate in candidates if candidate.text.strip()]
    if not non_empty:
        return candidates[0]
    return max(non_empty, key=score)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in {"autostart", "metrics", "uninstall"}:
        from dictation.cli import main as cli_main

        cli_main()
        return

    import AppKit
    from pynput import keyboard

    from dictation.asr.whisper_cpp import WhisperCppAdapter
    from dictation.audio.capture import MicCapture
    from dictation.audio.vad import SileroVAD
    from dictation.cleanup.ollama import OllamaCleanup
    from dictation.commands.handler import detect_command, execute_command
    from dictation.context.cache import ScreenContextCache
    from dictation.context.dictionary import load_dictionary
    from dictation.context.style_memory import StyleMemory
    from dictation.output.injector import ClipboardInjector
    from dictation.output.overlay import GhostPreviewOverlay
    from dictation.telemetry.latency import LatencyTracker
    from dictation.telemetry.metrics import RoutingMetrics
    from dictation.transcript.buffer import TranscriptBuffer

    cfg = load_config()
    input_cfg = cfg.get("input", {})
    context_cfg = cfg.get("context", {})
    preview_cfg = cfg.get("preview", {})
    routing_cfg = cfg.get("routing", {})
    metrics_cfg = cfg.get("metrics", {})
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
    input_mode = input_cfg.get("mode", "push_to_talk")
    direct_insert_enabled = cfg["output"].get("prefer_direct_insert", True)
    live_silence_ms = input_cfg.get("live_silence_duration_ms", 450)
    live_min_frames = max(2, int(input_cfg.get("live_min_utterance_ms", 180) / frame_ms))
    routing_alt_enabled = routing_cfg.get("alternative_hypotheses_enabled", True)
    routing_alt_threshold = routing_cfg.get("alternative_hypothesis_threshold", 0.72)
    routing_max_hypotheses = max(2, int(routing_cfg.get("max_hypotheses", 4)))
    context_cache_ttl_ms = context_cfg.get("cache_ttl_ms", 1200)
    preview_before_insert_s = preview_cfg.get("before_insert_delay_ms", 180) / 1000.0
    metrics_enabled = metrics_cfg.get("enabled", True)
    typing_fallback_enabled = cfg["output"].get("prefer_typing_fallback", True)
    typing_max_chars = int(cfg["output"].get("typing_max_chars", 220))

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
    style_memory = StyleMemory()
    context_cache = ScreenContextCache(ttl_ms=context_cache_ttl_ms)
    routing_metrics = RoutingMetrics() if metrics_enabled else None
    overlay = GhostPreviewOverlay(
        enabled=preview_cfg.get("enabled", True),
        preview_ms=preview_cfg.get("display_ms", 1200),
        status_ms=preview_cfg.get("status_ms", 500),
        max_chars=preview_cfg.get("max_chars", 120),
    )
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

    live_vad = None
    if input_mode == "live":
        _progress("Loading live-mode VAD")
        live_vad = SileroVAD(
            sample_rate=sample_rate,
            threshold=cfg["vad"].get("threshold", 0.5),
            silence_duration_ms=live_silence_ms,
            frame_size_ms=frame_ms,
        )
        _progress_done()

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

    injector = ClipboardInjector(
        paste_delay_ms=cfg["output"]["paste_delay_ms"],
        prefer_direct_insertion=direct_insert_enabled,
        prefer_typing_fallback=typing_fallback_enabled,
        typing_max_chars=typing_max_chars,
    )
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
    recording_source: str | None = None
    utterance_id = 0
    shutdown_event = threading.Event()

    def _prefetch_context() -> None:
        try:
            context_cache.prefetch(include_full_text=direct_insert_enabled)
        except Exception as exc:
            _log(f"[Context] Prefetch failed: {type(exc).__name__}: {exc}")

    def _persist_post_insert(
        *,
        cleaned: str,
        event: dict | None = None,
    ) -> None:
        def _persist() -> None:
            try:
                style_memory.observe(cleaned)
            except Exception as exc:
                _log(f"[StyleMemory] Persist failed: {type(exc).__name__}: {exc}")
            if routing_metrics is None or event is None:
                return
            try:
                routing_metrics.record(event)
            except Exception as exc:
                _log(f"[Metrics] Persist failed: {type(exc).__name__}: {exc}")

        threading.Thread(target=_persist, daemon=True).start()

    # =====================================================================
    # Audio capture loop — always running, feeds rolling buffer + recording
    # =====================================================================
    def capture_loop() -> None:
        nonlocal recording, recorded_frames, utterance_id, recording_source
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

                if input_mode == "live" and live_vad is not None:
                    with lock:
                        if recording:
                            recorded_frames.append(frame)
                        else:
                            pre_buffer.append(frame)

                    was_speaking = live_vad.is_speaking
                    live_vad.process_frame(frame)
                    is_speaking = live_vad.is_speaking

                    if not was_speaking and is_speaking:
                        start_live = False
                        with lock:
                            if not recording:
                                utterance_id += 1
                                recording = True
                                recording_source = "live"
                                recorded_frames = list(pre_buffer)
                                pre_buffer.clear()
                                start_live = True
                        if start_live:
                            injector.clear_clipboard()
                            _prefetch_context()
                            overlay.show_status("Listening…")

                    if was_speaking and not is_speaking:
                        with lock:
                            if recording and recording_source == "live":
                                recording = False
                                recording_source = None
                                frames = list(recorded_frames)
                                recorded_frames = []
                                uid = utterance_id
                            else:
                                frames = []
                                uid = utterance_id
                        live_vad.reset()
                        if len(frames) >= live_min_frames:
                            overlay.show_status("Processing…")
                            threading.Thread(
                                target=process_utterance,
                                args=(frames, uid, "live"),
                                daemon=True,
                            ).start()
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
    def process_utterance(frames: list[np.ndarray], uid: int, trigger_mode: str) -> None:
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

            with latency.track("context"):
                context_details = context_cache.get(include_full_text=direct_insert_enabled)

            screen_context = {
                "field_text": context_details.field_text,
                "selected_text": context_details.selected_text,
            }
            recent_utterances = transcript_buf.get_last(3)
            prompt_variants = _build_whisper_prompt_variants(
                recent_utterances=recent_utterances,
                dictionary_hint=dictionary["whisper_hint"],
                field_text=screen_context["field_text"],
                selected_text=screen_context["selected_text"],
                max_hypotheses=routing_max_hypotheses,
            )

            with latency.track("asr"):
                primary_source, primary_prompt = prompt_variants[0]
                primary_candidate = asr.transcribe_candidate(
                    audio,
                    initial_prompt=primary_prompt,
                    source=primary_source,
                )
                candidates = [primary_candidate]
                if routing_alt_enabled and primary_candidate.confidence < routing_alt_threshold:
                    for source, prompt in prompt_variants[1:]:
                        candidate = asr.transcribe_candidate(
                            audio,
                            initial_prompt=prompt,
                            source=source,
                        )
                        if not candidate.text.strip():
                            continue
                        if any(existing.text == candidate.text for existing in candidates):
                            continue
                        candidates.append(candidate)

                selected_candidate = _select_best_candidate(
                    candidates,
                    field_text=screen_context["field_text"],
                    selected_text=screen_context["selected_text"],
                    recent_utterances=recent_utterances,
                    dictionary_terms=dictionary["terms"],
                )
                raw_text = selected_candidate.text

            if not raw_text.strip():
                print("  ⚠ No speech detected. Make sure your mic is working.")
                return

            _log(f"[ASR] Raw: {raw_text} [{selected_candidate.confidence:.2f}]")

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
                if routing_metrics is not None:
                    routing_metrics.record(
                        {
                            "event": "command",
                            "trigger_mode": trigger_mode,
                            "command": cmd.action,
                            "asr_confidence": round(selected_candidate.confidence, 3),
                            "candidate_count": len(candidates),
                            "candidate_sources": [candidate.source for candidate in candidates],
                            "selected_candidate": selected_candidate.source,
                            "timings": latency.timings,
                        }
                    )
                return

            with lock:
                if uid != utterance_id:
                    _log("[Pipeline] Cancelled before cleanup.")
                    return

            cleanup_context = {
                "field_text": screen_context.get("field_text", ""),
                "selected_text": screen_context.get("selected_text", ""),
                "recent_utterances": recent_utterances,
                "dictionary_hint": dictionary["prompt_hint"],
                "style_hint": style_memory.build_prompt_hint(),
                "has_backtrack": has_backtrack,
            }

            with latency.track("cleanup"):
                cleaned = cleanup.cleanup(normalized, context=cleanup_context)

            _log(f"[Cleanup] {cleaned}")

            with lock:
                if uid != utterance_id:
                    _log("[Pipeline] Cancelled before injection.")
                    return

            with latency.track("preview"):
                overlay.show_preview(cleaned)
                if preview_before_insert_s > 0:
                    time.sleep(preview_before_insert_s)
                overlay.hide()
                time.sleep(0.03)
            with latency.track("injection"):
                inject_route = injector.inject(
                    cleaned,
                    prefer_direct=direct_insert_enabled,
                    context_details=context_details,
                )
            _log(f"[Injection] Route: {inject_route}")

            transcript_buf.commit(cleaned)
            total = latency.finish_pipeline()
            metrics_event = None
            if routing_metrics is not None:
                metrics_event = {
                    "event": "dictation",
                    "trigger_mode": trigger_mode,
                    "asr_confidence": round(selected_candidate.confidence, 3),
                    "candidate_count": len(candidates),
                    "candidate_sources": [candidate.source for candidate in candidates],
                    "selected_candidate": selected_candidate.source,
                    "injection_route": inject_route,
                    "cleanup_used": True,
                    "timings": latency.timings,
                }
            _persist_post_insert(
                cleaned=cleaned,
                event=metrics_event,
            )
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
        nonlocal recording, recorded_frames, utterance_id, recording_source

        with lock:
            pressed_keys.add(key)
            if not _is_hotkey_match(pressed_keys):
                return
            if recording:
                return
            utterance_id += 1
            recording = True
            recording_source = "push_to_talk"
            # Grab the pre-buffer — this captures ~500ms of audio from
            # BEFORE the keypress, so the first word isn't cut off.
            recorded_frames = list(pre_buffer)
            pre_buffer.clear()

        injector.clear_clipboard()
        _prefetch_context()
        play_sound(sound_start)
        overlay.show_status("Recording…")
        print("  🎙️  Recording...")

    def on_release(key: keyboard.Key | keyboard.KeyCode) -> None:
        nonlocal recording, recorded_frames, recording_source

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
            recording_source = None
            pressed_keys.clear()
            frames = list(recorded_frames)
            recorded_frames = []
            uid = utterance_id

        play_sound(sound_stop)
        overlay.show_status("Processing…")
        print("  ⏳ Processing...")
        threading.Thread(target=process_utterance, args=(frames, uid, "push_to_talk"), daemon=True).start()

    # --- Ready ---
    print()
    print("  ✔ Ready!")
    print()
    if input_mode == "live":
        print("  Live mode enabled. Speak naturally and pause to send. Ctrl+C to quit.")
    else:
        print("  Hold Right Option key to dictate. Ctrl+C to quit.")
    print()

    if VERBOSE and routing_metrics is not None:
        print(f"  Routing metrics: {routing_metrics.path}")

    if input_mode == "live":
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
    else:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                pass

    print("\n  Shutting down...")
    shutdown_event.set()
    overlay.hide()
    mic.stop()
    cleanup.close()
    asr.unload()
    capture_thread.join(timeout=2)
    print("  Bye.")


if __name__ == "__main__":
    main()
