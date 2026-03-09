#!/usr/bin/env python3
"""Measure end-to-end latency per pipeline stage with real models.

Usage:
    python bench/measure_latency.py [--audio samples/test_audio.wav]

If no audio file is provided, generates 3s of synthetic speech-like audio.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dictation.app import load_config
from dictation.telemetry.latency import LatencyTracker


def generate_test_audio(duration_s: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate synthetic audio for benchmarking."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    # Mix of frequencies to simulate speech-like spectral content
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 400 * t)
        + 0.1 * np.sin(2 * np.pi * 800 * t)
    )
    return (audio * 16000).astype(np.int16)


def load_audio(path: Path, sample_rate: int = 16000) -> np.ndarray:
    """Load audio from a WAV file."""
    from scipy.io import wavfile

    sr, data = wavfile.read(path)
    if sr != sample_rate:
        # Simple resampling
        ratio = sample_rate / sr
        indices = np.arange(0, len(data), 1 / ratio).astype(int)
        indices = indices[indices < len(data)]
        data = data[indices]
    if data.ndim > 1:
        data = data[:, 0]  # mono
    return data.astype(np.int16)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure pipeline latency")
    parser.add_argument("--audio", type=Path, help="Path to WAV file")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    args = parser.parse_args()

    cfg = load_config()
    project_root = Path(__file__).resolve().parents[1]

    # Load audio
    if args.audio and args.audio.exists():
        print(f"Loading audio from {args.audio}")
        audio = load_audio(args.audio)
    else:
        print("Generating 3s synthetic audio")
        audio = generate_test_audio()

    duration_s = len(audio) / cfg["audio"]["sample_rate"]
    print(f"Audio duration: {duration_s:.1f}s ({len(audio)} samples)")
    print()

    # Load models
    from dictation.asr.whisper_cpp import WhisperCppAdapter
    from dictation.cleanup.ollama import OllamaCleanup
    from dictation.transcript.buffer import TranscriptBuffer

    model_path = project_root / cfg["asr"]["model_path"]
    print(f"Loading ASR model from {model_path}...")
    asr = WhisperCppAdapter(
        model_path=model_path,
        language=cfg["asr"]["language"],
        beam_size=cfg["asr"]["beam_size"],
    )
    asr.load()
    print("ASR loaded.")

    cleanup = OllamaCleanup(
        model=cfg["cleanup"]["model"],
        base_url=cfg["cleanup"]["base_url"],
        timeout_ms=cfg["cleanup"]["timeout_ms"],
    )

    tracker = LatencyTracker(
        budgets={
            "asr": cfg["latency"]["asr_budget_ms"],
            "normalize": 20,
            "cleanup": cfg["latency"]["cleanup_budget_ms"],
        },
        total_budget_ms=cfg["latency"]["total_budget_ms"],
    )

    # Warmup
    print("Warmup run...")
    buf = TranscriptBuffer()
    asr.transcribe(audio)
    cleanup.cleanup("warmup test")
    print()

    # Benchmark
    for i in range(args.runs):
        print(f"--- Run {i + 1}/{args.runs} ---")
        buf = TranscriptBuffer()
        tracker.start_pipeline()

        with tracker.track("asr"):
            raw = asr.transcribe(audio)
        print(f"  ASR: {raw[:80]}...")

        with tracker.track("normalize"):
            norm = buf.add(raw)

        with tracker.track("cleanup"):
            cleaned = cleanup.cleanup(norm)
        print(f"  Cleaned: {cleaned[:80]}...")

        total = tracker.finish_pipeline()
        print(tracker.summary())
        print()

    cleanup.close()
    print("Done.")


if __name__ == "__main__":
    main()
