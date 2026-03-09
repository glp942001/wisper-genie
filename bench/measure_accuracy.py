#!/usr/bin/env python3
"""Measure ASR word error rate against reference transcripts.

Usage:
    python bench/measure_accuracy.py --audio samples/test_audio.wav --reference "expected text here"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate using dynamic programming."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    r = len(ref_words)
    h = len(hyp_words)

    # DP table
    d = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1):
        d[i][0] = i
    for j in range(h + 1):
        d[0][j] = j

    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # deletion
                    d[i][j - 1] + 1,      # insertion
                    d[i - 1][j - 1] + 1,  # substitution
                )

    return d[r][h] / max(r, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure ASR accuracy (WER)")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--reference", type=str, required=True)
    args = parser.parse_args()

    from dictation.app import load_config
    from dictation.asr.whisper_cpp import WhisperCppAdapter

    cfg = load_config()
    project_root = Path(__file__).resolve().parents[1]

    # Load audio
    from scipy.io import wavfile

    sr, audio = wavfile.read(args.audio)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.int16)

    # Load model
    model_path = project_root / cfg["asr"]["model_path"]
    asr = WhisperCppAdapter(model_path=model_path, language=cfg["asr"]["language"])
    asr.load()

    # Transcribe
    hypothesis = asr.transcribe(audio)

    wer = word_error_rate(args.reference, hypothesis)
    print(f"Reference:  {args.reference}")
    print(f"Hypothesis: {hypothesis}")
    print(f"WER:        {wer:.1%}")


if __name__ == "__main__":
    main()
