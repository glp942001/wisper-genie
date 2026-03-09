"""whisper.cpp ASR adapter using pywhispercpp."""

from __future__ import annotations

from pathlib import Path

import re

import numpy as np
from pywhispercpp.model import Model as WhisperModel

# Whisper hallucination tokens emitted on silence or near-silence.
# These should be stripped from output, not passed to the pipeline.
_HALLUCINATION_PATTERN = re.compile(
    r"\[(?:BLANK_AUDIO|BLANK|SILENCE|MUSIC|APPLAUSE|LAUGHTER)\]",
    re.IGNORECASE,
)


class WhisperCppAdapter:
    """ASR adapter backed by whisper.cpp with Metal acceleration.

    Conforms to the ASRAdapter protocol.
    """

    def __init__(
        self,
        model_path: str | Path,
        language: str = "en",
        n_threads: int = 4,
    ) -> None:
        self._model_path = Path(model_path)
        self._language = language
        self._n_threads = n_threads
        self._model: WhisperModel | None = None

    def load(self) -> None:
        """Load the whisper.cpp model."""
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Whisper model not found: {self._model_path}\n"
                "Run: bash scripts/setup_models.sh"
            )
        self._model = WhisperModel(
            str(self._model_path),
            n_threads=self._n_threads,
            language=self._language,
            print_progress=False,
            print_realtime=False,
            print_timestamps=False,
            print_special=False,
            single_segment=False,
            no_context=False,
        )

    def transcribe(self, audio: np.ndarray, initial_prompt: str | None = None) -> str:
        """Transcribe int16 PCM audio to text.

        Args:
            audio: int16 PCM at 16kHz, shape (N,) or (N, 1).
            initial_prompt: Optional text to bias Whisper toward expected
                vocabulary (e.g., personal dictionary terms, recent context).

        Returns:
            Transcribed text, stripped of leading/trailing whitespace.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # whisper.cpp expects float32 in [-1, 1]
        # Use 32767.0 for correct int16 range normalization
        audio_f32 = audio.ravel().astype(np.float32) / 32767.0

        # pywhispercpp accepts initial_prompt as a keyword arg
        kwargs = {}
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt

        segments = self._model.transcribe(audio_f32, **kwargs)

        # Collect text from all segments, filtering hallucination tokens
        parts = []
        for seg in segments:
            part = seg.text.strip()
            if not part:
                continue
            part = _HALLUCINATION_PATTERN.sub("", part).strip()
            if part:
                parts.append(part)
        return " ".join(parts).strip()

    def unload(self) -> None:
        """Release the model from memory."""
        self._model = None
