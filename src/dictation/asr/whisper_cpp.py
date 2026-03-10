"""whisper.cpp ASR adapter using pywhispercpp."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import re

import numpy as np
from pywhispercpp.model import Model as WhisperModel
from pywhispercpp.model import Segment

_HALLUCINATION_PATTERN = re.compile(
    r"\[(?:BLANK_AUDIO|BLANK|SILENCE|MUSIC|APPLAUSE|LAUGHTER)\]",
    re.IGNORECASE,
)
_ANNOTATION_PATTERN = re.compile(
    r"\([a-z\s]{2,30}\)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TranscriptionCandidate:
    text: str
    confidence: float
    source: str


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    confidence: float
    candidates: tuple[TranscriptionCandidate, ...]


def _candidate_from_segments(segments: list[Segment], source: str) -> TranscriptionCandidate:
    parts = []
    probabilities = []
    for seg in segments:
        part = seg.text.strip()
        if not part:
            continue
        part = _HALLUCINATION_PATTERN.sub("", part).strip()
        part = _ANNOTATION_PATTERN.sub("", part).strip()
        if not part:
            continue
        parts.append(part)
        probability = float(getattr(seg, "probability", np.nan))
        if not np.isnan(probability):
            probabilities.append(probability)

    text = " ".join(parts).strip()
    if probabilities:
        confidence = float(sum(probabilities) / len(probabilities))
    elif text:
        confidence = 0.65
    else:
        confidence = 0.0
    return TranscriptionCandidate(text=text, confidence=confidence, source=source)


class WhisperCppAdapter:
    """ASR adapter backed by whisper.cpp with Metal acceleration."""

    def __init__(
        self,
        model_path: str | Path,
        language: str = "en",
        n_threads: int = 8,
    ) -> None:
        self._model_path = Path(model_path)
        self._language = language
        self._n_threads = n_threads
        self._model: WhisperModel | None = None

    def load(self) -> None:
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
            single_segment=True,
            no_context=True,
        )

    def transcribe(self, audio: np.ndarray, initial_prompt: str | None = None) -> str:
        return self.transcribe_detailed(audio, initial_prompt=initial_prompt).text

    def transcribe_candidate(
        self,
        audio: np.ndarray,
        *,
        initial_prompt: str | None = None,
        source: str | None = None,
    ) -> TranscriptionCandidate:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        audio_f32 = audio.ravel().astype(np.float32) / 32767.0
        segments = self._model.transcribe(
            audio_f32,
            extract_probability=True,
            **({"initial_prompt": initial_prompt} if initial_prompt else {}),
        )
        candidate_source = source or ("prompted" if initial_prompt else "unprompted")
        return _candidate_from_segments(segments, candidate_source)

    def transcribe_detailed(
        self,
        audio: np.ndarray,
        *,
        initial_prompt: str | None = None,
        include_alternative: bool = False,
    ) -> TranscriptionResult:
        primary = self.transcribe_candidate(
            audio,
            initial_prompt=initial_prompt,
            source="prompted",
        )
        candidates = [primary]

        if include_alternative:
            alternate = self.transcribe_candidate(
                audio,
                initial_prompt=None,
                source="unprompted",
            )
            if alternate.text and alternate.text != primary.text:
                candidates.append(alternate)

        selected = max(candidates, key=lambda candidate: candidate.confidence)
        return TranscriptionResult(
            text=selected.text,
            confidence=selected.confidence,
            candidates=tuple(candidates),
        )

    def unload(self) -> None:
        self._model = None
