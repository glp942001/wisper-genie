"""ASR adapter protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ASRAdapter(Protocol):
    """Interface for ASR backends.

    Implementations must accept 16kHz int16 mono audio and return text.
    """

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe an audio buffer to text.

        Args:
            audio: int16 PCM audio at 16kHz, shape (N,) or (N, 1).

        Returns:
            Transcribed text string.
        """
        ...

    def load(self) -> None:
        """Load the model into memory. Called once at startup."""
        ...
