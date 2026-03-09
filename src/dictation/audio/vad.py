"""Voice Activity Detection using Silero VAD."""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Callable


class VADEvent(enum.Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


class SileroVAD:
    """Wraps Silero VAD for speech start/end detection.

    Processes fixed-size audio frames and emits SPEECH_START / SPEECH_END events
    based on configurable thresholds.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        silence_duration_ms: int = 300,
        frame_size_ms: int = 30,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.frame_size_ms = frame_size_ms
        self.frame_size = int(sample_rate * frame_size_ms / 1000)

        # Number of consecutive silent frames to confirm endpoint
        self._silence_frames_needed = int(silence_duration_ms / frame_size_ms)
        self._silent_frame_count = 0
        self._is_speaking = False

        # Load Silero VAD
        self._model, _utils = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            trust_repo=True,
        )

        self._listeners: list[Callable[[VADEvent, np.ndarray | None], None]] = []

    def on_event(self, callback: Callable[[VADEvent, np.ndarray | None], None]) -> None:
        """Register a callback for VAD events."""
        self._listeners.append(callback)

    def _emit(self, event: VADEvent, audio: np.ndarray | None = None) -> None:
        for cb in self._listeners:
            cb(event, audio)

    def process_frame(self, frame: np.ndarray) -> float:
        """Process a single audio frame and return speech probability.

        Args:
            frame: int16 PCM audio, shape (N,) or (N, 1)

        Returns:
            Speech probability [0, 1].
        """
        # Convert int16 to float32 in [-1, 1]
        audio = frame.flatten().astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio)

        with torch.no_grad():
            prob = self._model(tensor, self.sample_rate).item()

        if prob >= self.threshold:
            self._silent_frame_count = 0
            if not self._is_speaking:
                self._is_speaking = True
                self._emit(VADEvent.SPEECH_START)
        else:
            if self._is_speaking:
                self._silent_frame_count += 1
                if self._silent_frame_count >= self._silence_frames_needed:
                    self._is_speaking = False
                    self._silent_frame_count = 0
                    self._emit(VADEvent.SPEECH_END)

        return prob

    def reset(self) -> None:
        """Reset VAD state for a new utterance."""
        self._is_speaking = False
        self._silent_frame_count = 0
        self._model.reset_states()

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking
