"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from dictation.asr.whisper_cpp import TranscriptionCandidate


SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 480 samples


@pytest.fixture
def sample_rate() -> int:
    return SAMPLE_RATE


@pytest.fixture
def silence_frame() -> np.ndarray:
    """A 30ms frame of silence (int16)."""
    return np.zeros(FRAME_SIZE, dtype=np.int16)


@pytest.fixture
def speech_frame() -> np.ndarray:
    """A 30ms frame of synthetic 'speech' (440Hz sine wave, int16)."""
    t = np.linspace(0, FRAME_DURATION_MS / 1000, FRAME_SIZE, endpoint=False)
    wave = np.sin(2 * np.pi * 440 * t)
    return (wave * 16000).astype(np.int16)


@pytest.fixture
def short_audio() -> np.ndarray:
    """1 second of synthetic audio (440Hz sine, int16)."""
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    wave = np.sin(2 * np.pi * 440 * t)
    return (wave * 16000).astype(np.int16)


@pytest.fixture
def sample_audio_path() -> Path:
    """Path to the test audio file (may not exist)."""
    return Path(__file__).parent.parent / "samples" / "test_audio.wav"


@pytest.fixture
def mock_asr():
    """A mock ASR adapter that returns a fixed transcript."""
    mock = MagicMock()
    mock.transcribe.return_value = "hello world this is a test"
    mock.transcribe_candidate.return_value = TranscriptionCandidate(
        text="hello world this is a test",
        confidence=0.9,
        source="primary",
    )
    mock.load.return_value = None
    return mock


@pytest.fixture
def mock_cleanup():
    """A mock cleanup adapter (passthrough)."""
    mock = MagicMock()
    mock.cleanup.side_effect = lambda text, context=None: text
    return mock


@pytest.fixture
def mock_injector():
    """A mock text injector that records injected text."""
    mock = MagicMock()
    mock.injected: list[str] = []

    def record_inject(text: str, **kwargs) -> str:
        mock.injected.append(text)
        return "clipboard"

    mock.inject.side_effect = record_inject
    return mock
