"""Tests for VAD module.

Note: VAD is not currently used in the pipeline (push-to-talk only),
but we keep these tests for when VAD integration is added in Phase 2.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dictation.audio.vad import SileroVAD, VADEvent


@pytest.fixture
def mock_vad_model():
    """Create a mock Silero VAD model."""
    model = MagicMock()
    model.reset_states.return_value = None
    return model


@pytest.fixture
def vad_with_mock(mock_vad_model):
    """Create a SileroVAD with a mocked model."""
    with patch("dictation.audio.vad.torch.hub.load") as mock_load:
        mock_load.return_value = (mock_vad_model, None)
        v = SileroVAD(
            sample_rate=16000,
            threshold=0.5,
            silence_duration_ms=90,  # 3 frames at 30ms
            frame_size_ms=30,
        )
    return v, mock_vad_model


class TestSileroVAD:
    def test_speech_start_event(self, vad_with_mock, speech_frame):
        vad, model = vad_with_mock
        model.return_value = MagicMock(item=MagicMock(return_value=0.8))

        events = []
        vad.on_event(lambda event, audio: events.append(event))

        vad.process_frame(speech_frame)
        assert VADEvent.SPEECH_START in events
        assert vad.is_speaking

    def test_no_event_on_silence(self, vad_with_mock, silence_frame):
        vad, model = vad_with_mock
        model.return_value = MagicMock(item=MagicMock(return_value=0.1))

        events = []
        vad.on_event(lambda event, audio: events.append(event))

        vad.process_frame(silence_frame)
        assert len(events) == 0
        assert not vad.is_speaking

    def test_speech_end_after_silence(self, vad_with_mock, speech_frame, silence_frame):
        vad, model = vad_with_mock
        events = []
        vad.on_event(lambda event, audio: events.append(event))

        model.return_value = MagicMock(item=MagicMock(return_value=0.8))
        vad.process_frame(speech_frame)
        assert VADEvent.SPEECH_START in events

        model.return_value = MagicMock(item=MagicMock(return_value=0.1))
        for _ in range(3):
            vad.process_frame(silence_frame)

        assert VADEvent.SPEECH_END in events
        assert not vad.is_speaking

    def test_no_speech_end_before_threshold(self, vad_with_mock, speech_frame, silence_frame):
        vad, model = vad_with_mock
        events = []
        vad.on_event(lambda event, audio: events.append(event))

        model.return_value = MagicMock(item=MagicMock(return_value=0.8))
        vad.process_frame(speech_frame)

        model.return_value = MagicMock(item=MagicMock(return_value=0.1))
        vad.process_frame(silence_frame)
        vad.process_frame(silence_frame)

        assert VADEvent.SPEECH_END not in events
        assert vad.is_speaking

    def test_reset(self, vad_with_mock, speech_frame):
        vad, model = vad_with_mock
        model.return_value = MagicMock(item=MagicMock(return_value=0.8))

        vad.process_frame(speech_frame)
        assert vad.is_speaking

        vad.reset()
        assert not vad.is_speaking
