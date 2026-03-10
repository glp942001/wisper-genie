"""Tests for microphone capture buffering."""

from unittest.mock import MagicMock

import numpy as np

from dictation.audio.capture import MicCapture


class TestMicCapture:
    def test_ring_buffer_drops_oldest_frames(self):
        capture = MicCapture()
        status = MagicMock()
        status.__bool__.return_value = False

        for idx in range(capture._max_buffered_frames + 5):
            frame = np.full((capture.frame_size, 1), idx, dtype=np.int16)
            capture._audio_callback(frame, capture.frame_size, None, status)

        frames = capture.read_all()
        assert len(frames) == capture._max_buffered_frames
        assert int(frames[0][0][0]) == 5
        assert capture.stats["frames_dropped"] == 5
