"""Tests for app-level audio helpers."""

import numpy as np

from dictation.app import _normalize_audio, _trim_audio_energy


class TestAppHelpers:
    def test_trim_audio_energy_removes_silence(self):
        silence = np.zeros(480 * 3, dtype=np.int16)
        speech = np.full(480 * 4, 4000, dtype=np.int16)
        audio = np.concatenate([silence, speech, silence])

        trimmed = _trim_audio_energy(audio, frame_size=480)

        assert len(trimmed) < len(audio)
        assert np.max(np.abs(trimmed)) == 4000

    def test_trim_audio_energy_returns_original_when_no_speech(self):
        audio = np.zeros(480 * 4, dtype=np.int16)
        trimmed = _trim_audio_energy(audio, frame_size=480)
        assert np.array_equal(trimmed, audio)

    def test_normalize_audio_caps_gain_on_low_energy_noise(self):
        quiet = np.full(1600, 20, dtype=np.int16)
        normalized = _normalize_audio(quiet, min_rms=200)
        assert np.array_equal(normalized, quiet)

    def test_normalize_audio_scales_speech_but_caps_gain(self):
        audio = np.full(1600, 1000, dtype=np.int16)
        normalized = _normalize_audio(audio, target_peak=28000, max_gain=2.0, min_rms=10)
        assert int(np.max(np.abs(normalized))) == 2000
