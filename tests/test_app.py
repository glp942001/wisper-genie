"""Tests for app-level audio helpers."""

import sys
from unittest.mock import patch

import numpy as np

import dictation.app as app_module
from dictation.app import (
    _build_whisper_prompt_variants,
    _normalize_audio,
    _select_best_candidate,
    _trim_audio_energy,
)
from dictation.asr.whisper_cpp import TranscriptionCandidate


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

    def test_select_best_candidate_uses_context_overlap(self):
        candidates = [
            TranscriptionCandidate(text="launch timing tomorrow", confidence=0.72, source="unprompted"),
            TranscriptionCandidate(text="launch timing tuesday", confidence=0.7, source="prompted"),
        ]

        selected = _select_best_candidate(
            candidates,
            field_text="following up on launch timing for Tuesday",
            selected_text="",
            recent_utterances=["mention launch timing"],
            dictionary_terms=["Tuesday"],
        )

        assert selected.text == "launch timing tuesday"

    def test_build_whisper_prompt_variants_generates_multiple_unique_candidates(self):
        variants = _build_whisper_prompt_variants(
            recent_utterances=["follow up with Ollama tomorrow"],
            dictionary_hint="Ollama, Whisper",
            field_text="mention the launch timing",
            selected_text="",
            max_hypotheses=4,
        )

        assert len(variants) >= 3
        assert variants[0][0] == "prompted_full"
        assert variants[-1][0] == "unprompted"

    def test_main_dispatches_cli_subcommands(self):
        with patch.object(sys, "argv", ["wisper-genie", "metrics"]):
            with patch("dictation.cli.main") as mock_cli_main:
                app_module.main()

        mock_cli_main.assert_called_once()
