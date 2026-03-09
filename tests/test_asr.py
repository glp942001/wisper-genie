"""Tests for ASR module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dictation.asr.base import ASRAdapter
from dictation.asr.whisper_cpp import WhisperCppAdapter


class TestASRProtocol:
    def test_mock_conforms_to_protocol(self, mock_asr):
        assert hasattr(mock_asr, "transcribe")
        assert hasattr(mock_asr, "load")
        assert mock_asr.transcribe(np.zeros(16000, dtype=np.int16)) == "hello world this is a test"


class TestWhisperCppAdapter:
    def test_raises_if_model_not_found(self, tmp_path):
        adapter = WhisperCppAdapter(model_path=tmp_path / "nonexistent.bin")
        with pytest.raises(FileNotFoundError, match="Whisper model not found"):
            adapter.load()

    def test_raises_if_not_loaded(self, short_audio):
        adapter = WhisperCppAdapter(model_path="/fake/path.bin")
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.transcribe(short_audio)

    @patch("dictation.asr.whisper_cpp.WhisperModel")
    def test_transcribe_returns_text(self, mock_model_cls, tmp_path, short_audio):
        model_file = tmp_path / "model.bin"
        model_file.touch()

        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = " Hello world "
        mock_model.transcribe.return_value = [mock_segment]
        mock_model_cls.return_value = mock_model

        adapter = WhisperCppAdapter(model_path=model_file)
        adapter.load()
        result = adapter.transcribe(short_audio)

        assert result == "Hello world"
        mock_model.transcribe.assert_called_once()

    @patch("dictation.asr.whisper_cpp.WhisperModel")
    def test_transcribe_uses_ravel(self, mock_model_cls, tmp_path):
        """Verify audio is passed as float32 normalized by 32767."""
        model_file = tmp_path / "model.bin"
        model_file.touch()

        mock_model = MagicMock()
        mock_model.transcribe.return_value = []
        mock_model_cls.return_value = mock_model

        adapter = WhisperCppAdapter(model_path=model_file)
        adapter.load()

        audio = np.array([32767, -32767], dtype=np.int16)
        adapter.transcribe(audio)

        call_args = mock_model.transcribe.call_args[0][0]
        assert call_args.dtype == np.float32
        assert abs(call_args[0] - 1.0) < 0.001
        assert abs(call_args[1] - (-1.0)) < 0.001

    def test_unload(self, tmp_path):
        adapter = WhisperCppAdapter(model_path=tmp_path / "model.bin")
        adapter._model = MagicMock()
        adapter.unload()
        assert adapter._model is None
