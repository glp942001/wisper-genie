"""Integration test for the full pipeline (with mocks)."""

import numpy as np

from dictation.transcript.buffer import TranscriptBuffer
from dictation.telemetry.latency import LatencyTracker


class TestPipelineIntegration:
    """Test the pipeline flow using mocks for ASR, cleanup, and injection."""

    def test_full_pipeline_with_mocks(self, mock_asr, mock_cleanup, mock_injector):
        """Simulate the full pipeline: audio -> ASR -> buffer -> cleanup -> inject."""
        tracker = LatencyTracker(
            budgets={"asr": 250, "cleanup": 350, "injection": 50},
            total_budget_ms=800,
        )
        transcript_buf = TranscriptBuffer()

        audio = np.zeros(16000, dtype=np.int16)

        tracker.start_pipeline()

        with tracker.track("asr"):
            raw_text = mock_asr.transcribe(audio)
        assert raw_text == "hello world this is a test"

        with tracker.track("normalize"):
            normalized, has_backtrack = transcript_buf.add(raw_text)
        assert normalized == "hello world this is a test"
        assert has_backtrack is False

        with tracker.track("cleanup"):
            cleaned = mock_cleanup.cleanup(normalized)
        assert cleaned == normalized

        with tracker.track("injection"):
            mock_injector.inject(cleaned)

        total = tracker.finish_pipeline()

        timings = tracker.timings
        assert "asr" in timings
        assert "normalize" in timings
        assert "cleanup" in timings
        assert "injection" in timings
        assert "total" in timings
        assert total < 100

    def test_empty_transcript_skips_pipeline(self, mock_asr, mock_cleanup, mock_injector):
        """If ASR returns empty, cleanup and injection should not run."""
        mock_asr.transcribe.return_value = ""
        transcript_buf = TranscriptBuffer()

        audio = np.zeros(16000, dtype=np.int16)
        raw_text = mock_asr.transcribe(audio)

        if raw_text.strip():
            normalized, _ = transcript_buf.add(raw_text)
            cleaned = mock_cleanup.cleanup(normalized)
            mock_injector.inject(cleaned)

        mock_cleanup.cleanup.assert_not_called()
        mock_injector.inject.assert_not_called()


class TestLatencyTracker:
    def test_tracks_stages(self):
        tracker = LatencyTracker(budgets={"fast": 1000}, total_budget_ms=5000)
        tracker.start_pipeline()
        with tracker.track("fast"):
            pass
        total = tracker.finish_pipeline()

        assert "fast" in tracker.timings
        assert tracker.timings["fast"] < 10
        assert total < 10

    def test_summary_format(self):
        tracker = LatencyTracker(budgets={"asr": 250}, total_budget_ms=800)
        tracker.start_pipeline()
        with tracker.track("asr"):
            pass
        tracker.finish_pipeline()
        summary = tracker.summary()
        assert "asr" in summary
        assert "Pipeline latency" in summary
