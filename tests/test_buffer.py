"""Tests for transcript buffer."""

from dictation.transcript.buffer import (
    TranscriptBuffer,
    MAX_HISTORY,
    DICTATION_COMMANDS,
    DICTATION_COMMANDS_END_ONLY,
)


class TestTranscriptBuffer:
    def test_basic_passthrough(self):
        buf = TranscriptBuffer(strip_fillers=False)
        text, bt = buf.add("hello world")
        assert text == "hello world"
        assert bt is False

    def test_strips_verbal_fillers(self):
        buf = TranscriptBuffer(strip_fillers=True)
        text, _ = buf.add("um hello uh world")
        assert "um" not in text.lower().split()
        assert "uh" not in text.lower().split()
        assert "hello" in text
        assert "world" in text

    def test_keeps_intentional_words(self):
        """Words like 'like', 'so', 'well' should NOT be stripped."""
        buf = TranscriptBuffer(strip_fillers=True)
        assert "like" in buf.add("I like this")[0]
        assert "so" in buf.add("so we went home")[0]
        assert "well" in buf.add("the well was deep")[0]

    def test_dictation_command_period(self):
        buf = TranscriptBuffer()
        text, _ = buf.add("hello world period")
        assert text == "hello world."

    def test_dictation_command_comma_end_of_text(self):
        """Single-word commands like 'comma' only trigger at end of text."""
        buf = TranscriptBuffer()
        text, _ = buf.add("first second third comma")
        assert text == "first second third,"

    def test_dictation_command_comma_mid_text_preserved(self):
        """'comma' mid-text should NOT trigger — avoids false positives."""
        buf = TranscriptBuffer()
        text, _ = buf.add("first comma second comma third")
        assert text == "first comma second comma third"

    def test_dictation_command_question_mark(self):
        buf = TranscriptBuffer()
        text, _ = buf.add("how are you question mark")
        assert text == "how are you?"

    def test_dictation_command_new_line(self):
        buf = TranscriptBuffer()
        text, _ = buf.add("first line new line second line")
        assert text == "first line\nsecond line"

    def test_dictation_command_new_paragraph(self):
        buf = TranscriptBuffer()
        text, _ = buf.add("first paragraph new paragraph second paragraph")
        assert text == "first paragraph\n\nsecond paragraph"

    def test_collapses_whitespace(self):
        buf = TranscriptBuffer()
        text, _ = buf.add("hello    world")
        assert text == "hello world"

    def test_strips_leading_trailing(self):
        buf = TranscriptBuffer()
        text, _ = buf.add("  hello world  ")
        assert text == "hello world"

    def test_empty_input(self):
        buf = TranscriptBuffer()
        assert buf.add("")[0] == ""
        assert buf.add("   ")[0] == ""

    def test_history_tracking(self):
        buf = TranscriptBuffer()
        buf.add("first")
        buf.add("second")
        buf.add("third")
        assert buf.get_history() == ["first", "second", "third"]
        assert buf.get_last(2) == ["second", "third"]

    def test_context_returns_recent_utterances(self):
        buf = TranscriptBuffer()
        buf.add("I went to the store")
        buf.add("Then I came home")
        context = buf.get_context(n=2)
        assert "I went to the store" in context
        assert "Then I came home" in context

    def test_context_empty_when_no_history(self):
        buf = TranscriptBuffer()
        assert buf.get_context(n=2) == ""

    def test_clear(self):
        buf = TranscriptBuffer()
        buf.add("hello")
        buf.clear()
        assert buf.get_history() == []

    def test_filler_only_input(self):
        buf = TranscriptBuffer(strip_fillers=True)
        text, _ = buf.add("um uh hmm")
        assert text == ""

    def test_no_false_positive_on_common_words(self):
        """Words like 'period', 'dash', 'colon' mid-text should NOT convert."""
        buf = TranscriptBuffer()
        assert buf.add("I have a period every month")[0] == "I have a period every month"
        assert buf.add("a dash of salt")[0] == "a dash of salt"
        assert buf.add("the colon is important")[0] == "the colon is important"

    def test_end_only_commands_at_end(self):
        """Single-word commands at end of text should convert."""
        buf = TranscriptBuffer()
        assert buf.add("hello world period")[0] == "hello world."
        assert buf.add("one two three ellipsis")[0] == "one two three..."

    def test_history_bounded(self):
        """Utterance history should not grow beyond MAX_HISTORY."""
        buf = TranscriptBuffer()
        for i in range(MAX_HISTORY + 20):
            buf.add(f"utterance {i}")
        assert len(buf.get_history()) == MAX_HISTORY

    def test_thread_safe_access(self):
        """Basic smoke test that concurrent access doesn't crash."""
        import threading
        buf = TranscriptBuffer()
        errors = []

        def writer():
            try:
                for i in range(50):
                    buf.add(f"text {i}")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    buf.get_context(n=2)
                    buf.get_history()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    # --- Backtrack detection ---
    def test_backtrack_detected(self):
        buf = TranscriptBuffer()
        _, bt = buf.add("lets meet at two actually three")
        assert bt is True

    def test_backtrack_scratch_that(self):
        buf = TranscriptBuffer()
        _, bt = buf.add("scratch that I want pizza")
        assert bt is True

    def test_no_backtrack_on_normal_text(self):
        buf = TranscriptBuffer()
        _, bt = buf.add("hello world how are you")
        assert bt is False

    def test_actually_intentional_still_detects(self):
        """'actually' always sets backtrack flag — LLM decides if it's a correction."""
        buf = TranscriptBuffer()
        _, bt = buf.add("it was actually good")
        assert bt is True
