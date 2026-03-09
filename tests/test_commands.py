"""Tests for voice command handler."""

from dictation.commands.handler import detect_command, CommandResult


class TestDetectCommand:
    def test_delete_that(self):
        cmd = detect_command("delete that")
        assert cmd is not None
        assert cmd.action == "delete_last"

    def test_scratch_that(self):
        cmd = detect_command("scratch that")
        assert cmd is not None
        assert cmd.action == "delete_last"

    def test_undo(self):
        cmd = detect_command("undo")
        assert cmd is not None
        assert cmd.action == "undo"

    def test_undo_that(self):
        cmd = detect_command("undo that")
        assert cmd is not None
        assert cmd.action == "undo"

    def test_select_all(self):
        cmd = detect_command("select all")
        assert cmd is not None
        assert cmd.action == "select_all"

    def test_replace_with(self):
        cmd = detect_command("replace hello with goodbye")
        assert cmd is not None
        assert cmd.action == "replace"
        assert cmd.args["find"] == "hello"
        assert cmd.args["replacement"] == "goodbye"

    def test_replace_multi_word(self):
        cmd = detect_command("replace good morning with good evening")
        assert cmd is not None
        assert cmd.args["find"] == "good morning"
        assert cmd.args["replacement"] == "good evening"

    def test_normal_text_returns_none(self):
        assert detect_command("hello world") is None
        assert detect_command("I want to delete that file") is None
        assert detect_command("let's go to the store") is None

    def test_empty_returns_none(self):
        assert detect_command("") is None
        assert detect_command("   ") is None

    def test_case_insensitive(self):
        assert detect_command("DELETE THAT") is not None
        assert detect_command("Undo That") is not None
        assert detect_command("REPLACE foo WITH bar") is not None
