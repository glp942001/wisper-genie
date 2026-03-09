"""Tests for personal dictionary."""

from pathlib import Path
from dictation.context.dictionary import load_dictionary


class TestDictionary:
    def test_returns_expected_keys(self, tmp_path):
        d = load_dictionary(tmp_path / "nonexistent.toml")
        assert "terms" in d
        assert "prompt_hint" in d
        assert "whisper_hint" in d

    def test_missing_file_returns_empty(self, tmp_path):
        d = load_dictionary(tmp_path / "nope.toml")
        assert d["terms"] == []
        assert d["prompt_hint"] == ""
        assert d["whisper_hint"] == ""

    def test_loads_names_and_terms(self, tmp_path):
        f = tmp_path / "dict.toml"
        f.write_text('names = ["Alice", "Bob"]\nterms = ["kubectl", "FastAPI"]')
        d = load_dictionary(f)
        assert "Alice" in d["terms"]
        assert "Bob" in d["terms"]
        assert "kubectl" in d["terms"]
        assert "FastAPI" in d["terms"]
        assert len(d["terms"]) == 4

    def test_prompt_hint_contains_terms(self, tmp_path):
        f = tmp_path / "dict.toml"
        f.write_text('names = ["NASA"]\nterms = ["kubectl"]')
        d = load_dictionary(f)
        assert "NASA" in d["prompt_hint"]
        assert "kubectl" in d["prompt_hint"]

    def test_whisper_hint_is_comma_separated(self, tmp_path):
        f = tmp_path / "dict.toml"
        f.write_text('names = ["Alice"]\nterms = ["Bob"]')
        d = load_dictionary(f)
        assert "Alice" in d["whisper_hint"]
        assert "Bob" in d["whisper_hint"]

    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "dict.toml"
        f.write_text("")
        d = load_dictionary(f)
        assert d["terms"] == []
