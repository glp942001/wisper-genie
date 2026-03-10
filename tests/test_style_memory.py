"""Tests for local writing-style memory."""

from __future__ import annotations

import json

from dictation.context.style_memory import StyleMemory


def test_observe_persists_and_builds_global_hint(tmp_path) -> None:
    path = tmp_path / "style_memory.json"
    memory = StyleMemory(path)

    memory.observe("Hi team\nPlease review this today.\nThanks.")
    memory.observe("Hi team\nCan you send the update?\nThanks.")

    reloaded = StyleMemory(path)
    hint = reloaded.build_prompt_hint()
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert "User writing memory:" in hint
    assert "your accepted dictation" in hint
    assert "often greets with Hi" in hint
    assert "often signs off with Thanks" in hint
    assert "leans short and direct" in hint
    assert payload["global"]["count"] == 2


def test_build_prompt_hint_requires_multiple_examples(tmp_path) -> None:
    memory = StyleMemory(tmp_path / "style_memory.json")

    memory.observe("Hello there.\nThanks.")

    assert memory.build_prompt_hint() == ""


def test_invalid_saved_shape_is_coerced_on_load(tmp_path) -> None:
    path = tmp_path / "style_memory.json"
    path.write_text('{"global": [], "apps": {"Slack": "bad"}}', encoding="utf-8")

    memory = StyleMemory(path)

    assert memory.build_prompt_hint() == ""

    memory.observe("Hello team.\nThanks.")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["global"]["count"] == 1
