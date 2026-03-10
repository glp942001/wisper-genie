"""Local style memory for dictation cleanup prompts."""

from __future__ import annotations

import json
import re
import threading
from collections import Counter
from pathlib import Path

_GREETING_RE = re.compile(r"^(hi|hello|hey|dear)\b", re.IGNORECASE)
_SIGNOFF_RE = re.compile(r"^(thanks|thank you|best|best regards|regards|sincerely|cheers)\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z']+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "is", "it", "this", "that", "we", "i", "you", "me", "my", "our",
    "your", "be", "are", "was", "were", "as", "at", "by", "from", "if",
    "let", "lets", "just", "really",
}


def _default_scope() -> dict:
    return {
        "count": 0,
        "punctuated_count": 0,
        "capitalized_count": 0,
        "short_count": 0,
        "polished_count": 0,
        "greetings": {},
        "signoffs": {},
        "phrases": {},
    }


def _fresh_data() -> dict:
    return {"global": _default_scope()}


def _coerce_scope(raw: object) -> dict:
    if not isinstance(raw, dict):
        return _default_scope()

    scope = _default_scope()
    for key in ("count", "punctuated_count", "capitalized_count", "short_count", "polished_count"):
        value = raw.get(key)
        if isinstance(value, int) and value >= 0:
            scope[key] = value

    for key in ("greetings", "signoffs", "phrases"):
        value = raw.get(key)
        if isinstance(value, dict):
            scope[key] = {
                str(name): count
                for name, count in value.items()
                if isinstance(count, int) and count >= 0
            }
    return scope


def _coerce_data(raw: object) -> dict:
    if not isinstance(raw, dict):
        return _fresh_data()
    return {"global": _coerce_scope(raw.get("global"))}


def _extract_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _extract_phrase_counts(text: str) -> Counter[str]:
    words = [word.lower() for word in _WORD_RE.findall(text)]
    phrases: Counter[str] = Counter()
    for size in (2, 3):
        for idx in range(0, len(words) - size + 1):
            chunk = words[idx: idx + size]
            if any(word in _STOPWORDS for word in chunk):
                continue
            phrases[" ".join(chunk)] += 1
    return phrases


class StyleMemory:
    """Persists local writing-style hints derived from accepted dictation."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (Path.home() / ".wisper-genie" / "style_memory.json")
        self._lock = threading.Lock()
        self._data = _fresh_data()
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            self._data = _fresh_data()
            return
        self._data = _coerce_data(raw)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def observe(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return

        with self._lock:
            self._observe_scope(self._data["global"], cleaned)
            self._save()

    def build_prompt_hint(self) -> str:
        with self._lock:
            global_scope = self._data["global"]

            hint = self._scope_hint(global_scope, label="your accepted dictation")
            if not hint:
                return ""
            return "User writing memory: " + hint

    @staticmethod
    def _observe_scope(scope: dict, text: str) -> None:
        scope["count"] += 1
        if text.endswith((".", "!", "?", ":", ";")):
            scope["punctuated_count"] += 1
        if text[:1].isupper():
            scope["capitalized_count"] += 1

        word_count = len(_WORD_RE.findall(text))
        if word_count <= 10:
            scope["short_count"] += 1
        if word_count >= 14 or text.count(",") >= 1:
            scope["polished_count"] += 1

        lines = _extract_lines(text)
        if lines:
            greeting = StyleMemory._extract_greeting(lines[0])
            if greeting:
                scope["greetings"][greeting] = scope["greetings"].get(greeting, 0) + 1

            signoff = StyleMemory._extract_signoff(lines[-1])
            if signoff:
                scope["signoffs"][signoff] = scope["signoffs"].get(signoff, 0) + 1

        for phrase, count in _extract_phrase_counts(text).items():
            scope["phrases"][phrase] = scope["phrases"].get(phrase, 0) + count

    @staticmethod
    def _extract_greeting(line: str) -> str | None:
        match = _GREETING_RE.match(line)
        if not match:
            return None
        return match.group(1).title()

    @staticmethod
    def _extract_signoff(line: str) -> str | None:
        match = _SIGNOFF_RE.match(line.rstrip(","))
        if not match:
            return None
        return match.group(1).title()

    @staticmethod
    def _top_counter_items(data: dict[str, int], limit: int = 2) -> list[str]:
        counter = Counter(data)
        return [item for item, _count in counter.most_common(limit)]

    def _scope_hint(self, scope: dict, *, label: str) -> str:
        if scope.get("count", 0) < 2:
            return ""

        traits: list[str] = []
        punctuated_ratio = scope["punctuated_count"] / max(scope["count"], 1)
        if punctuated_ratio >= 0.7:
            traits.append("prefers strong punctuation")
        elif punctuated_ratio <= 0.3:
            traits.append("often leaves punctuation light")

        capitalized_ratio = scope["capitalized_count"] / max(scope["count"], 1)
        if capitalized_ratio >= 0.8:
            traits.append("usually starts with capitalization")

        if scope["short_count"] > scope["polished_count"]:
            traits.append("leans short and direct")
        elif scope["polished_count"] > scope["short_count"]:
            traits.append("leans polished and sentence-like")

        greetings = self._top_counter_items(scope.get("greetings", {}), limit=1)
        if greetings:
            traits.append(f"often greets with {greetings[0]}")

        signoffs = self._top_counter_items(scope.get("signoffs", {}), limit=1)
        if signoffs:
            traits.append(f"often signs off with {signoffs[0]}")

        phrases = self._top_counter_items(scope.get("phrases", {}), limit=2)
        if phrases:
            traits.append("recurring phrases: " + ", ".join(phrases))

        if not traits:
            return ""
        return f"For {label}, " + "; ".join(traits) + "."
