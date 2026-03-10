"""Transcript buffer — normalizes and stabilizes ASR output."""

from __future__ import annotations

import re
import threading

FILLER_WORDS = {"um", "uh", "uhh", "umm", "hmm", "hm", "erm", "ah", "ahh"}
_FILLER_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(f) for f in sorted(FILLER_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

DICTATION_COMMANDS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bnew paragraph\b", re.IGNORECASE), "\n\n"),
    (re.compile(r"\bnew line\b", re.IGNORECASE), "\n"),
    (re.compile(r"\bnewline\b", re.IGNORECASE), "\n"),
    (re.compile(r"\bexclamation point\b", re.IGNORECASE), "!"),
    (re.compile(r"\bexclamation mark\b", re.IGNORECASE), "!"),
    (re.compile(r"\bquestion mark\b", re.IGNORECASE), "?"),
    (re.compile(r"\bopen quote\b", re.IGNORECASE), '"'),
    (re.compile(r"\bclose quote\b", re.IGNORECASE), '"'),
    (re.compile(r"\bfull stop\b", re.IGNORECASE), "."),
]

DICTATION_COMMANDS_END_ONLY: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bsemicolon\s*$", re.IGNORECASE), ";"),
    (re.compile(r"\bellipsis\s*$", re.IGNORECASE), "..."),
    (re.compile(r"\bperiod\s*$", re.IGNORECASE), "."),
    (re.compile(r"\bcomma\s*$", re.IGNORECASE), ","),
    (re.compile(r"\bcolon\s*$", re.IGNORECASE), ":"),
    (re.compile(r"\bdash\s*$", re.IGNORECASE), " — "),
    (re.compile(r"\bhyphen\s*$", re.IGNORECASE), "-"),
]

_STRONG_BACKTRACK_PATTERN = re.compile(
    r"\b(?:scratch that|no wait|wait no|i meant|i mean|no no)\b",
    re.IGNORECASE,
)
_WORD_PATTERN = re.compile(r"[A-Za-z']+")
_NON_CORRECTION_PREVIOUS_WORDS = {
    "was", "is", "are", "were", "am", "be", "been", "being",
    "sounds", "sound", "seems", "seem", "feels", "feel",
    "looks", "look", "really", "just",
}

_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?\"\.\.\.])")
_SPACE_AROUND_NEWLINE = re.compile(r" *\n *")
MAX_HISTORY = 50


def _looks_like_backtrack(text: str) -> bool:
    if _STRONG_BACKTRACK_PATTERN.search(text):
        return True

    words = [word.lower() for word in _WORD_PATTERN.findall(text)]
    for idx, word in enumerate(words):
        if word != "actually":
            continue
        if idx < 2 or idx >= len(words) - 1:
            continue
        if words[idx - 1] in _NON_CORRECTION_PREVIOUS_WORDS:
            continue
        return True
    return False


class TranscriptBuffer:
    """Manages normalized ASR output and finalized dictation history."""

    def __init__(self, strip_fillers: bool = True) -> None:
        self._strip_fillers = strip_fillers
        self._finalized_utterances: list[str] = []
        self._lock = threading.Lock()

    def add(self, text: str) -> tuple[str, bool]:
        """Normalize raw ASR text and detect whether it looks like a correction."""
        has_backtrack = _looks_like_backtrack(text)
        cleaned = self._normalize(text)
        return cleaned, has_backtrack

    def commit(self, text: str) -> None:
        """Append finalized cleaned text to conversational history."""
        cleaned = self._normalize(text)
        if not cleaned:
            return
        with self._lock:
            self._finalized_utterances.append(cleaned)
            if len(self._finalized_utterances) > MAX_HISTORY:
                self._finalized_utterances = self._finalized_utterances[-MAX_HISTORY:]

    def _normalize(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""

        if self._strip_fillers:
            text = _FILLER_PATTERN.sub("", text)

        for pattern, replacement in DICTATION_COMMANDS:
            text = pattern.sub(replacement, text)
        for pattern, replacement in DICTATION_COMMANDS_END_ONLY:
            text = pattern.sub(replacement, text)

        text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
        text = _SPACE_AROUND_NEWLINE.sub("\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    def get_context(self, n: int = 2) -> str:
        with self._lock:
            recent = self._finalized_utterances[-n:] if self._finalized_utterances else []
        return " ".join(recent)

    def get_history(self) -> list[str]:
        with self._lock:
            return list(self._finalized_utterances)

    def get_last(self, n: int = 1) -> list[str]:
        with self._lock:
            return list(self._finalized_utterances[-n:])

    def clear(self) -> None:
        with self._lock:
            self._finalized_utterances.clear()
