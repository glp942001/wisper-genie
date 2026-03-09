"""Transcript buffer — normalizes and stabilizes ASR output."""

from __future__ import annotations

import re
import threading

# Only true verbal fillers — not words that could be intentional
FILLER_WORDS = {"um", "uh", "uhh", "umm", "hmm", "hm", "erm", "ah", "ahh"}

_FILLER_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(f) for f in sorted(FILLER_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Dictation commands: spoken word/phrase → replacement
# Multi-word commands are safe to match anywhere (low false-positive risk)
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

# Single-word commands that are also common English words.
# Only match at end of text to avoid false positives
# (e.g., "I have a period every month" should NOT become "I have a. every month").
# The LLM cleanup step handles mid-sentence punctuation naturally.
DICTATION_COMMANDS_END_ONLY: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bsemicolon\s*$", re.IGNORECASE), ";"),
    (re.compile(r"\bellipsis\s*$", re.IGNORECASE), "..."),
    (re.compile(r"\bperiod\s*$", re.IGNORECASE), "."),
    (re.compile(r"\bcomma\s*$", re.IGNORECASE), ","),
    (re.compile(r"\bcolon\s*$", re.IGNORECASE), ":"),
    (re.compile(r"\bdash\s*$", re.IGNORECASE), " — "),
    (re.compile(r"\bhyphen\s*$", re.IGNORECASE), "-"),
]

# Backtrack phrases — speaker is correcting themselves.
# When detected, the LLM should interpret the correction, not transcribe literally.
_BACKTRACK_PATTERN = re.compile(
    r"\b(?:actually|scratch that|no wait|wait no|I meant|I mean|no no)\b",
    re.IGNORECASE,
)

# Clean up spacing around injected punctuation
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?\"\.\.\.])")
_SPACE_AROUND_NEWLINE = re.compile(r" *\n *")

# Maximum utterances to keep in history (prevents unbounded memory growth)
MAX_HISTORY = 50


class TranscriptBuffer:
    """Manages and normalizes ASR transcript output.

    Thread-safe: all access to _utterances is protected by an internal lock.
    Handles dictation commands, strips verbal fillers, and tracks utterance history.
    """

    def __init__(self, strip_fillers: bool = True) -> None:
        self._strip_fillers = strip_fillers
        self._utterances: list[str] = []
        self._lock = threading.Lock()

    def add(self, text: str) -> tuple[str, bool]:
        """Add a transcript and return (normalized_text, has_backtrack).

        The backtrack flag indicates the speaker corrected themselves
        (e.g., "actually", "scratch that"). The LLM should interpret
        the correction rather than transcribing literally.
        """
        has_backtrack = bool(_BACKTRACK_PATTERN.search(text))
        cleaned = self._normalize(text)
        if cleaned:
            with self._lock:
                self._utterances.append(cleaned)
                # Evict oldest entries if over limit
                if len(self._utterances) > MAX_HISTORY:
                    self._utterances = self._utterances[-MAX_HISTORY:]
        return cleaned, has_backtrack

    def _normalize(self, text: str) -> str:
        """Normalize a transcript string."""
        text = text.strip()
        if not text:
            return ""

        # 1. Remove verbal fillers only (um, uh, hmm — not "like", "so", etc.)
        if self._strip_fillers:
            text = _FILLER_PATTERN.sub("", text)

        # 2. Process dictation commands (spoken punctuation → symbols)
        for pattern, replacement in DICTATION_COMMANDS:
            text = pattern.sub(replacement, text)
        # Single-word commands: only at end of text to avoid false positives
        for pattern, replacement in DICTATION_COMMANDS_END_ONLY:
            text = pattern.sub(replacement, text)

        # 3. Clean spacing around punctuation and newlines
        text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
        text = _SPACE_AROUND_NEWLINE.sub("\n", text)

        # 4. Collapse multiple spaces
        text = re.sub(r" {2,}", " ", text)

        text = text.strip()
        return text

    def get_context(self, n: int = 2) -> str:
        """Return the last n utterances joined, for LLM context."""
        with self._lock:
            recent = self._utterances[-n:] if self._utterances else []
        return " ".join(recent)

    def get_history(self) -> list[str]:
        """Return all previous utterances."""
        with self._lock:
            return list(self._utterances)

    def get_last(self, n: int = 1) -> list[str]:
        """Return the last n utterances."""
        with self._lock:
            return list(self._utterances[-n:])

    def clear(self) -> None:
        """Clear the utterance history."""
        with self._lock:
            self._utterances.clear()
