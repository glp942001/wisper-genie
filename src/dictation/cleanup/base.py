"""Cleanup LLM protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CleanupAdapter(Protocol):
    """Interface for transcript cleanup backends."""

    def cleanup(self, raw_transcript: str, context: dict | None = None) -> str:
        """Clean up a raw ASR transcript.

        Args:
            raw_transcript: Raw text from ASR + buffer normalization.
            context: Optional dict with app_name, field_text, etc.

        Returns:
            Cleaned, formatted text ready for injection.
        """
        ...
