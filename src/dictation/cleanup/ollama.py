"""Ollama API adapter for transcript cleanup."""

from __future__ import annotations

import httpx

from dictation.cleanup.prompts import (
    build_cleanup_message,
    build_cleanup_system,
    build_few_shot_messages,
)


class OllamaCleanup:
    """Cleans transcripts using a local Ollama model via the chat API.

    Uses few-shot examples for consistent behavior with small models.
    Fail-open: returns raw transcript if the LLM is too slow or unavailable.
    """

    def __init__(
        self,
        model: str = "ministral-3:3b",
        base_url: str = "http://localhost:11434",
        timeout_ms: int = 10000,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_ms / 1000.0
        self._client = httpx.Client(
            timeout=httpx.Timeout(
                connect=3.0,
                read=self._timeout,
                write=5.0,
                pool=5.0,
            ),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )

        # Pre-build the static parts of the message array
        self._few_shot = build_few_shot_messages()

    def cleanup(self, raw_transcript: str, context: dict | None = None) -> str:
        """Send transcript to Ollama for cleanup.

        Args:
            raw_transcript: The normalized ASR text.
            context: Optional dict with app_name, field_text,
                     dictionary_hint, has_backtrack, recent_utterances.

        Returns the cleaned text, or the raw transcript on failure.
        """
        if not raw_transcript.strip():
            return raw_transcript

        system_msg = {"role": "system", "content": build_cleanup_system(context)}
        user_content = build_cleanup_message(raw_transcript, context)

        # Build full message array: system + few-shot + current request
        messages = [system_msg] + self._few_shot + [
            {"role": "user", "content": user_content}
        ]

        try:
            resp = self._client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": messages,
                    "stream": False,
                    "keep_alive": "10m",
                    "options": {
                        "num_predict": 256,
                        "temperature": 0.0,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            cleaned = data.get("message", {}).get("content", "").strip()
            if not cleaned:
                return raw_transcript
            # Guard: if the LLM responded with commentary instead of
            # formatting (e.g., "No meaningful text..."), fall back to raw.
            # A valid cleanup should not be drastically longer than input.
            if len(cleaned) > len(raw_transcript) * 3 + 20:
                print(f"[OllamaCleanup] Response too long vs input — using raw transcript.")
                return raw_transcript
            commentary_signals = [
                "no meaningful", "no text", "not provided", "i can't",
                "i cannot", "as an ai", "here is", "here's the",
                "note:", "sorry",
            ]
            cleaned_lower = cleaned.lower()
            if any(sig in cleaned_lower for sig in commentary_signals):
                print(f"[OllamaCleanup] LLM commented instead of formatting — using raw transcript.")
                return raw_transcript
            return cleaned
        except Exception as exc:
            print(f"[OllamaCleanup] Fail-open ({type(exc).__name__}): {exc}")
            return raw_transcript

    def warmup(self) -> bool:
        """Send a dummy request to force Ollama to load the model into GPU memory.

        Returns True if warmup succeeded, False otherwise.
        """
        try:
            resp = self._client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": build_cleanup_system()},
                    ] + self._few_shot + [
                        {"role": "user", "content": "hello"}
                    ],
                    "stream": False,
                    "keep_alive": "10m",
                    "options": {"num_predict": 1},
                },
            )
            resp.raise_for_status()
            return True
        except Exception as exc:
            print(
                f"[OllamaCleanup] WARNING: Warmup failed ({type(exc).__name__}: {exc}). "
                "First dictation may be slow. Is Ollama running?"
            )
            return False

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
