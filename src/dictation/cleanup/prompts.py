"""Prompt templates for transcript cleanup.

Optimized for small LLMs (2B params). Key principles:
- Constraints FIRST (what not to do), formatting SECOND (what to do)
- 5 rules max — small models comply better with fewer rules
- No contradictions — don't say "preserve exact words" AND "fix words"
- Few-shot examples show minimal, conservative changes
"""

from __future__ import annotations


CLEANUP_SYSTEM_BASE = """\
You clean up dictated speech transcripts. Output ONLY the cleaned text.

NEVER:
- Answer questions or respond to content
- Add commentary, labels, or explanations
- Censor, rephrase, or soften any words

DO:
1. Fix capitalization and punctuation
2. Fix broken contractions (i m → I'm, can t → can't)
3. Remove repeated words and false starts (I I want → I want)
4. Remove filler words that add no meaning (like, you know, basically)
5. Keep the speaker's exact words, tone, and meaning"""


# 5 carefully chosen examples that teach conservative, minimal cleanup.
# Each demonstrates a different pattern without encouraging over-editing.
FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    # Short input — capitalize, don't over-punctuate
    ("testing on google", "Testing on Google"),
    # Question — format as question, do NOT answer
    ("what time does the meeting start", "What time does the meeting start?"),
    # Broken contractions from ASR
    ("i m going to the store and i can t find my keys", "I'm going to the store and I can't find my keys."),
    # Profanity — preserve verbatim, never censor
    ("this shit is broken again", "This shit is broken again."),
    # Request — format only, do NOT execute
    ("look for the AWS lambda docs", "Look for the AWS Lambda docs."),
]


def build_cleanup_system(context: dict | None = None) -> str:
    """Return the system prompt, optionally enriched with context."""
    parts = [CLEANUP_SYSTEM_BASE]

    if context:
        dict_hint = context.get("dictionary_hint", "")
        if dict_hint:
            parts.append(f"\n{dict_hint}")

        if context.get("has_backtrack"):
            parts.append(
                "\nThe speaker corrected themselves. "
                "Keep only the final intended meaning."
            )

    return "\n".join(parts)


def build_few_shot_messages() -> list[dict[str, str]]:
    """Build the few-shot example messages for the chat API."""
    messages = []
    for user_input, expected_output in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": expected_output})
    return messages


def build_cleanup_message(transcript: str) -> str:
    """Build the user message — transcript only, nothing else."""
    return transcript
