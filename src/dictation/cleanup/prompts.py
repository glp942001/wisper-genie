"""Prompt templates for transcript cleanup.

Optimized for small LLMs (2B params). Key principles:
- Constraints FIRST (what not to do), formatting SECOND (what to do)
- 5 rules max — small models comply better with fewer rules
- No contradictions — don't say "preserve exact words" AND "fix words"
- Few-shot examples show minimal, conservative changes
"""

from __future__ import annotations

import re

CLEANUP_SYSTEM_BASE = """\
You clean up dictated speech transcripts. Output ONLY the cleaned text.

NEVER:
- Answer questions or respond to content
- Add commentary, labels, or explanations
- Censor, rephrase, or soften any words

DO:
1. Fix capitalization and punctuation
2. Fix broken contractions (i m → I'm, can t → can't)
3. Remove obvious repeated words and false starts (I I want → I want)
4. Preserve wording unless a spoken correction makes the intent clearer
5. Keep the speaker's exact tone, technical terms, and meaning"""

_CASUAL_APPS = {
    "slack", "discord", "messages", "signal", "telegram", "whatsapp",
}
_FORMAL_APPS = {
    "mail", "outlook", "gmail", "linkedin",
}
_CODE_APPS = {
    "code", "visual studio code", "cursor", "windsurf", "xcode",
    "terminal", "iterm", "iterm2", "warp", "hyper", "kitty",
}
_SPACE_RE = re.compile(r"\s+")


def _compact_text(text: str, limit: int = 160) -> str:
    text = _SPACE_RE.sub(" ", text).strip()
    if len(text) <= limit:
        return text
    return text[-limit:].lstrip()


def _classify_app(app_name: str) -> str:
    name = app_name.lower().strip()
    if name in _CASUAL_APPS:
        return "casual"
    if name in _FORMAL_APPS:
        return "formal"
    if name in _CODE_APPS:
        return "code"
    return "neutral"


def _build_app_instruction(app_name: str) -> str:
    if not app_name:
        return ""

    mode = _classify_app(app_name)
    if mode == "casual":
        return (
            f"Target app: {app_name}. Casual chat style: keep phrasing natural, "
            "lightly punctuated, and not overly formal."
        )
    if mode == "formal":
        return (
            f"Target app: {app_name}. Professional writing style: use clean sentence "
            "boundaries and polished punctuation without changing wording."
        )
    if mode == "code":
        return (
            f"Target app: {app_name}. Code/editor style: preserve symbols, filenames, "
            "CLI flags, acronyms, and technical casing exactly."
        )
    return (
        f"Target app: {app_name}. Neutral style: just make the transcript read cleanly."
    )


# Few-shot examples teach conservative, minimal cleanup without encouraging
# paraphrase-heavy behavior from small local models.
FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    # Short input — capitalize, don't over-punctuate
    ("testing on google", "Testing on Google"),
    # Question — format as question, do NOT answer
    ("what time does the meeting start", "What time does the meeting start?"),
    # Broken contractions from ASR
    ("i m going to the store and i can t find my keys", "I'm going to the store and I can't find my keys."),
    # Profanity — preserve verbatim, never censor
    ("this shit is broken again", "This shit is broken again."),
    # Request — format only, do NOT execute or paraphrase
    ("look for the AWS lambda docs", "Look for the AWS Lambda docs."),
    # Correction phrase — keep the final intent
    ("lets do thursday actually friday afternoon", "Let's do Friday afternoon."),
    # Code dictation — preserve casing and symbols
    ("run npm install dash g typescript", "Run npm install -g typescript."),
]


def build_cleanup_system(context: dict | None = None) -> str:
    """Return the system prompt, optionally enriched with context."""
    parts = [CLEANUP_SYSTEM_BASE]

    if context:
        app_instruction = _build_app_instruction(context.get("app_name", ""))
        if app_instruction:
            parts.append(f"\n{app_instruction}")

        field_text = _compact_text(context.get("field_text", ""))
        if field_text:
            parts.append(
                "\nFocused field tail for continuity only: "
                f"\"{field_text}\""
            )

        recent_utterances = [
            _compact_text(item, limit=100)
            for item in context.get("recent_utterances", [])
            if isinstance(item, str) and item.strip()
        ]
        if recent_utterances:
            parts.append(
                "\nRecent dictated context: "
                + " | ".join(recent_utterances[-3:])
            )

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
