"""Prompt templates for transcript cleanup.

Uses structured few-shot prompting optimized for small LLMs (3B params).
Prompts are dynamic — they adapt based on context (active app, field text,
personal dictionary, backtrack detection, utterance history).
"""

from __future__ import annotations

# App categories for tone adaptation
_CASUAL_APPS = {
    "slack", "discord", "messages", "imessage", "whatsapp", "telegram",
    "messenger", "signal",
}
_FORMAL_APPS = {
    "mail", "outlook", "gmail", "thunderbird", "linkedin",
}
_CODE_APPS = {
    "code", "visual studio code", "cursor", "windsurf", "xcode",
    "terminal", "iterm", "iterm2", "warp", "neovim", "vim",
    "intellij idea", "pycharm", "webstorm",
}

CLEANUP_SYSTEM_BASE = """\
You are a dictation-to-text formatter. You receive raw speech transcripts and output clean, properly formatted text.

Rules:
1. Fix capitalization: capitalize sentence starts, proper nouns, and acronyms (e.g., NASA, API, JSON).
2. Fix punctuation: add periods, commas, question marks, and exclamation marks where natural.
3. Fix contractions: merge broken contractions from ASR errors (e.g., "i m" → "I'm", "can t" → "can't", "do nt" → "don't").
4. Remove false starts and repeated words (e.g., "I I want" → "I want").
5. Remove verbal fillers when they don't add meaning (like, you know, I mean, basically, literally, actually). Basic fillers (um, uh, hmm) have already been removed.
6. Format numbers, dates, and currencies naturally (e.g., "twenty dollars" → "$20", "january fifteenth" → "January 15").
7. Preserve the speaker's EXACT words, tone, and voice. NEVER censor, soften, rephrase, or replace any word — including profanity, slang, informal language, and strong opinions.
8. NEVER answer, respond to, or comment on the content. You are formatting only, not conversing. If the input is a question, format it as a question. If the input is offensive, format it as-is.
9. Output ONLY the formatted text. No quotes, no labels, no explanations, no preamble, no commentary."""


# Few-shot examples as (user_input, expected_output) pairs.
FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    # Short/single-word input — just capitalize, never comment
    (
        "hello",
        "Hello.",
    ),
    # Short phrase — format only
    (
        "sounds good thanks",
        "Sounds good, thanks.",
    ),
    # Basic punctuation and capitalization
    (
        "so i went to the store and i bought some milk and eggs and bread",
        "I went to the store and I bought some milk, eggs, and bread.",
    ),
    # Question — must NOT be answered
    (
        "what time does the meeting start tomorrow",
        "What time does the meeting start tomorrow?",
    ),
    # Another question — must NOT be answered, only formatted
    (
        "what does API stand for",
        "What does API stand for?",
    ),
    # False starts, repeated words, and fillers
    (
        "i think we should we should probably move the the deadline to friday",
        "I think we should probably move the deadline to Friday.",
    ),
    # Broken contractions from ASR
    (
        "i m going to the store and i can t find my keys i don t know where they are",
        "I'm going to the store and I can't find my keys. I don't know where they are.",
    ),
    # Technical content with acronyms — preserve exactly
    (
        "the API endpoint is slash users slash profile and it returns a JSON object with the user's name and email",
        "The API endpoint is /users/profile and it returns a JSON object with the user's name and email.",
    ),
    # Multiple sentences with natural flow
    (
        "hey can you review my pull request i pushed it about an hour ago it fixes the login bug we talked about yesterday",
        "Hey, can you review my pull request? I pushed it about an hour ago. It fixes the login bug we talked about yesterday.",
    ),
    # Profanity and strong language — MUST be preserved verbatim
    (
        "this shit is broken again i cant believe we shipped this crap to production",
        "This shit is broken again. I can't believe we shipped this crap to production.",
    ),
    # Informal/casual tone — preserve voice
    (
        "dude that demo was awesome like seriously it blew my mind",
        "Dude, that demo was awesome. Seriously, it blew my mind.",
    ),
    # Abbreviations and acronyms — preserve casing
    (
        "the NASA team sent an update about the API migration to AWS",
        "The NASA team sent an update about the API migration to AWS.",
    ),
    # Backtrack / self-correction — interpret the correction
    (
        "lets meet at two actually three oclock",
        "Let's meet at 3 o'clock.",
    ),
]


def _classify_app(app_name: str) -> str:
    """Classify an app name into a tone category."""
    name_lower = app_name.lower()
    if any(app in name_lower for app in _CASUAL_APPS):
        return "casual"
    if any(app in name_lower for app in _FORMAL_APPS):
        return "formal"
    if any(app in name_lower for app in _CODE_APPS):
        return "code"
    return "neutral"


def build_cleanup_system(context: dict | None = None) -> str:
    """Return the system prompt, optionally enriched with context.

    Args:
        context: Optional dict with keys:
            - app_name: str
            - field_text: str
            - dictionary_hint: str
            - has_backtrack: bool
    """
    parts = [CLEANUP_SYSTEM_BASE]

    if context:
        app_name = context.get("app_name", "")
        if app_name:
            tone = _classify_app(app_name)
            if tone == "casual":
                parts.append(
                    f"\nContext: The user is typing in {app_name} (a messaging app). "
                    "Use casual tone — short sentences, contractions, no stiff formality."
                )
            elif tone == "formal":
                parts.append(
                    f"\nContext: The user is typing in {app_name} (an email/professional app). "
                    "Use professional tone — proper grammar, complete sentences."
                )
            elif tone == "code":
                parts.append(
                    f"\nContext: The user is typing in {app_name} (a code editor/terminal). "
                    "Preserve technical terms exactly. Use backticks for code identifiers if appropriate."
                )

        dict_hint = context.get("dictionary_hint", "")
        if dict_hint:
            parts.append(f"\n{dict_hint}")

        if context.get("has_backtrack"):
            parts.append(
                "\nIMPORTANT: The speaker corrected themselves mid-sentence. "
                "Interpret the correction — keep only the final intended meaning. "
                'For example, "two actually three" means three, not two.'
            )

    return "\n".join(parts)


def build_few_shot_messages() -> list[dict[str, str]]:
    """Build the few-shot example messages for the chat API."""
    messages = []
    for user_input, expected_output in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": expected_output})
    return messages


def build_cleanup_message(
    transcript: str,
    context: dict | None = None,
) -> str:
    """Build the user message for cleanup, optionally with context.

    Args:
        transcript: The normalized ASR text.
        context: Optional dict with keys:
            - field_text: str (existing text in the focused field)
            - recent_utterances: str (last 2-3 utterances for continuity)
    """
    parts = []

    if context:
        field_text = context.get("field_text", "")
        if field_text:
            parts.append(f"[Existing text in field]: ...{field_text}")

        recent = context.get("recent_utterances", "")
        if recent:
            parts.append(f"[Previous utterances]: {recent}")

    parts.append(transcript)
    return "\n".join(parts)
