# Intelligence Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add context-awareness, backtrack detection, personal dictionary, screen context, voice commands, entity formatting, and Whisper initial prompt to transform basic dictation into an intelligent pipeline.

**Architecture:** New `context/` package reads active app + text field via macOS AX APIs. New `commands/` package detects and routes voice commands. Prompts become dynamic, accepting context dict. Dictionary loaded from TOML config. All context flows through `process_utterance` in app.py.

**Tech Stack:** AppKit/Quartz (already imported), ApplicationServices AX APIs (via pyobjc), TOML config.

---

### Task 1: Screen Context — App Name + Text Field Reader

**Files:**
- Create: `src/dictation/context/__init__.py`
- Create: `src/dictation/context/screen.py`
- Test: `tests/test_screen_context.py`

Reads the frontmost app name via NSWorkspace and the focused text field content via AX accessibility APIs. Returns a dict: `{"app_name": "Slack", "field_text": "last ~200 chars..."}`.

### Task 2: Personal Dictionary

**Files:**
- Create: `src/dictation/context/dictionary.py`
- Create: `config/dictionary.toml`
- Test: `tests/test_dictionary.py`

Loads custom terms from `config/dictionary.toml`. Returns a list of terms to inject into LLM prompt and Whisper initial_prompt.

### Task 3: Backtrack Detection

**Files:**
- Modify: `src/dictation/transcript/buffer.py`
- Test: `tests/test_buffer.py` (add cases)

Detect backtrack phrases ("actually", "scratch that", "no wait", "I meant") and set a flag so the LLM knows to interpret corrections.

### Task 4: Dynamic Prompts with Context

**Files:**
- Modify: `src/dictation/cleanup/prompts.py`
- Modify: `src/dictation/cleanup/base.py`
- Test: `tests/test_cleanup.py` (add cases)

Make prompt building accept a context dict (app_name, field_text, recent_utterances, dictionary_terms, has_backtrack, entity_formatting). System prompt and user message become dynamic.

### Task 5: Wire Context into OllamaCleanup

**Files:**
- Modify: `src/dictation/cleanup/ollama.py`
- Test: `tests/test_cleanup.py` (update)

`cleanup()` accepts optional context dict, passes to dynamic prompt builder.

### Task 6: Voice Command Handler

**Files:**
- Create: `src/dictation/commands/__init__.py`
- Create: `src/dictation/commands/handler.py`
- Test: `tests/test_commands.py`

Detects command-intent utterances ("delete that", "undo that", "replace X with Y", "make this a list"). Returns either a command action or None (meaning normal dictation).

### Task 7: Whisper Initial Prompt

**Files:**
- Modify: `src/dictation/asr/whisper_cpp.py`
- Test: `tests/test_asr.py` (add case)

Accept an `initial_prompt` string (dictionary terms + recent context) to bias Whisper toward correct recognition.

### Task 8: Orchestrate in app.py

**Files:**
- Modify: `src/dictation/app.py`
- Modify: `config/default.toml`

Wire everything together: screen context capture before cleanup, dictionary loading at startup, backtrack flag from buffer, voice command routing, Whisper initial prompt, multi-utterance context from TranscriptBuffer.
