# Wisper Genie

AI-powered dictation for macOS. Hold a key, speak naturally, and polished text appears wherever your cursor is — Slack, email, code editors, anywhere.

Everything runs locally on your Mac. No audio leaves your machine.

## Features

- **Push-to-talk dictation** — Hold Right Option, speak, release. Text appears instantly.
- **App-aware formatting** — Detects the active app and adapts tone: casual for Slack, professional for Mail, code-aware for VS Code.
- **Screen context** — Reads existing text in the focused field so it continues naturally (capitalization, tone, sentence flow).
- **Backtrack detection** — Say "actually" or "scratch that" mid-sentence and it rewrites to your corrected intent.
- **Multi-utterance context** — Remembers your last few utterances so follow-up dictation flows naturally.
- **Personal dictionary** — Add names, jargon, and acronyms to `config/dictionary.toml` for better recognition.
- **Voice commands** — "Delete that", "undo", "select all", "replace X with Y".
- **LLM cleanup** — Fixes punctuation, capitalization, contractions, filler words, and false starts via a local LLM.
- **Entity formatting** — Numbers, dates, and currencies formatted naturally ("twenty dollars" becomes "$20").
- **Smart mic selection** — Auto-detects AirPods and external mics, prefers them over built-in.
- **Whisper hallucination filtering** — Strips `[BLANK_AUDIO]` and other artifacts from silent recordings.
- **Auto-start on login** — Optional launch agent via `dictation autostart`.

## Requirements

- macOS 13+ (Apple Silicon or Intel)
- ~3 GB disk space (models)
- ~2 GB RAM (during dictation)

The installer handles everything else (Homebrew, Python, Ollama, models).

## Installation

```bash
curl -fsSL https://raw.githubusercontent.com/glp942001/wisper-genie/main/install.sh | bash
```

The installer will:

1. Install Homebrew (if missing)
2. Install Python 3.12+ (if missing)
3. Clone this repo to `~/.wisper-genie/`
4. Create a Python virtual environment and install dependencies
5. Install the `dictation` command to `~/.local/bin/`
6. Prompt to install Ollama (local LLM runtime)
7. Download the Whisper small model (~465 MB)
8. Pull the qwen3.5:2b cleanup model (~2.7 GB)
9. Open macOS Accessibility and Microphone permission settings

After installation, open a new terminal (or run `source ~/.zshrc`) and you're ready.

## Usage

```bash
dictation              # Start dictating
dictation --help       # Show all commands
dictation install      # Re-download models if needed
dictation autostart    # Launch on login
dictation autostart --remove  # Stop launching on login
dictation uninstall    # Remove Wisper Genie completely
```

**Dictating:**

1. Run `dictation`
2. Hold **Right Option** key
3. Speak naturally
4. Release the key — formatted text is pasted at your cursor

**Voice commands** (say these instead of dictating text):

| Command | What it does |
|---|---|
| "delete that" / "scratch that" | Undo the last paste (Cmd+Z) |
| "undo" | Undo (Cmd+Z) |
| "select all" | Select all (Cmd+A) |
| "replace X with Y" | Replace text |

**Dictation commands** (converted to punctuation):

| Say | Get |
|---|---|
| "new paragraph" | Line break + blank line |
| "new line" | Line break |
| "question mark" | ? |
| "exclamation point" | ! |
| "open quote" / "close quote" | " |
| "period" (at end of utterance) | . |
| "comma" (at end of utterance) | , |

## Architecture

```
Audio → ASR (Whisper small, with initial_prompt from dictionary + context)
  → Normalize (filler removal, dictation commands, backtrack detection)
  → Voice command check (routes commands to handler, skips LLM)
  → Screen context capture (active app name + text field content via AX APIs)
  → LLM cleanup (Ollama qwen3.5:2b, with full context: app, field, history, dictionary)
  → Text injection (NSPasteboard + CGEvent Cmd+V)
```

### Components

| Component | Path | Description |
|---|---|---|
| Orchestrator | `src/dictation/app.py` | Main pipeline, hotkey listener, threading |
| ASR | `src/dictation/asr/whisper_cpp.py` | Whisper.cpp with Metal acceleration |
| Audio capture | `src/dictation/audio/capture.py` | Mic input via sounddevice, auto-selection |
| Transcript buffer | `src/dictation/transcript/buffer.py` | Filler removal, dictation commands, backtrack |
| LLM cleanup | `src/dictation/cleanup/ollama.py` | Ollama API adapter, fail-open design |
| Prompts | `src/dictation/cleanup/prompts.py` | Dynamic prompt building with context |
| Screen context | `src/dictation/context/screen.py` | App name + text field via macOS AX APIs |
| Dictionary | `src/dictation/context/dictionary.py` | Custom vocabulary loader |
| Voice commands | `src/dictation/commands/handler.py` | Command detection and execution |
| Text injection | `src/dictation/output/injector.py` | Clipboard paste via native macOS APIs |
| Latency tracker | `src/dictation/telemetry/latency.py` | Per-stage timing and budget warnings |
| CLI | `src/dictation/cli.py` | Autostart subcommand |

## Configuration

### Main config — `config/default.toml`

```toml
[audio]
device = "auto"          # "auto" prefers AirPods/external, "default" for system default

[asr]
model_path = "models/ggml-small.bin"
language = "en"

[cleanup]
model = "qwen3.5:2b"
timeout_ms = 10000       # Fail-open: raw text used if LLM is slow/down

[hotkey]
keys = ["<alt_r>"]       # Right Option key

[latency]
total_budget_ms = 800    # Warning threshold
```

### Personal dictionary — `config/dictionary.toml`

Add names, jargon, and technical terms for better ASR accuracy and LLM formatting:

```toml
names = [
    "Supabase",
    "Vercel",
]

terms = [
    "API", "JSON", "AWS", "OAuth",
    "Kubernetes", "kubectl", "Docker",
    "PostgreSQL", "Redis", "FastAPI",
]
```

These terms are injected into both the Whisper prompt (for better recognition) and the LLM prompt (for correct spelling).

## How It Works

### App-aware formatting

The tool reads the frontmost app via `NSWorkspace` and classifies it:

- **Casual** (Slack, Discord, Messages) — short sentences, contractions, relaxed tone
- **Formal** (Mail, Outlook, LinkedIn) — complete sentences, professional tone
- **Code** (VS Code, Cursor, Terminal) — preserves technical terms, backtick formatting
- **Neutral** (everything else) — balanced formatting

### Screen context

Using macOS Accessibility APIs, the tool reads the last 200 characters from the focused text field. This lets the LLM:

- Continue a sentence without re-capitalizing
- Match the tone of existing text
- Avoid re-greeting in replies

### Fail-open design

If Ollama is down or slow, the pipeline returns the raw transcript with basic normalization instead of hanging. The tool always produces output.

## Development

```bash
# Clone
git clone https://github.com/glp942001/wisper-genie.git
cd wisper-genie

# Install in dev mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run the tool
python -m dictation.app
```

### Project structure

```
wisper-genie/
├── config/
│   ├── default.toml          # Runtime config
│   └── dictionary.toml       # Personal vocabulary
├── src/dictation/
│   ├── app.py                # Main orchestrator
│   ├── cli.py                # CLI subcommands
│   ├── asr/                  # Speech-to-text (Whisper)
│   ├── audio/                # Mic capture + auto-selection
│   ├── cleanup/              # LLM cleanup (Ollama)
│   ├── commands/             # Voice command handler
│   ├── context/              # Screen context + dictionary
│   ├── output/               # Text injection
│   ├── telemetry/            # Latency tracking
│   └── transcript/           # Normalization + backtrack
├── tests/                    # Test suite
├── scripts/
│   ├── setup_models.sh       # Model downloader
│   └── dictation-wrapper     # CLI wrapper for ~/.local/bin
├── install.sh                # One-line installer
└── pyproject.toml            # Build config
```

### Running tests

```bash
pytest tests/ -v
```

Note: `test_asr.py` and `test_injector.py` require native macOS dependencies (`pywhispercpp`, `AppKit`) and may skip on CI or non-macOS environments.

## Troubleshooting

**"No speech detected" on every recording**
- Check that your mic is working: `python scripts/test_mic.py`
- If laptop lid is closed, you need AirPods or an external mic
- Grant Microphone permission to your terminal app

**"[BLANK_AUDIO]" or silence**
- Same as above — the built-in mic is muffled or disabled
- Connect AirPods/headphones and restart `dictation`

**Text doesn't appear / paste doesn't work**
- Grant Accessibility permission to your terminal app
- System Settings → Privacy & Security → Accessibility → add your terminal

**Ollama errors / "Fail-open" messages**
- Make sure Ollama is running: `ollama serve` or open Ollama.app
- Check the model is pulled: `ollama list` should show `qwen3.5:2b`
- Re-pull if needed: `ollama pull qwen3.5:2b`

**"command not found: dictation"**
- Run `source ~/.zshrc` or open a new terminal
- Verify: `ls ~/.local/bin/dictation`

**Uninstall**
```bash
dictation uninstall
```
This removes `~/.wisper-genie/`, the `dictation` command, and the autostart plist. Ollama and its models are left in place (shared system tool). To remove those too: `brew uninstall --cask ollama && ollama rm qwen3.5:2b`.

**Reinstall from scratch**
```bash
dictation uninstall
curl -fsSL https://raw.githubusercontent.com/glp942001/wisper-genie/main/install.sh | bash
```

## Tech Stack

| Component | Technology |
|---|---|
| Speech-to-text | [whisper.cpp](https://github.com/ggerganov/whisper.cpp) via pywhispercpp (Metal GPU) |
| LLM cleanup | [Ollama](https://ollama.com) with qwen3.5:2b |
| Audio capture | [sounddevice](https://python-sounddevice.readthedocs.io/) (PortAudio) |
| Hotkey listener | [pynput](https://pynput.readthedocs.io/) |
| Text injection | Native macOS APIs (NSPasteboard + Quartz CGEvent) |
| Screen context | macOS Accessibility APIs (AXUIElement) |
| Config | TOML (Python 3.11+ tomllib) |

## License

Internal use only. Do not distribute outside the organization.
