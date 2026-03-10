# Wisper Genie

AI-powered dictation for macOS. Hold a key, speak naturally, and polished text appears wherever your cursor is — Slack, email, code editors, anywhere.

Everything runs locally on your Mac. No audio leaves your machine.

## Features

- **Push-to-talk or live dictation** — Hold Right Option for precise capture, or switch to live mode and speak naturally with VAD-based endpointing.
- **Screen context** — Reads text near the cursor in the focused field so it continues naturally (capitalization, tone, sentence flow).
- **Ghost preview overlay** — Shows a transient HUD with the text about to be inserted before it lands in the field.
- **Backtrack detection** — Say "actually" or "scratch that" mid-sentence and it rewrites to your corrected intent.
- **Multi-utterance context** — Remembers your last few utterances so follow-up dictation flows naturally.
- **Personal writing memory** — Learns preferred punctuation, capitalization, sign-offs, and common phrases locally on disk.
- **Personal dictionary** — Add names, jargon, and acronyms to `config/dictionary.toml` for better recognition.
- **Voice commands** — "Delete that", "undo", "select all", "replace X with Y".
- **Multi-hypothesis ASR reranking** — Can compare multiple prompted variants plus an unprompted baseline against screen context and dictionary terms.
- **Direct insertion first** — Uses Accessibility-based text insertion when supported, then falls back to typed-text events or clipboard paste safely.
- **LLM cleanup always-on** — Every non-command utterance goes through the local LLM for punctuation, capitalization, contractions, filler cleanup, and phrasing polish.
- **Entity formatting** — Numbers, dates, and currencies formatted naturally ("twenty dollars" becomes "$20").
- **Smart mic selection** — Auto-detects AirPods and external mics, prefers them over built-in.
- **Whisper hallucination filtering** — Strips `[BLANK_AUDIO]` and other artifacts from silent recordings.
- **Local routing metrics** — Writes per-utterance timing and routing decisions to a local JSONL log for tuning.
- **Auto-start on login** — Optional launch agent via `wisper-genie autostart`.

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
wisper-genie              # Start dictating
wisper-genie metrics      # Show recent routing/latency summary
wisper-genie --help       # Show all commands
wisper-genie install      # Re-download models if needed
wisper-genie autostart    # Launch on login
wisper-genie autostart --remove  # Stop launching on login
wisper-genie uninstall    # Remove Wisper Genie completely
```

**Dictating:**

1. Run `wisper-genie`
2. In push-to-talk mode, hold **Right Option**, speak, and release
3. In live mode, speak naturally and pause briefly to send
4. A ghost preview appears, then the final text is inserted directly when possible, or pasted as a fallback

**Voice commands** (say these instead of dictating text):

| Command | What it does |
|---|---|
| "delete that" / "scratch that" | Undo the last paste (Cmd+Z) |
| "undo" | Undo (Cmd+Z) |
| "select all" | Select all (Cmd+A) |
| "replace X with Y" | Replace unique text safely, or replace the current selection |

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
Audio → Capture ring buffer / live VAD
  → Context prefetch (frontmost app + focused field snapshot)
  → ASR (Whisper small, multiple prompt variants + unprompted baseline)
  → Candidate reranking (screen context + history + dictionary)
  → Normalize (filler removal, dictation commands, backtrack detection)
  → Voice command check (routes commands to handler, skips LLM)
  → LLM cleanup (Ollama qwen3.5:2b, with field, history, dictionary, and writing memory)
  → Ghost preview overlay
  → Direct AX insertion, typed-text fallback, or clipboard fallback
  → Local style-memory + routing metrics persistence
```

### Components

| Component | Path | Description |
|---|---|---|
| Orchestrator | `src/dictation/app.py` | Main pipeline, hotkey listener, threading |
| ASR | `src/dictation/asr/whisper_cpp.py` | Whisper.cpp with Metal acceleration |
| Audio capture | `src/dictation/audio/capture.py` | Mic input via sounddevice, auto-selection |
| Live VAD | `src/dictation/audio/vad.py` | Speech start/end detection for hands-free mode |
| Transcript buffer | `src/dictation/transcript/buffer.py` | Filler removal, dictation commands, backtrack |
| LLM cleanup | `src/dictation/cleanup/ollama.py` | Ollama API adapter, fail-open design |
| Prompts | `src/dictation/cleanup/prompts.py` | Dynamic prompt building with context |
| Screen context | `src/dictation/context/screen.py` | App name + text field via macOS AX APIs |
| Context cache | `src/dictation/context/cache.py` | Prefetches and reuses focused field metadata |
| Style memory | `src/dictation/context/style_memory.py` | Learns local writing preferences from accepted output |
| Dictionary | `src/dictation/context/dictionary.py` | Custom vocabulary loader |
| Voice commands | `src/dictation/commands/handler.py` | Command detection and execution |
| Preview overlay | `src/dictation/output/overlay.py` | Ghost HUD before insertion |
| Text injection | `src/dictation/output/injector.py` | Direct AX insertion with clipboard fallback |
| Latency tracker | `src/dictation/telemetry/latency.py` | Per-stage timing and budget warnings |
| Routing metrics | `src/dictation/telemetry/metrics.py` | Local JSONL event log for routing and latency |
| CLI | `src/dictation/cli.py` | Autostart, uninstall, and metrics subcommands |

## Configuration

### Main config — `config/default.toml`

```toml
[audio]
device = "auto"          # "auto" prefers AirPods/external, "default" for system default

[input]
mode = "push_to_talk"    # or "live"

[asr]
model_path = "models/ggml-small.bin"
language = "en"

[cleanup]
model = "qwen3.5:2b"
timeout_ms = 10000       # Fail-open: raw text used if LLM is slow/down

[hotkey]
keys = ["<alt_r>"]       # Right Option key

[output]
prefer_direct_insert = true
prefer_typing_fallback = true

[preview]
enabled = true

[metrics]
enabled = true

[routing]
max_hypotheses = 4

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

### Screen context

Using macOS Accessibility APIs, the tool reads text near the insertion point in the focused field when available. This lets the LLM:

- Continue a sentence without re-capitalizing
- Match the tone of existing text
- Avoid re-greeting in replies

For destructive voice commands like "replace X with Y", the app only applies the change when the match is unambiguous or the exact target text is already selected.

### Live mode and cached context

When live mode is enabled, Silero VAD starts a dictation segment when speech begins and ends it after a short silence. On either hotkey press or live speech start, the app prefetches screen context and caches it briefly so cleanup and insertion can reuse the same focused-field snapshot.

### Local writing memory

Accepted output is stored as local style hints in `~/.wisper-genie/style_memory.json`. Those hints bias cleanup toward the user's preferred punctuation density, capitalization, greetings, sign-offs, and common phrases without sending anything off-device.

### Routing metrics

Each dictation can write a local event to `~/.wisper-genie/routing_metrics.jsonl`, including ASR confidence, candidate count, injection route, and per-stage timings. Use `wisper-genie metrics` to print a compact summary.

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
- If direct insertion is unsupported in the current app, Wisper Genie will fall back to clipboard paste automatically

**Ollama errors / "Fail-open" messages**
- Make sure Ollama is running: `ollama serve` or open Ollama.app
- Check the model is pulled: `ollama list` should show `qwen3.5:2b`
- Re-pull if needed: `ollama pull qwen3.5:2b`

**"command not found: wisper-genie"**
- Run `source ~/.zshrc` or open a new terminal
- Verify: `ls ~/.local/bin/wisper-genie`

**Inspect routing quality and latency**
- Run `wisper-genie metrics`
- Check `~/.wisper-genie/routing_metrics.jsonl` for per-utterance details

**Uninstall**
```bash
wisper-genie uninstall
```
This removes `~/.wisper-genie/`, the `dictation` command, and the autostart plist. Ollama and its models are left in place (shared system tool). To remove those too: `brew uninstall --cask ollama && ollama rm qwen3.5:2b`.

**Reinstall from scratch**
```bash
wisper-genie uninstall
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
