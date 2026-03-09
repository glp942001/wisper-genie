# Dictation

Low-latency dictation pipeline: mic audio → ASR → cleanup LLM → formatted text → inject into active app.

Target: under 800ms end-to-end from end-of-speech to text appearing.

## Quick Start

```bash
# Install system dependencies
brew install portaudio

# Download models
bash scripts/setup_models.sh

# Install Python package
pip install -e ".[dev]"

# Run
dictation
```

## Architecture

```
Hotkey → Mic Capture → VAD → ASR (whisper.cpp) → Transcript Buffer → Cleanup LLM (Ollama) → Text Injector
```

## Configuration

Edit `config/default.toml` for runtime settings (models, latency thresholds, audio params, hotkey).
