#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

echo "=== Dictation Pipeline — Model Setup ==="

# 1. Download whisper.cpp small model
echo ""
echo "--- Downloading whisper small model ---"
mkdir -p "$MODELS_DIR"
MODEL_FILE="$MODELS_DIR/ggml-small.bin"

if [ -f "$MODEL_FILE" ]; then
    echo "Model already exists at $MODEL_FILE, skipping download."
else
    echo "Downloading ggml-small.bin (~465MB)..."
    curl -L --progress-bar \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin" \
        -o "$MODEL_FILE"
    echo "Downloaded to $MODEL_FILE"
fi

# 2. Pull Ollama qwen3.5:2b
echo ""
echo "--- Pulling Ollama qwen3.5:2b model ---"
if command -v ollama &> /dev/null; then
    ollama pull qwen3.5:2b
    echo "Ollama qwen3.5:2b ready."
else
    echo "WARNING: ollama not found. Install from https://ollama.com and run: ollama pull qwen3.5:2b"
fi

echo ""
echo "=== Setup complete ==="
echo "Whisper model: $MODEL_FILE (small)"
echo "Ollama model: qwen3.5:2b"
