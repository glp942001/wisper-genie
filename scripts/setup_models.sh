#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

echo "=== Dictation Pipeline — Model Setup ==="

# 1. Download whisper.cpp large-v3-turbo model
echo ""
echo "--- Downloading whisper large-v3-turbo model ---"
mkdir -p "$MODELS_DIR"
MODEL_FILE="$MODELS_DIR/ggml-large-v3-turbo.bin"

if [ -f "$MODEL_FILE" ]; then
    echo "Model already exists at $MODEL_FILE, skipping download."
else
    echo "Downloading ggml-large-v3-turbo.bin (~1.6GB)..."
    curl -L --progress-bar \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin" \
        -o "$MODEL_FILE"
    echo "Downloaded to $MODEL_FILE"
fi

# 2. Pull Ollama ministral:3b
echo ""
echo "--- Pulling Ollama ministral:3b model ---"
if command -v ollama &> /dev/null; then
    ollama pull qwen3.5:2b
    echo "Ollama qwen3.5:2b ready."
else
    echo "WARNING: ollama not found. Install from https://ollama.com and run: ollama pull qwen3.5:2b"
fi

echo ""
echo "=== Setup complete ==="
echo "Whisper model: $MODEL_FILE (large-v3-turbo)"
echo "Ollama model: qwen3.5:2b"
