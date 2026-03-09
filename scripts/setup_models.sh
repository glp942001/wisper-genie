#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

echo "=== Dictation Pipeline — Model Setup ==="

# 1. Download whisper.cpp medium model
echo ""
echo "--- Downloading whisper medium model ---"
mkdir -p "$MODELS_DIR"
MODEL_FILE="$MODELS_DIR/ggml-medium.bin"

if [ -f "$MODEL_FILE" ]; then
    echo "Model already exists at $MODEL_FILE, skipping download."
else
    echo "Downloading ggml-medium.bin (~1.5GB)..."
    curl -L --progress-bar \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin" \
        -o "$MODEL_FILE"
    echo "Downloaded to $MODEL_FILE"
fi

# 2. Pull Ollama ministral:3b
echo ""
echo "--- Pulling Ollama ministral:3b model ---"
if command -v ollama &> /dev/null; then
    ollama pull ministral-3:3b
    echo "Ollama ministral-3:3b ready."
else
    echo "WARNING: ollama not found. Install from https://ollama.com and run: ollama pull ministral-3:3b"
fi

echo ""
echo "=== Setup complete ==="
echo "Whisper model: $MODEL_FILE (medium)"
echo "Ollama model: ministral-3:3b"
