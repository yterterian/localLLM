#!/bin/bash
set -e

echo "📥 Installing Ollama models..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found. Please install Ollama first."
    echo "Visit: https://ollama.com/download"
    exit 1
fi

# Start Ollama service
echo "🔄 Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!
sleep 5

# List of required models
MODELS=(
    "qwen2.5:7b"
    "qwen2.5-coder:32b"
    "llama4:16x17b"
    "qwen3:30b-a3b"
    "mxbai-embed-large"
)

for MODEL in "${MODELS[@]}"; do
    if ! ollama list | grep -q "$MODEL"; then
        echo "📥 Model $MODEL not found. Installing..."
        ollama pull "$MODEL"
    else
        echo "🔄 Model $MODEL already installed. Checking for updates..."
        ollama pull "$MODEL"
    fi
done

echo "✅ Models installed and up to date!"

# Stop the background Ollama process
kill $OLLAMA_PID 2>/dev/null || true
