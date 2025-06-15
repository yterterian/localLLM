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

# Install models
echo "📥 Downloading small model (qwen2.5:7b)..."
ollama pull qwen2.5:7b

echo "📥 Downloading large model (qwen2.5-coder:32b)..."
ollama pull qwen2.5-coder:32b

echo "📥 Downloading llama4:16x17b..."
ollama pull llama4:16x17b

echo "📥 Downloading qwen3:30b-a3b..."
ollama pull qwen3:30b-a3b

echo "✅ Models installed successfully!"

# Stop the background Ollama process
kill $OLLAMA_PID 2>/dev/null || true
