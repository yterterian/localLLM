#!/bin/bash
set -e

# Load environment
source .env 2>/dev/null || true

echo "🚀 Starting Local LLM System..."

# Start Ollama in background
echo "🔄 Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!
sleep 3

# Start the router
echo "🤖 Starting Cline Memory Bank Router..."
python src/cline/memory_aware_router.py &
ROUTER_PID=$!

# Start the context expansion server
echo "🔍 Starting Context Expansion Server..."
python src/integration/cline_server.py &
CONTEXT_PID=$!

echo "✅ System started successfully!"
echo "📊 Router API: http://localhost:8000"
echo "🔍 Context API: http://localhost:8001"
echo "📖 API docs: http://localhost:8000/docs"

# Cleanup function
cleanup() {
    echo "🛑 Shutting down system..."
    kill $CONTEXT_PID $ROUTER_PID $OLLAMA_PID 2>/dev/null || true
    exit 0
}

# Trap signals for clean shutdown
trap cleanup SIGINT SIGTERM

# Wait for processes
wait
