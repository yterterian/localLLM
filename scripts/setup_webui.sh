#!/bin/bash
# File: scripts/setup_openwebui_native.sh
# Native Open WebUI setup without Docker

set -e

echo "🚀 Setting up Open WebUI (Native Python Installation)"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Using virtual environment: $VIRTUAL_ENV"
else
    echo "⚠️  Warning: Not in a virtual environment"
    echo "   Consider activating your conda/venv environment first"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install Open WebUI
echo "📦 Installing Open WebUI..."
pip install open-webui

# Create configuration directory
mkdir -p config/openwebui

# Create Open WebUI environment file
echo "📝 Creating Open WebUI configuration..."
cat > config/openwebui/.env << EOF
# Open WebUI Configuration
WEBUI_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
WEBUI_URL=http://localhost:3000
DATA_DIR=./data/openwebui

# Connect to your local router
OPENAI_API_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=dummy-key

# Optional: Default models (your router will handle routing)
DEFAULT_MODELS=qwen2.5:7b,qwen2.5-coder:32b,llama4:16x17b

# Enable features
ENABLE_RAG_INGESTION=true
ENABLE_RAG_WEB_SEARCH=true
ENABLE_IMAGE_GENERATION=false
EOF

# Create data directory
mkdir -p data/openwebui

# Add CORS support to your router
echo "🔧 Adding CORS support to your router..."

ROUTER_FILE="src/cline/memory_aware_router.py"
if [ -f "$ROUTER_FILE" ]; then
    # Check if CORS is already added
    if ! grep -q "CORSMiddleware" "$ROUTER_FILE"; then
        echo "  Adding CORS middleware..."
        
        # Create backup
        cp "$ROUTER_FILE" "${ROUTER_FILE}.backup"
        
        # Add CORS import
        sed -i '/from fastapi import FastAPI/a from fastapi.middleware.cors import CORSMiddleware' "$ROUTER_FILE"
        
        # Add CORS middleware after app creation
        sed -i '/^app = FastAPI(/a\\n# Add CORS middleware for Open WebUI\napp.add_middleware(\n    CORSMiddleware,\n    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],\n    allow_credentials=True,\n    allow_methods=["*"],\n    allow_headers=["*"],\n)' "$ROUTER_FILE"
        
        echo "  ✅ CORS support added to router"
    else
        echo "  ✅ CORS support already present"
    fi
else
    echo "  ⚠️  Router file not found at $ROUTER_FILE"
fi

# Create startup script
echo "📜 Creating startup script..."
cat > scripts/start_with_openwebui.sh << 'EOF'
#!/bin/bash
# File: scripts/start_with_openwebui.sh
# Start the complete system with Open WebUI

set -e

echo "🚀 Starting Local LLM System with Open WebUI..."

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "Port $1 is already in use"
        return 1
    fi
    return 0
}

# Function to wait for service
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $name to start..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            echo "✅ $name is ready!"
            return 0
        fi
        echo "  Attempt $attempt/$max_attempts..."
        sleep 2
        ((attempt++))
    done
    
    echo "❌ $name failed to start within expected time"
    return 1
}

# Check ports
echo "🔍 Checking ports..."
if ! check_port 11434; then
    echo "  Ollama already running on port 11434"
else
    echo "  Starting Ollama..."
    ollama serve &
    OLLAMA_PID=$!
    sleep 3
fi

if ! check_port 8000; then
    echo "  Router already running on port 8000"
else
    echo "  Starting Cline Memory Router..."
    cd "$(dirname "$0")/.."
    python src/cline/memory_aware_router.py &
    ROUTER_PID=$!
    wait_for_service "http://localhost:8000/health" "Router"
fi

if ! check_port 3000; then
    echo "❌ Port 3000 is already in use. Please stop the service using it."
    exit 1
else
    echo "🌐 Starting Open WebUI..."
    cd config/openwebui
    open-webui serve --port 3000 --host 0.0.0.0 &
    WEBUI_PID=$!
    cd ../..
    wait_for_service "http://localhost:3000" "Open WebUI"
fi

echo ""
echo "🎉 System started successfully!"
echo ""
echo "🔗 Access Points:"
echo "   • Open WebUI:     http://localhost:3000"
echo "   • Router API:     http://localhost:8000"
echo "   • API Docs:       http://localhost:8000/docs"
echo "   • Ollama API:     http://localhost:11434"
echo ""
echo "💡 First time setup:"
echo "   1. Go to http://localhost:3000"
echo "   2. Create admin account"
echo "   3. Models should be auto-detected from your router"
echo ""
echo "🛑 To stop all services:"
echo "   Press Ctrl+C or run: ./scripts/stop_system.sh"
echo ""

# Wait for interrupt
trap 'echo ""; echo "🛑 Shutting down..."; kill $OLLAMA_PID $ROUTER_PID $WEBUI_PID 2>/dev/null || true; exit 0' INT

wait
EOF

# Create stop script
cat > scripts/stop_system.sh << 'EOF'
#!/bin/bash
# File: scripts/stop_system.sh
# Stop all system services

echo "🛑 Stopping Local LLM System..."

# Kill processes by port
echo "  Stopping Open WebUI (port 3000)..."
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

echo "  Stopping Router (port 8000)..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

echo "  Stopping Ollama (port 11434)..."
lsof -ti:11434 | xargs kill -9 2>/dev/null || true

# Alternative: kill by process name
pkill -f "open-webui" 2>/dev/null || true
pkill -f "memory_aware_router" 2>/dev/null || true
pkill -f "ollama serve" 2>/dev/null || true

echo "✅ All services stopped"
EOF

# Make scripts executable
chmod +x scripts/start_with_openwebui.sh
chmod +x scripts/stop_system.sh

echo ""
echo "🎉 Native Open WebUI setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Run: ./scripts/start_with_openwebui.sh"
echo "2. Open http://localhost:3000"
echo "3. Create admin account"
echo "4. Start chatting!"
echo ""
echo "📁 Configuration:"
echo "   • Open WebUI config: config/openwebui/.env"
echo "   • Data directory: data/openwebui/"
echo "   • Logs: Check terminal output"
echo ""
echo "🔧 Your router will automatically handle:"
echo "   • Intelligent model selection"
echo "   • Memory bank integration"  
echo "   • Context optimization"
echo "   • Performance improvements"