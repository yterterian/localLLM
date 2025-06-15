#!/bin/bash
set -e

echo "🚀 Setting up Local LLM System..."

# Create directories
mkdir -p logs vector_stores data/projects

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Copy environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Created .env file - please review and modify as needed"
fi

# Install pre-commit hooks
pre-commit install

echo "✅ Setup complete!"
echo "Next steps:"
echo "1. Review and update .env file"
echo "2. Run 'scripts/install_models.sh' to download models"
echo "3. Run 'scripts/run_system.sh' to start the system"
