# techContext.md

This file documents the technologies used, development setup, technical constraints, and dependencies.

## Technologies Used:

### Core Runtime:
- **Python 3.13.5**: Latest Python with performance improvements and modern syntax support
- **FastAPI 0.115.0+**: Modern async web framework for high-performance API server
- **Uvicorn 0.30.0+**: ASGI server for production-ready FastAPI deployment
- **Pydantic 2.9.0+**: Data validation and settings management with type hints

### LLM Integration:
- **Ollama**: Local LLM model management and execution platform
- **aiohttp 3.10.0+**: Async HTTP client for Ollama API communication
- **OpenAI-compatible API**: Standard interface for maximum tool compatibility

### AI/ML Libraries:
- **LangChain 0.3.0+**: Framework for LLM application development and chaining
- **sentence-transformers 3.0.0+**: Local embedding models for vector operations
- **torch 2.4.0+**: PyTorch for deep learning model support

### Vector Stores:
- **ChromaDB 0.5.0+**: Primary vector database for development (embedded)
- **FAISS 1.8.0+**: Facebook AI Similarity Search for high-performance vector operations
- **Qdrant-client 1.11.0+**: Python client for Qdrant vector database (production option)
- **tiktoken 0.8.0+**: OpenAI's tokenizer for accurate token counting

### Development Tools:
- **structlog 24.4.0+**: Structured logging for production observability
- **rich 13.8.0+**: Beautiful terminal output and debugging
- **pytest 7.4.0+**: Testing framework with async support
- **black 23.9.0+**: Code formatting
- **isort 5.12.0+**: Import sorting
- **mypy 1.6.0+**: Static type checking

### Models (via Ollama):
- **Qwen3 30B-A3B**: MoE model for fast routing and memory bank operations
- **Qwen2.5 Coder 32B**: Specialized coding model with 128K context
- **Llama 4 16x17B**: Multimodal analysis model with 1M context capability

## Development Setup:

### Environment Requirements:
```bash
# Hardware Requirements
- GPU: RTX 5080 (16GB VRAM minimum)
- RAM: 128GB DDR5-6400 (for concurrent model loading)
- Storage: 500GB+ SSD (for models and vector stores)
- CPU: Modern multi-core processor (for embedding calculations)

# Software Requirements
- Windows 11 / macOS 13+ / Ubuntu 22.04+
- Python 3.13.5
- Git 2.40+
- Docker (optional, for Qdrant)
- VSCode with Cline extension
```

### Installation Process:
```bash
# 1. Create project environment
conda create -n local-llm python=3.13.5
conda activate local-llm

# 2. Clone and setup project
git clone <repository>
cd local-llm-system
chmod +x scripts/*.sh
./scripts/setup.sh

# 3. Install Ollama and models
# Download Ollama from https://ollama.com
ollama pull qwen3:30b-a3b
ollama pull qwen2.5-coder:32b
ollama pull llama4:16x17b

# 4. Start the system
./scripts/run_system.sh
```

### Development Workflow:
```bash
# Start development environment
conda activate local-llm
ollama serve  # In background
python src/cline/memory_aware_router.py

# Run tests
pytest tests/

# Code quality checks
black src/
isort src/
mypy src/
```

### VSCode Configuration:
```json
{
  "python.defaultInterpreterPath": "./local-llm-env/bin/python",
  "python.terminal.activateEnvironment": true,
  "cline.apiProvider": "openai-compatible",
  "cline.openaiCompatible": {
    "baseUrl": "http://localhost:8000/v1",
    "modelName": "cline-memory-router"
  }
}
```

## Technical Constraints:

### Hardware Constraints:
- **VRAM Limitation**: RTX 5080 16GB VRAM requires careful model loading strategy
- **RAM Management**: 128GB allows concurrent model loading but requires monitoring
- **Storage I/O**: Vector store operations can be I/O intensive, SSD recommended
- **GPU Compute**: Single GPU limits to sequential model execution (planned optimization)

### Software Constraints:
- **Python 3.13.5 Compatibility**: Some libraries may lag Python version support
- **Ollama Dependency**: System tied to Ollama's model management and API
- **Local Network Only**: Designed for localhost deployment, not distributed
- **Single User**: Current architecture supports one concurrent user per instance

### Model Constraints:
- **Context Windows**: Models have fixed context limits (40K, 128K, 1M respectively)
- **Response Time**: Larger models trade speed for capability
- **Memory Requirements**: All models must fit within available system memory
- **Token Limits**: Output token limits affect response comprehensiveness

### Integration Constraints:
- **Cline Dependency**: Optimized specifically for Cline workflow patterns
- **VSCode Integration**: Best experience requires VSCode + Cline extension
- **Memory Bank Format**: Tied to Cline's specific memory bank file structure
- **API Compatibility**: Must maintain OpenAI API compatibility for tool integration

## Dependencies:

### Core Dependencies:
```yaml
fastapi: ">=0.115.0"        # Web framework
uvicorn: ">=0.30.0"         # ASGI server
pydantic: ">=2.9.0"         # Data validation
aiohttp: ">=3.10.0"         # Async HTTP client
pyyaml: ">=6.0.2"           # Configuration files
structlog: ">=24.4.0"       # Structured logging
```

### AI/ML Dependencies:
```yaml
langchain: ">=0.3.0"                    # LLM framework
langchain-community: ">=0.3.0"         # Community integrations
sentence-transformers: ">=3.0.0"       # Embeddings
torch: ">=2.4.0"                       # Deep learning
transformers: ">=4.45.0"               # Model support
tiktoken: ">=0.8.0"                    # Tokenization
```

### Vector Store Dependencies:
```yaml
chromadb: ">=0.5.0"         # Primary vector store
faiss-cpu: ">=1.8.0"        # High-performance alternative
qdrant-client: ">=1.11.0"   # Production vector store
```

### Development Dependencies:
```yaml
pytest: ">=7.4.0"           # Testing
pytest-asyncio: ">=0.21.0"  # Async testing
black: ">=23.9.0"           # Code formatting
isort: ">=5.12.0"           # Import sorting
mypy: ">=1.6.0"             # Type checking
rich: ">=13.8.0"            # Terminal output
```

### External Dependencies:
- **Ollama**: Model management and execution platform
- **NVIDIA Drivers**: GPU support for model execution
- **CUDA 12.1+**: GPU acceleration (included with Ollama)

## Configuration Management:

### Environment Variables:
```bash
PYTHONPATH=./src              # Python path for imports
OLLAMA_BASE_URL=http://localhost:11434  # Ollama API endpoint
LOG_LEVEL=INFO                # Logging level
VECTOR_STORE_PATH=./vector_stores        # Vector store persistence
MODEL_CACHE_DIR=./models      # Model cache directory
```

### Configuration Files:
- `config/models.yaml`: Model definitions and routing rules
- `config/server_config.yaml`: Server and performance settings
- `.env`: Environment-specific variables
- `requirements.txt`: Python package dependencies
- `pyproject.toml`: Project metadata and tool configuration

### Resource Allocation:
```yaml
# Approximate memory allocation
qwen3_30b_a3b: 20GB          # 3B active parameters
qwen2_5_coder_32b: 35GB      # 32B parameters
llama4_16x17b: 45GB          # 16x17B MoE
system_overhead: 15GB        # OS, Python, FastAPI
vector_stores: 10GB          # Embeddings and indices
available_buffer: 3GB        # Safety margin
```

## Performance Characteristics:

### Expected Response Times:
- **Memory Fast (Qwen3)**: 2-5 seconds for memory bank reading
- **Implementation (Qwen2.5 Coder)**: 15-45 seconds for complex coding
- **Memory Analysis (Llama 4)**: 60-180 seconds for comprehensive analysis

### Throughput Limitations:
- **Single Request Processing**: One model call at a time per model
- **Concurrent Capabilities**: Multiple models can run simultaneously
- **Vector Store QPS**: 10-50 queries per second depending on store type

### Resource Monitoring:
- **GPU Memory**: Monitor via `nvidia-smi`
- **System Memory**: Monitor via `htop` or Task Manager
- **API Health**: Built-in health check endpoints
- **Performance Metrics**: Structured logging with timing information

## Notes:
- All dependencies pinned to specific versions for reproducibility
- Development environment isolated via conda/venv
- Production deployment assumes same hardware configuration
- Vector store choice impacts performance characteristics significantly
- Model loading order optimized for memory efficiency
- Configuration designed for easy scaling to distributed deployment
- All external API calls (Ollama) include retry logic and timeouts
- System designed to gracefully degrade if models unavailable