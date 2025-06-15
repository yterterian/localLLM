# systemPatterns.md

This file documents system architecture, key technical decisions, and design patterns in use.

## System Architecture:

### Three-Tier Model Architecture:
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Memory Fast    │    │  Implementation  │    │ Memory Analysis │
│  Qwen3 30B-A3B  │    │ Qwen2.5 Coder 32B│    │ Llama4 16x17B   │
│                 │    │                  │    │                 │
│ • Memory bank   │    │ • Complex coding │    │ • Architecture  │
│ • Quick queries │    │ • Debugging      │    │ • Planning      │
│ • Explanations  │    │ • Refactoring    │    │ • Analysis      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │ Intelligent Router  │
                    │                     │
                    │ • Pattern matching  │
                    │ • Context analysis  │
                    │ • Model selection   │
                    │ • Session tracking  │
                    └─────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │   FastAPI Server    │
                    │                     │
                    │ • OpenAI compat API │
                    │ • Health monitoring │
                    │ • Session management│
                    └─────────────────────┘
```

### Context Expansion Layer:
```
┌─────────────────────────────────────────────────────────────┐
│                    Vector Store Manager                     │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Chroma    │  │    FAISS    │  │   Qdrant    │         │
│  │ (Default)   │  │ (Speed)     │  │ (Production)│         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  • Codebase indexing    • Semantic search                  │
│  • Context injection    • Memory bank integration          │
└─────────────────────────────────────────────────────────────┘
```

## Key Technical Decisions:

### Model Selection Strategy:
- **MoE Architecture Choice**: Qwen3 30B-A3B chosen for efficiency (3B active parameters, 30B total)
- **Specialized Coding Model**: Qwen2.5 Coder 32B for superior coding performance vs general models
- **Massive Context Model**: Llama 4 16x17B for comprehensive analysis with 1M token context
- **Complementary Capabilities**: Each model optimized for specific workflow phases

### Routing Logic Architecture:
- **Pattern-Based Routing**: Regex and keyword matching for reliable task classification
- **Context-Aware Scoring**: Multi-factor scoring system considering message content, file context, session history
- **Memory Bank Integration**: Automatic detection and injection of Cline memory bank content
- **Session Continuity**: Workspace-based session tracking for consistent experience

### API Design Pattern:
- **OpenAI Compatibility**: Standard OpenAI API format for maximum tool compatibility
- **Enhanced Metadata**: Additional routing information for debugging and optimization
- **Async Architecture**: Full async/await pattern for optimal performance under load
- **Error Handling**: Comprehensive retry logic and graceful degradation

### Configuration Management:
- **YAML-Based Config**: Centralized configuration for models, routing rules, vector stores
- **Environment Separation**: Clear separation between development and runtime configuration
- **Hot Reloading**: Configuration changes without service restart (where safe)
- **Validation**: Schema validation for all configuration inputs

## Design Patterns:

### Router Pattern:
```python
class IntelligentRouter:
    def determine_model(self, request) -> (model_type, reasoning):
        # Pattern matching → Scoring → Selection
        scores = self.score_all_models(request)
        return self.select_optimal_model(scores)
```

### Strategy Pattern for Vector Stores:
```python
class VectorStoreInterface(ABC):
    @abstractmethod
    def similarity_search(self, query: str) -> List[Document]:
        pass

class ChromaStore(VectorStoreInterface):
    # Chroma-specific implementation
    
class FAISSStore(VectorStoreInterface):
    # FAISS-specific implementation
```

### Factory Pattern for Model Creation:
```python
class ModelFactory:
    @staticmethod
    def create_router(config_path: str) -> Router:
        config = load_config(config_path)
        return Router(config)
```

### Observer Pattern for Session Tracking:
```python
class SessionTracker:
    def notify_request(self, workspace: str, model: str, reasoning: str):
        self.update_session_state(workspace, model, reasoning)
        self.update_metrics(model)
```

### Dependency Injection for Services:
```python
# Services injected rather than hardcoded
def create_app(router: Router, vector_manager: VectorManager) -> FastAPI:
    app = FastAPI()
    app.router = router
    app.vector_manager = vector_manager
    return app
```

## Component Relationships:

### Core Components:
- **Router**: Central intelligence for model selection and request routing
- **Model Clients**: Async HTTP clients for each Ollama model
- **Vector Store Manager**: Handles codebase indexing and context expansion
- **Memory Bank Reader**: Cline-specific memory bank file detection and parsing
- **Session Manager**: Tracks workspace state and user patterns

### Data Flow:
1. **Request Reception**: FastAPI receives Cline request with context
2. **Memory Bank Detection**: Automatic detection of .clinerules and memory bank files
3. **Model Selection**: Intelligent routing based on patterns and context
4. **Context Enhancement**: Memory bank and vector store content injection
5. **Model Execution**: Optimized call to selected Ollama model
6. **Response Processing**: Format response and update session tracking

### Integration Points:
- **Cline Integration**: Memory bank file reading and workspace detection
- **VSCode Integration**: Task definitions and health check endpoints
- **Ollama Integration**: Model management and execution
- **Vector Store Integration**: Pluggable storage backends for context expansion

## Architecture Patterns:

### Microservice-Ready Design:
- Clear separation of concerns between routing, execution, and storage
- Service interfaces that could be split into separate processes
- Configuration-driven service discovery and connection

### Event-Driven Updates:
- Background tasks for vector store updates
- Async processing for non-critical operations
- Webhook support for external integration triggers

### Extensible Plugin Architecture:
- New vector stores can be added via interface implementation
- Custom routing rules can be injected via configuration
- Model clients follow consistent interface for easy swapping

## Performance Patterns:

### Caching Strategy:
- Memory bank content cached per workspace
- Vector store indices persisted to disk
- Session state maintained in memory with periodic persistence

### Resource Management:
- Connection pooling for HTTP clients
- Lazy loading of vector stores
- Memory-mapped file access for large indices

### Monitoring and Observability:
- Structured logging throughout system
- Performance metrics collection
- Health check endpoints for all components

## Notes:
- Architecture designed for single-machine deployment initially but ready for distributed deployment
- All patterns chosen for maintainability and extensibility
- Performance patterns optimized for RTX 5080 + 128GB RAM configuration
- Component interfaces designed for testing and mocking
- Clear separation between Cline-specific logic and general LLM routing capabilities