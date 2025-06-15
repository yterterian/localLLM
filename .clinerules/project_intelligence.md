# project_intelligence.md

This file captures project-specific patterns, preferences, and intelligence for the Cline memory-aware system.

## Critical Implementation Paths:

### Three-Model Router Implementation:
- **Router Core**: `src/cline/memory_aware_router.py` is the central intelligence hub
- **Pattern Matching**: Regex-based routing logic in `determine_cline_model()` method
- **Context Enhancement**: Memory bank content injection must happen before model calls
- **Session Tracking**: Workspace-based state management critical for continuity

### Memory Bank Integration Flow:
1. **Detection**: `detect_memory_bank_files()` scans `.clinerules/memory-bank/` directory
2. **Caching**: Memory bank content cached per workspace to avoid repeated file I/O
3. **Injection**: Content selectively injected based on model type and task context
4. **Updates**: Background refresh mechanism for memory bank changes

### Vector Store Management:
- **Interface Pattern**: Abstract `VectorStoreInterface` allows pluggable backends
- **Default Choice**: ChromaDB for development, with FAISS/Qdrant as alternatives
- **Lazy Loading**: Vector stores initialized only when needed per project
- **Context Expansion**: Semantic search results injected into model context

### Model-Specific Optimizations:
- **Qwen3 (Memory Fast)**: Optimized for quick memory bank reading with 40K context
- **Qwen2.5 Coder (Implementation)**: 128K context for large file operations
- **Llama 4 (Analysis)**: 200K+ context allocation for comprehensive analysis

## User Preferences and Workflow:

### Hardware Configuration:
- **RTX 5080 + 128GB RAM**: All three models can run concurrently
- **Python 3.13.5**: Latest Python features and performance optimizations preferred
- **Local-First Philosophy**: Complete privacy and control over all data and models

### Development Environment:
- **VSCode + Cline**: Primary development interface with memory bank integration
- **Conda Environment**: Isolated environment management preferred over venv
- **Professional Structure**: Proper project organization with scripts and configuration

### Workflow Patterns:
- **Session Start**: Always begin with "read memory bank" for context
- **Implementation Focus**: Prefer specialized coding model for complex tasks
- **Documentation Updates**: Regular memory bank updates to maintain project intelligence
- **Performance Monitoring**: Active interest in response times and resource utilization

### Communication Style:
- **Technical Depth**: Appreciates detailed technical explanations and architecture
- **Practical Focus**: Values working solutions over theoretical discussions
- **Optimization Mindset**: Consistently seeks performance improvements and efficiency
- **Future Planning**: Considers scalability and extension points (Home Assistant, agentic workflows)

## Project-Specific Patterns:

### Configuration Management Pattern:
```python
# YAML-based configuration with validation
config = yaml.safe_load(config_file)
model_name = config['models'][model_type]['name']
```

### Async-First Architecture:
```python
# All model calls use async/await pattern
async def call_ollama_with_context(self, model_key: str, request: ClineRequest):
    async with aiohttp.ClientSession() as session:
        # Async HTTP operations
```

### Enhanced Context Pattern:
```python
# Memory bank content injection for appropriate models
if model_key in ["memory_analysis", "memory_fast"]:
    memory_context = build_memory_context(workspace)
    enhanced_messages.append({"role": "system", "content": memory_context})
```

### Scoring-Based Model Selection:
```python
# Multi-factor scoring system for intelligent routing
scores = {"fast": 0, "coding": 0, "analysis": 0}
for pattern in patterns:
    matches = re.findall(pattern, message)
    scores[model_type] += len(matches) * weight
```

### Workspace Session Tracking:
```python
# Maintain state across requests per workspace
self.cline_sessions[workspace_path] = {
    "last_model": selected_model,
    "message_count": count,
    "memory_bank_available": bool(memory_content)
}
```

## Known Challenges:

### Memory Management:
- **Challenge**: Three large models consuming 100GB+ RAM simultaneously
- **Solution**: Careful loading order and memory monitoring
- **Mitigation**: Graceful degradation if models fail to load

### Context Window Optimization:
- **Challenge**: Balancing context size vs. response time for each model
- **Solution**: Model-specific context limits in configuration
- **Ongoing**: Fine-tuning context allocation based on usage patterns

### Routing Accuracy:
- **Challenge**: Ensuring correct model selection for ambiguous requests
- **Solution**: Multi-factor scoring with pattern matching and context analysis
- **Improvement**: Learning from user corrections and session patterns

### Vector Store Performance:
- **Challenge**: Large codebase indexing can be slow
- **Solution**: Background indexing and incremental updates
- **Optimization**: Choice between speed (FAISS) vs. features (ChromaDB)

### Cline Integration Complexity:
- **Challenge**: Memory bank detection across different project structures
- **Solution**: Robust file scanning with fallback mechanisms
- **Evolution**: Support for custom memory bank organizations

## Evolution of Project Decisions:

### Model Selection Evolution:
1. **Initial**: Single model approach considered insufficient
2. **Iteration 1**: Two-model system (fast + coding) proposed
3. **Final Decision**: Three-model system with specialized analysis model
4. **Rationale**: Need for comprehensive analysis capabilities and memory bank updates

### Vector Store Choice Evolution:
1. **Initial**: FAISS-only for maximum speed
2. **Consideration**: ChromaDB for ease of use
3. **Final Architecture**: Pluggable system supporting multiple backends
4. **Rationale**: Different use cases benefit from different vector store characteristics

### API Design Evolution:
1. **Initial**: Custom API format
2. **Pivot**: OpenAI-compatible API for tool integration
3. **Enhancement**: Additional metadata for debugging and optimization
4. **Rationale**: Compatibility with existing tools while providing enhanced functionality

### Context Management Evolution:
1. **Initial**: Simple context injection
2. **Enhancement**: Memory bank aware context building
3. **Optimization**: Model-specific context strategies
4. **Current**: Intelligent context allocation based on task type

## Tool Usage Patterns:

### Development Workflow:
- **Primary Interface**: VSCode with Cline extension
- **Model Management**: Ollama for local model execution
- **Environment**: Conda for dependency isolation
- **Configuration**: YAML files for settings management
- **Monitoring**: curl commands for health checks and debugging

### Debugging and Monitoring:
```bash
# Health check pattern
curl http://localhost:8000/health

# Memory bank status check
curl http://localhost:8000/cline/memory-bank-status/$(pwd)

# Model statistics
curl http://localhost:8000/cline/model-stats
```

### Memory Bank Maintenance:
- **Regular Updates**: "update memory bank" trigger for comprehensive documentation
- **Incremental Changes**: Background context refresh for file changes
- **Validation**: Memory bank status endpoints for troubleshooting
- **Optimization**: Caching strategy to minimize file I/O

### Performance Optimization:
- **Resource Monitoring**: nvidia-smi and htop for hardware utilization
- **Response Time Tracking**: Structured logging for performance analysis
- **Model Switching**: Intelligent routing based on task complexity
- **Context Tuning**: Adjusting context windows based on performance metrics

### Integration Testing:
- **Cline Workflow**: Regular testing of memory bank integration
- **Model Routing**: Verification of appropriate model selection
- **API Compatibility**: Testing with various OpenAI-compatible tools
- **Vector Store Performance**: Benchmarking different backends

## Learned Patterns:

### Cline Memory Bank Triggers:
- **"read memory bank"** → Always routes to memory_fast model
- **"update memory bank"** → Always routes to memory_analysis model
- **Code implementation keywords** → Routes to implementation model
- **Architecture/planning keywords** → Routes to memory_analysis model

### Effective Context Injection:
- **Memory bank content** should be injected for analysis and fast models
- **File context** (open files, current file) useful for implementation model
- **Session continuity** important for maintaining conversation context
- **Workspace detection** critical for proper memory bank association

### Performance Optimization Insights:
- **Model preloading** reduces first-request latency
- **Context caching** significantly improves repeated requests
- **Background indexing** keeps vector stores current without blocking requests
- **Async architecture** essential for handling multiple concurrent operations

## Notes:
- System designed around Cline's sophisticated memory bank workflow
- Architecture emphasizes local control and privacy while matching cloud capabilities
- Performance targets are aggressive but achievable with proper hardware utilization
- Future extensions (Home Assistant, agentic workflows) considered in design
- Project serves as foundation for broader AI-assisted development tool ecosystem
- Documentation patterns established here should be maintained as system evolves
- User feedback and usage patterns will drive future optimization priorities