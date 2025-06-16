import asyncio
import aiohttp
import json
import re
import yaml
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import structlog
from datetime import datetime, timedelta
import hashlib
from collections import defaultdict

logger = structlog.get_logger()

class ClineMessage(BaseModel):
    role: str
    content: str

class ClineRequest(BaseModel):
    model: str
    messages: List[ClineMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    # Cline-specific fields
    workspace_path: Optional[str] = None
    current_file: Optional[str] = None
    open_files: Optional[List[str]] = []
    recent_changes: Optional[List[str]] = []
    task_context: Optional[str] = None

class ClineMemoryAwareRouter:
    def __init__(self, config_path: str = "config/models.yaml"):
        """Router optimized for Cline's memory bank workflow"""
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.ollama_base_url = self.config['ollama']['base_url']
        
        # Model assignments optimized for Cline memory bank patterns
        self.models = {
            # Fast model for memory bank reading, quick explanations, simple edits
            "memory_fast": self.config['models']['small']['name'],      # qwen3:30b-a3b
            
            # Coding specialist for implementation, debugging, complex coding
            "implementation": self.config['models']['large']['name'],    # qwen2.5-coder:32b
            
            # Analysis model for memory bank updates, architecture, complex analysis
            "memory_analysis": self.config['models']['multimodal']['name']  # llama4:16x17b
        }
        
        # Cline memory bank workflow patterns
        self.cline_memory_patterns = {
            "memory_fast": {
                "triggers": [
                    # Memory bank reading and understanding
                    "read memory bank", "check memory bank", "review memory bank",
                    "what's in the memory bank", "current context", "project status",
                    
                    # Quick explanations and simple questions
                    "explain this", "what does", "how does", "quick question",
                    "syntax", "simple fix", "rename", "typo",
                    
                    # File navigation and understanding
                    "show me", "find", "locate", "where is", "what's in"
                ],
                "patterns": [
                    r"\bread\s+(memory\s+bank|\.clinerules)\b",
                    r"\bcheck\s+(memory\s+bank|context|status)\b",
                    r"\bwhat('s|\s+is)\s+(in\s+the\s+)?(memory\s+bank|\.clinerules|current\s+context)\b",
                    r"\bexplain\s+(this|that)\s+(line|function|file|variable)\b",
                    r"\bquick\s+(question|help|fix|explanation)\b",
                    r"\b(simple|basic|easy)\s+(fix|change|update)\b"
                ],
                "memory_bank_files": [
                    "projectbrief.md", "productContext.md", "activeContext.md",
                    "systemPatterns.md", "techContext.md", "progress.md"
                ]
            },
            
            "implementation": {
                "triggers": [
                    # Implementation tasks
                    "implement", "create", "build", "develop", "code", "write",
                    "add feature", "new component", "new function",
                    
                    # Debugging and fixing
                    "debug", "fix", "solve", "error", "bug", "issue", "problem",
                    "not working", "failing", "broken",
                    
                    # Refactoring and optimization
                    "refactor", "optimize", "improve", "clean up", "reorganize",
                    
                    # Testing
                    "test", "unit test", "integration test", "testing"
                ],
                "patterns": [
                    r"\b(implement|create|build|develop|write)\s+.*(function|class|component|feature|module)\b",
                    r"\b(debug|fix|solve)\s+.*(bug|error|issue|problem)\b",
                    r"\b(refactor|optimize|improve|enhance)\b",
                    r"\b(test|testing|unit\s+test|integration\s+test)\b",
                    r"```[\s\S]*```",  # Code blocks
                    r"\b(add|modify|update|change)\s+.*(file|function|class|component)\b",
                    r"\b(api|endpoint|route|database|sql)\b"
                ],
                "excludes": [
                    r"\bmemory\s+bank\b", r"\b\.clinerules\b", r"\bplan\s+mode\b"
                ]
            },
            
            "memory_analysis": {
                "triggers": [
                    # Memory bank updates and maintenance
                    "update memory bank", "update .clinerules", "document this",
                    "add to memory bank", "memory bank update",
                    
                    # Planning and architecture
                    "plan", "strategy", "approach", "architecture", "design",
                    "overall", "comprehensive", "complete",
                    
                    # Analysis and review
                    "analyze", "review", "evaluate", "assess", "examine",
                    "compare", "best practices", "patterns",
                    
                    # Project-wide operations
                    "entire project", "whole codebase", "all files", "project structure"
                ],
                "patterns": [
                    r"\bupdate\s+(memory\s+bank|\.clinerules)\b",
                    r"\b(plan|strategy|approach|architecture|design)\b",
                    r"\b(analyze|review|evaluate|assess|examine)\s+.*(project|codebase|system|architecture)\b",
                    r"\b(overall|comprehensive|complete|entire|whole)\s+.*(review|analysis|plan|design)\b",
                    r"\b(memory\s+bank|\.clinerules)\s+.*(update|add|document|maintain)\b",
                    r"\bplan\s+mode\b",
                    r"\bdocument\s+.*(pattern|decision|change|progress)\b"
                ],
                "memory_operations": [
                    "create", "update", "review", "maintain", "document"
                ]
            }
        }
        
        # Track Cline sessions with memory bank awareness
        self.cline_sessions = {}  # workspace_path -> session_info
        self.memory_bank_cache = {}  # workspace_path -> memory_bank_content
        
        logger.info("Cline memory-aware router initialized", models=self.models)

    def detect_memory_bank_files(self, workspace_path: str) -> Dict[str, str]:
        """Detect and read Cline memory bank files"""
        
        memory_bank_content = {}
        
        if not workspace_path:
            return memory_bank_content
        
        workspace = Path(workspace_path)
        memory_bank_dir = workspace / ".clinerules" / "memory-bank"
        clinerules_file = workspace / ".clinerules"
        
        # Core memory bank files
        core_files = [
            "projectbrief.md", "productContext.md", "activeContext.md",
            "systemPatterns.md", "techContext.md", "progress.md"
        ]
        
        # Read memory bank files
        if memory_bank_dir.exists():
            for file_name in core_files:
                file_path = memory_bank_dir / file_name
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            memory_bank_content[file_name] = f.read()
                    except Exception as e:
                        logger.warning(f"Failed to read {file_name}", error=str(e))
        
        # Read .clinerules file
        if clinerules_file.exists():
            try:
                with open(clinerules_file, 'r', encoding='utf-8') as f:
                    memory_bank_content[".clinerules"] = f.read()
            except Exception as e:
                logger.warning("Failed to read .clinerules", error=str(e))
        
        return memory_bank_content

    def determine_cline_model(self, request: ClineRequest) -> Tuple[str, str]:
        """Determine best model for Cline workflow with memory bank awareness"""
        
        # Get the latest user message
        user_messages = [msg.content.lower() for msg in request.messages if msg.role == "user"]
        if not user_messages:
            return "memory_fast", "no_user_messages"
        
        latest_message = user_messages[-1]
        
        # Check for memory bank content in workspace
        memory_bank_available = False
        if request.workspace_path:
            memory_bank_content = self.detect_memory_bank_files(request.workspace_path)
            memory_bank_available = bool(memory_bank_content)
            self.memory_bank_cache[request.workspace_path] = memory_bank_content
        
        # Scoring system with memory bank awareness
        scores = {"memory_fast": 0, "implementation": 0, "memory_analysis": 0}
        
        # 1. Explicit memory bank operations (highest priority)
        memory_triggers = [
            "update memory bank", "read memory bank", "check memory bank",
            "review memory bank", ".clinerules", "plan mode"
        ]
        
        for trigger in memory_triggers:
            if trigger in latest_message:
                if "update" in trigger or "plan mode" in trigger:
                    scores["memory_analysis"] += 15  # Major memory bank operations
                else:
                    scores["memory_fast"] += 10     # Reading memory bank
        
        # 2. Pattern matching for each model type
        for model_type, pattern_config in self.cline_memory_patterns.items():
            # Trigger word matching
            for trigger in pattern_config["triggers"]:
                if trigger in latest_message:
                    weight = 3 if model_type == "memory_analysis" else 2
                    scores[model_type] += weight
            
            # Regex pattern matching
            for pattern in pattern_config["patterns"]:
                matches = len(re.findall(pattern, latest_message, re.IGNORECASE))
                if matches > 0:
                    weight = 4 if model_type == "memory_analysis" else 3
                    scores[model_type] += matches * weight
            
            # Exclude patterns for implementation model
            if model_type == "implementation" and "excludes" in pattern_config:
                for exclude_pattern in pattern_config["excludes"]:
                    if re.search(exclude_pattern, latest_message, re.IGNORECASE):
                        scores[model_type] = max(0, scores[model_type] - 5)
        
        # 3. Context-based adjustments
        
        # Task context override
        if request.task_context:
            context_mapping = {
                "explain": "memory_fast",
                "implement": "implementation",
                "debug": "implementation",
                "refactor": "implementation",
                "review": "memory_analysis",
                "analyze": "memory_analysis",
                "plan": "memory_analysis"
            }
            if request.task_context.lower() in context_mapping:
                target_model = context_mapping[request.task_context.lower()]
                scores[target_model] += 8
        
        # File context analysis
        if request.current_file:
            file_path = Path(request.current_file)
            
            # Memory bank files
            if ".clinerules" in str(file_path) or "memory-bank" in str(file_path):
                scores["memory_analysis"] += 5
            
            # Complex implementation files
            elif file_path.suffix.lower() in ['.py', '.js', '.ts', '.java', '.cpp', '.rs']:
                scores["implementation"] += 2
        
        # Multi-file operations
        if request.open_files:
            if len(request.open_files) > 10:
                scores["memory_analysis"] += 4  # Large scale analysis
            elif len(request.open_files) > 3:
                scores["implementation"] += 2   # Multi-file implementation
        
        # Message complexity
        message_length = len(latest_message)
        if message_length > 1000:
            scores["memory_analysis"] += 3
        elif message_length > 300:
            scores["implementation"] += 2
        else:
            scores["memory_fast"] += 1
        
        # Code block detection
        code_blocks = len(re.findall(r'```[\s\S]*?```', latest_message))
        if code_blocks > 0:
            scores["implementation"] += code_blocks * 3
        
        # Session continuity
        if request.workspace_path in self.cline_sessions:
            session = self.cline_sessions[request.workspace_path]
            last_model = session.get("last_model")
            
            # Memory bank session continuity
            if last_model == "memory_analysis" and memory_bank_available:
                scores["memory_analysis"] += 2
            elif last_model == "implementation":
                scores["implementation"] += 1
        
        # Determine winner
        selected_model = max(scores, key=scores.get)
        max_score = scores[selected_model]
        
        # Default handling
        if max_score == 0:
            if memory_bank_available:
                selected_model = "memory_fast"
                reasoning = "default_with_memory_bank"
            else:
                selected_model = "memory_fast"
                reasoning = "default_no_patterns"
        else:
            reasoning = f"score_{max_score}_memory_{memory_bank_available}"
        
        # Update session tracking
        if request.workspace_path:
            if request.workspace_path not in self.cline_sessions:
                self.cline_sessions[request.workspace_path] = {}
            
            self.cline_sessions[request.workspace_path].update({
                "last_model": selected_model,
                "last_request": datetime.now(),
                "message_count": self.cline_sessions[request.workspace_path].get("message_count", 0) + 1,
                "memory_bank_available": memory_bank_available
            })
        
        logger.info("Cline model selection",
                   selected=selected_model,
                   scores=scores,
                   reasoning=reasoning,
                   memory_bank_available=memory_bank_available,
                   workspace=request.workspace_path)
        
        return selected_model, reasoning

    async def call_ollama_with_memory_context(self, model_key: str, request: ClineRequest) -> Dict:
        """Call Ollama with Cline memory bank context enhancement"""
        
        model_name = self.models[model_key]
        
        # Build enhanced context with memory bank awareness
        enhanced_messages = []
        
        # Add memory bank context for appropriate models
        if model_key in ["memory_analysis", "memory_fast"] and request.workspace_path:
            memory_bank_content = self.memory_bank_cache.get(request.workspace_path, {})
            
            if memory_bank_content:
                context_parts = ["CLINE MEMORY BANK CONTEXT:"]
                
                # Prioritize core memory bank files
                priority_files = ["activeContext.md", "progress.md", "projectbrief.md"]
                
                for file_name in priority_files:
                    if file_name in memory_bank_content:
                        content = memory_bank_content[file_name][:2000]  # Limit size
                        context_parts.append(f"\n=== {file_name} ===\n{content}")
                
                # Add .clinerules if available
                if ".clinerules" in memory_bank_content:
                    clinerules_content = memory_bank_content[".clinerules"][:1000]
                    context_parts.append(f"\n=== .clinerules ===\n{clinerules_content}")
                
                # Add project context
                if request.current_file:
                    context_parts.append(f"\nCURRENT FILE: {request.current_file}")
                
                if request.open_files:
                    context_parts.append(f"OPEN FILES: {', '.join(request.open_files[:5])}")
                
                memory_context = "\n".join(context_parts)
                enhanced_messages.append({"role": "system", "content": memory_context})
        
        # Add regular context for implementation model
        elif model_key == "implementation" and request.workspace_path:
            context_parts = []
            
            if request.current_file:
                context_parts.append(f"Working on: {request.current_file}")
            
            if request.open_files:
                context_parts.append(f"Open files: {', '.join(request.open_files[:5])}")
            
            if request.recent_changes:
                context_parts.append(f"Recent changes: {', '.join(request.recent_changes[:3])}")
            
            if request.task_context:
                context_parts.append(f"Task: {request.task_context}")
            
            if context_parts:
                impl_context = "IMPLEMENTATION CONTEXT: " + " | ".join(context_parts)
                enhanced_messages.append({"role": "system", "content": impl_context})
        
        # Add original messages
        for msg in request.messages:
            enhanced_messages.append({"role": msg.role, "content": msg.content})
        
        # Model-specific settings optimized for Cline workflows
        model_configs = {
            "memory_fast": {
                "temperature": 0.3,      # More focused for memory bank reading
                "num_ctx": 40000,        # Qwen3's full context for memory bank
                "num_predict": 3000      # Reasonable response length
            },
            "implementation": {
                "temperature": 0.1,      # Precise for coding
                "num_ctx": 131072,       # Qwen2.5 Coder's full 128K context
                "num_predict": 8000      # Longer responses for implementation
            },
            "memory_analysis": {
                "temperature": 0.2,      # Balanced for analysis
                "num_ctx": 200000,       # Use Llama 4's massive context
                "num_predict": 12000     # Comprehensive responses
            }
        }
        
        config = model_configs[model_key]
        temperature = request.temperature or config["temperature"]
        
        payload = {
            "model": model_name,
            "messages": enhanced_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": config["num_ctx"],
                "num_predict": config["num_predict"]
            }
        }
        
        if request.max_tokens:
            payload["options"]["num_predict"] = min(request.max_tokens, config["num_predict"])
        
        start_time = datetime.now()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.ollama_base_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=600)  # Longer timeout for complex analysis
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    duration = (datetime.now() - start_time).total_seconds()
                    
                    logger.info("Cline memory-aware call successful",
                               model=model_name,
                               model_key=model_key,
                               duration=duration,
                               workspace=request.workspace_path,
                               memory_bank_used=bool(request.workspace_path and 
                                                   request.workspace_path in self.memory_bank_cache))
                    
                    return result
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=error_text)

# FastAPI app optimized for Cline memory bank workflow
app = FastAPI(
    title="Cline Memory Bank Router",
    description="Intelligent routing optimized for Cline's memory bank workflow",
    version="3.0.0"
)

router = ClineMemoryAwareRouter()

@app.post("/v1/chat/completions")
async def cline_memory_chat_completions(request: ClineRequest):
    """Cline memory bank optimized chat completions"""

    # Error handling: invalid model
    if request.model not in router.models.values():
        raise HTTPException(status_code=400, detail=f"Invalid model: {request.model}")

    # Error handling: empty or malformed messages
    if not request.messages or not any(msg.content.strip() for msg in request.messages):
        raise HTTPException(status_code=400, detail="Empty or missing messages")

    # Error handling: at least one user message required
    if not any(msg.role == "user" and msg.content.strip() for msg in request.messages):
        raise HTTPException(status_code=400, detail="No user message found in messages")

    start_time = datetime.now()

    # Determine optimal model for this Cline request
    model_key, reasoning = router.determine_cline_model(request)
    selected_model = router.models[model_key]

    try:
        result = await router.call_ollama_with_memory_context(model_key, request)
        duration = (datetime.now() - start_time).total_seconds()

        # Enhanced response with Cline memory bank info
        response = {
            "id": f"cline-memory-{int(datetime.now().timestamp())}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": selected_model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["message"]["content"]
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": result.get("prompt_eval_count", 0),
                "completion_tokens": result.get("eval_count", 0),
                "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
            },
            "cline_memory_info": {
                "model_type": model_key,
                "selected_model": selected_model,
                "reasoning": reasoning,
                "duration": duration,
                "workspace": request.workspace_path,
                "memory_bank_available": bool(request.workspace_path and 
                                            request.workspace_path in router.memory_bank_cache),
                "memory_bank_files": list(router.memory_bank_cache.get(request.workspace_path, {}).keys())
            }
        }

        return response

    except Exception as e:
        logger.error("Cline memory request failed", error=str(e), workspace=request.workspace_path)
        raise

@app.post("/cline/refresh-memory-bank")
async def refresh_memory_bank(workspace_path: str):
    """Refresh memory bank cache for a workspace"""
    
    memory_bank_content = router.detect_memory_bank_files(workspace_path)
    router.memory_bank_cache[workspace_path] = memory_bank_content
    
    return {
        "status": "memory_bank_refreshed",
        "workspace": workspace_path,
        "files_found": list(memory_bank_content.keys()),
        "total_content_size": sum(len(content) for content in memory_bank_content.values())
    }

@app.get("/cline/memory-bank-status/{workspace_path:path}")
async def get_memory_bank_status(workspace_path: str):
    """Get memory bank status for a workspace"""
    
    memory_bank_content = router.memory_bank_cache.get(workspace_path, {})
    
    if not memory_bank_content:
        # Try to detect files
        memory_bank_content = router.detect_memory_bank_files(workspace_path)
        router.memory_bank_cache[workspace_path] = memory_bank_content
    
    return {
        "workspace": workspace_path,
        "memory_bank_available": bool(memory_bank_content),
        "files": list(memory_bank_content.keys()),
        "core_files_present": [
            f for f in ["projectbrief.md", "productContext.md", "activeContext.md", 
                       "systemPatterns.md", "techContext.md", "progress.md"]
            if f in memory_bank_content
        ],
        "clinerules_present": ".clinerules" in memory_bank_content
    }

@app.get("/v1/models")
async def list_openai_models():
    """
    OpenAI-compatible endpoint for listing available models.
    """
    # OpenAI expects a list of models with id, object, created, owned_by, permission, etc.
    # We'll provide a minimal compatible response.
    import time
    models = []
    for key, model_name in router.models.items():
        models.append({
            "id": model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local-llm",
            "permission": [],
            "root": model_name,
            "parent": None
        })
    return {
        "object": "list",
        "data": models
    }

if __name__ == "__main__":
    print("🚀 Starting Cline Memory Bank Router...")
    print(f"📚 Memory Bank + Fast: {router.models['memory_fast']}")
    print(f"💻 Implementation: {router.models['implementation']}")
    print(f"🧠 Memory Analysis: {router.models['memory_analysis']}")
    print("📖 Optimized for Cline's memory bank workflow")
    print("🌐 Server starting on http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
