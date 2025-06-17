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
    task_context: Optional[str] = None  # "implement", "debug", "refactor", "review", "explain"

class ClineOptimizedRouter:
    def __init__(self, config_path: str = "config/models.yaml"):
        """Router optimized specifically for Cline workflows"""
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.ollama_base_url = self.config['ollama']['base_url']
        
        # Model assignments optimized for Cline use cases
        self.models = {
            # Fast model for quick interactions, simple edits, explanations
            "fast": self.config['models']['small']['name'],      # qwen3:30b-a3b
            
            # Coding specialist for complex implementation, debugging, refactoring
            "coding": self.config['models']['large']['name'],    # qwen2.5-coder:32b
            
            # Analysis model for architecture, review, complex problem solving
            "analysis": self.config['models']['multimodal']['name']  # llama4:16x17b
        }
        
        # Cline workflow patterns
        self.cline_patterns = {
            "fast": {
                "keywords": [
                    "explain this", "what does", "quick question", "help me understand",
                    "syntax", "how to", "what is", "simple", "basic", "briefly"
                ],
                "patterns": [
                    r"\bexplain\b.*\b(line|function|variable)\b",
                    r"\bwhat\s+(is|does)\b",
                    r"\bhow\s+to\b(?!.*implement|.*build|.*create)",
                    r"\bquick\b.*\b(help|question|fix)\b",
                    r"^tell me about",
                    r"\bsyntax\b.*\berror\b"
                ],
                "file_indicators": [
                    "single line", "one liner", "quick fix", "typo", "rename"
                ]
            },
            
            "coding": {
                "keywords": [
                    "implement", "create function", "build", "develop", "code", "write",
                    "debug", "fix bug", "error", "exception", "refactor", "optimize",
                    "test", "unit test", "integration", "api", "endpoint", "database"
                ],
                "patterns": [
                    r"\b(implement|create|build|develop|write)\b.*\b(function|class|method|component)\b",
                    r"\b(debug|fix|solve)\b.*\b(bug|error|issue|problem)\b",
                    r"\b(refactor|optimize|improve|enhance)\b",
                    r"\b(test|testing|unit test|integration test)\b",
                    r"\b(api|endpoint|route|controller)\b",
                    r"\b(database|sql|query|schema)\b",
                    r"```[\s\S]*```",  # Code blocks
                    r"\b(add|modify|update|change)\b.*\b(file|function|class)\b"
                ],
                "file_indicators": [
                    "multiple files", "several functions", "complex logic", "algorithm"
                ]
            },
            
            "analysis": {
                "keywords": [
                    "architecture", "design", "approach", "strategy", "review",
                    "analyze", "evaluate", "assess", "compare", "best practices",
                    "patterns", "structure", "organization", "performance", "security"
                ],
                "patterns": [
                    r"\b(architecture|design|structure)\b.*\b(review|analysis|evaluation)\b",
                    r"\b(analyze|evaluate|assess|review)\b.*\b(codebase|project|implementation)\b",
                    r"\b(best\s+practices|patterns|approaches)\b",
                    r"\b(performance|optimization|scalability)\b.*\b(analysis|review)\b",
                    r"\b(security|vulnerability|audit)\b",
                    r"\bcompare\b.*\b(approaches|solutions|implementations)\b",
                    r"\b(overall|complete|comprehensive)\b.*\b(review|analysis)\b"
                ],
                "file_indicators": [
                    "entire project", "whole codebase", "all files", "complete review"
                ]
            }
        }
        
        # Session tracking for Cline
        self.active_sessions = {}  # workspace_path -> session_info
        self.file_change_tracking = defaultdict(list)  # file_path -> list of changes
        
        logger.info("Cline-optimized router initialized", models=self.models)

    def determine_cline_model(self, request: ClineRequest) -> Tuple[str, str]:
        """Determine best model for Cline workflow"""
        
        # Get the latest user message
        user_messages = [msg.content.lower() for msg in request.messages if msg.role == "user"]
        if not user_messages:
            return "fast", "no_user_messages"
        
        latest_message = user_messages[-1]
        
        # Context indicators from Cline
        context_indicators = []
        if request.task_context:
            context_indicators.append(request.task_context.lower())
        if request.current_file:
            context_indicators.append(f"working_on_{Path(request.current_file).suffix}")
        if request.open_files and len(request.open_files) > 3:
            context_indicators.append("multiple_files_open")
        if request.recent_changes and len(request.recent_changes) > 2:
            context_indicators.append("recent_changes")
        
        # Scoring system
        scores = {"fast": 0, "coding": 0, "analysis": 0}
        
        # 1. Explicit task context (highest priority)
        if request.task_context:
            task_mapping = {
                "explain": "fast",
                "implement": "coding", 
                "debug": "coding",
                "refactor": "coding",
                "review": "analysis",
                "analyze": "analysis",
                "design": "analysis"
            }
            if request.task_context.lower() in task_mapping:
                scores[task_mapping[request.task_context.lower()]] += 10
        
        # 2. Pattern matching on message content
        for model_type, patterns_config in self.cline_patterns.items():
            # Keyword matching
            for keyword in patterns_config["keywords"]:
                if keyword in latest_message:
                    scores[model_type] += 2
            
            # Regex pattern matching
            for pattern in patterns_config["patterns"]:
                matches = len(re.findall(pattern, latest_message, re.IGNORECASE))
                scores[model_type] += matches * 3
            
            # File indicator matching
            for indicator in patterns_config["file_indicators"]:
                if indicator in latest_message:
                    scores[model_type] += 1
        
        # 3. File context analysis
        if request.current_file:
            file_ext = Path(request.current_file).suffix.lower()
            complex_files = ['.py', '.js', '.ts', '.java', '.cpp', '.rs']
            if file_ext in complex_files:
                scores["coding"] += 2
        
        # 4. Multi-file operations
        if request.open_files and len(request.open_files) > 5:
            scores["analysis"] += 3
        elif request.open_files and len(request.open_files) > 2:
            scores["coding"] += 2
        
        # 5. Message complexity
        if len(latest_message) > 500:
            scores["analysis"] += 2
        elif len(latest_message) > 200:
            scores["coding"] += 1
        else:
            scores["fast"] += 1
        
        # 6. Code block detection
        code_blocks = len(re.findall(r'```[\s\S]*?```', latest_message))
        if code_blocks > 0:
            scores["coding"] += code_blocks * 4
        
        # 7. Recent session context
        if request.workspace_path in self.active_sessions:
            session = self.active_sessions[request.workspace_path]
            # If we used a complex model recently, bias towards it for continuity
            last_model = session.get("last_model")
            if last_model in ["coding", "analysis"]:
                scores[last_model] += 1
        
        # Determine winner
        selected_model = max(scores, key=scores.get)
        max_score = scores[selected_model]
        
        # Default to fast if no clear winner
        if max_score == 0:
            selected_model = "fast"
            reasoning = "default_no_patterns"
        else:
            reasoning = f"score_{max_score}_context_{len(context_indicators)}"
        
        # Update session tracking
        if request.workspace_path:
            if request.workspace_path not in self.active_sessions:
                self.active_sessions[request.workspace_path] = {}
            self.active_sessions[request.workspace_path].update({
                "last_model": selected_model,
                "last_request": datetime.now(),
                "message_count": self.active_sessions[request.workspace_path].get("message_count", 0) + 1
            })
        
        logger.info("Cline model selection",
                   selected=selected_model,
                   scores=scores,
                   reasoning=reasoning,
                   context_indicators=context_indicators,
                   workspace=request.workspace_path)
        
        return selected_model, reasoning

    async def call_ollama_with_context(self, model_key: str, request: ClineRequest) -> Dict:
        """Call Ollama with Cline-optimized context"""
        
        model_name = self.models[model_key]
        
        # Build enhanced context for the model
        enhanced_messages = []
        
        # Add system message with context if it's a coding task
        if model_key in ["coding", "analysis"] and request.workspace_path:
            context_parts = []
            
            if request.current_file:
                context_parts.append(f"Currently working on: {request.current_file}")
            
            if request.open_files:
                context_parts.append(f"Open files: {', '.join(request.open_files[:5])}")
            
            if request.recent_changes:
                context_parts.append(f"Recent changes: {', '.join(request.recent_changes[:3])}")
            
            if request.task_context:
                context_parts.append(f"Task type: {request.task_context}")
            
            if context_parts:
                system_context = "CONTEXT: " + " | ".join(context_parts)
                enhanced_messages.append({"role": "system", "content": system_context})
        
        # Add original messages
        for msg in request.messages:
            enhanced_messages.append({"role": msg.role, "content": msg.content})
        
        # Model-specific settings
        model_configs = {
            "fast": {"temperature": 0.7, "num_ctx": 16384, "num_predict": 2048},
            "coding": {"temperature": 0.1, "num_ctx": 32768, "num_predict": 4096},
            "analysis": {"temperature": 0.2, "num_ctx": 100000, "num_predict": 8192}
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
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    duration = (datetime.now() - start_time).total_seconds()
                    
                    logger.info("Cline model call successful",
                               model=model_name,
                               model_key=model_key,
                               duration=duration,
                               workspace=request.workspace_path)
                    
                    return result
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=error_text)

# FastAPI app optimized for Cline
app = FastAPI(
    title="Cline-Optimized LLM Router",
    description="Intelligent routing optimized for Cline + VSCode workflows",
    version="2.0.0"
)

router = ClineOptimizedRouter()

@app.post("/v1/chat/completions")
async def cline_chat_completions(request: ClineRequest):
    """Cline-optimized chat completions endpoint"""
    
    start_time = datetime.now()
    
    # Determine optimal model for this Cline request
    model_key, reasoning = router.determine_cline_model(request)
    selected_model = router.models[model_key]
    
    try:
        result = await router.call_ollama_with_context(model_key, request)
        duration = (datetime.now() - start_time).total_seconds()
        
        # OpenAI-compatible response with Cline enhancements
        response = {
            "id": f"cline-{int(datetime.now().timestamp())}",
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
            "cline_info": {
                "model_type": model_key,
                "selected_model": selected_model,
                "reasoning": reasoning,
                "duration": duration,
                "workspace": request.workspace_path,
                "current_file": request.current_file,
                "task_context": request.task_context
            }
        }
        
        return response
        
    except Exception as e:
        logger.error("Cline request failed", error=str(e), workspace=request.workspace_path)
        raise

@app.post("/cline/context-refresh")
async def refresh_workspace_context(workspace_path: str, background_tasks: BackgroundTasks):
    """Refresh context for a specific workspace"""
    
    # This would trigger vector store refresh in the background
    background_tasks.add_task(refresh_vector_store, workspace_path)
    
    return {"status": "refresh_queued", "workspace": workspace_path}

@app.get("/cline/workspace-info/{workspace_path:path}")
async def get_workspace_info(workspace_path: str):
    """Get information about a workspace session"""
    
    session_info = router.active_sessions.get(workspace_path, {})
    
    return {
        "workspace": workspace_path,
        "session_active": workspace_path in router.active_sessions,
        "last_model": session_info.get("last_model"),
        "message_count": session_info.get("message_count", 0),
        "last_request": session_info.get("last_request")
    }

@app.get("/cline/model-stats")
async def get_model_stats():
    """Get usage statistics for Cline optimization"""
    
    # Aggregate stats from active sessions
    total_sessions = len(router.active_sessions)
    model_usage = defaultdict(int)
    
    for session in router.active_sessions.values():
        if "last_model" in session:
            model_usage[session["last_model"]] += 1
    
    return {
        "active_sessions": total_sessions,
        "model_usage": dict(model_usage),
        "available_models": router.models
    }

async def refresh_vector_store(workspace_path: str):
    """Background task to refresh vector store for workspace"""
    # Implementation would go here
    logger.info("Refreshing vector store", workspace=workspace_path)

if __name__ == "__main__":
    print("🚀 Starting Cline-Optimized LLM Router...")
    print(f"⚡ Fast model: {router.models['fast']}")
    print(f"💻 Coding model: {router.models['coding']}")
    print(f"🧠 Analysis model: {router.models['analysis']}")
    print("🔧 Optimized for Cline + VSCode workflows")
    print("🌐 Server starting on http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
