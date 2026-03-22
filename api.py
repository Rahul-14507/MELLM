from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import os
from pathlib import Path
from orchestrator import LLMRouter

app = FastAPI(
    title="MELLM API",
    description="Multi-Expert LLM Router — route queries to specialist models",
    version="0.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Single router instance shared across all requests
# Preference chain: user_config.yaml > config.yaml
config_path = "user_config.yaml" if Path("user_config.yaml").exists() else "config.yaml"
router = LLMRouter(config_path=config_path)

class QueryRequest(BaseModel):
    prompt: str
    domain_hint: Optional[str] = None  # optional override: "code", "math", etc.
    stream: Optional[bool] = False

class SubResult(BaseModel):
    domain: str
    sub_prompt: str
    response: str

class QueryResponse(BaseModel):
    domain: str
    response: str
    rewritten_prompt: str
    confidence: float
    specialist_load_time: float
    inference_time: float
    cache_hit: bool
    context_turns: int
    is_multi_agent: bool = False
    domains_used: Optional[List[str]] = None
    sub_results: Optional[List[SubResult]] = None

from fastapi.responses import StreamingResponse
import json

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        # Note: LLMRouter.query currently doesn't use domain_hint, but we could add it.
        result = router.query(request.prompt)
        return QueryResponse(
            domain=result["domain"],
            response=result["response"],
            rewritten_prompt=result["rewritten_prompt"],
            confidence=result["confidence"],
            specialist_load_time=result["specialist_load_time"],
            inference_time=result["inference_time_seconds"],
            cache_hit=result["cache_hit"],
            context_turns=result["context_turns"],
            is_multi_agent=result.get("is_multi_agent", False),
            domains_used=result.get("domains_used"),
            sub_results=result.get("sub_results")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Streaming version of /query. Returns Server-Sent Events.
    Each event is a JSON object with a 'type' field:
    - {"type": "routing", "domain": "...", ...}
    - {"type": "loaded", "load_time": 1.2, "cache_hit": false}
    - {"type": "token", "content": "..."}
    - {"type": "done", "response": "...", ...}
    
    Usage with curl:
    curl -X POST http://localhost:8000/query/stream \
      -H "Content-Type: application/json" \
      -d '{"prompt": "Binary search in Java"}' \
      --no-buffer
    """
    def generate():
        try:
            for event in router.stream_query(request.prompt):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

@app.delete("/context")
async def clear_context():
    router.conversation_history.clear()
    return {"status": "Context cleared"}

@app.get("/status")
async def status():
    return {
        "status": "running",
        "version": "0.3.0",
        "active_domain": router.last_domain,
        "context_turns": len(router.conversation_history),
        "session_stats": router.session_stats,
        "domains": list(router.config["specialists"].keys())
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.on_event("shutdown")
def shutdown_event():
    router.shutdown()

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
