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
    version="0.5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

from fastapi.responses import StreamingResponse, FileResponse
import json

# Single router instance shared across all requests
_router_instance: LLMRouter | None = None

def set_router(r: LLMRouter):
    global _router_instance
    _router_instance = r

def get_router() -> LLMRouter:
    global _router_instance
    if _router_instance is None:
        config_path = "user_config.yaml" if Path("user_config.yaml").exists() else "config.yaml"
        _router_instance = LLMRouter(config_path=config_path)
    return _router_instance

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

@app.get("/")
async def serve_ui():
    return FileResponse(Path(__file__).parent / "webui" / "index.html")

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        r = get_router()
        result = r.query(request.prompt)
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
    """
    def generate():
        try:
            r = get_router()
            for event in r.stream_query(request.prompt):
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

@app.get("/status")
async def status():
    from loader.airllm_loader import GGUF_REGISTRY
    r = get_router()
    specialists = {}
    for domain, spec in r.config["specialists"].items():
        model_id = spec["model_id"]
        gguf_file = GGUF_REGISTRY.get(model_id, ("", "unknown"))[1]
        cached = (r.loader.cache_dir / gguf_file).exists()
        active = r.last_domain == domain
        specialists[domain] = {
            "model_id": model_id,
            "gguf_file": gguf_file,
            "cached": cached,
            "active": active,
        }
    return {
        "version": "0.5.0",
        "active_domain": r.last_domain,
        "context_turns": len(r.conversation_history),
        "max_history": r.max_history,
        "session_stats": r.session_stats,
        "domain_streak": r.domain_streak[-5:],
        "specialists": specialists,
        "router": {
            "model_id": r.config["router"]["model_id"],
            "loaded": True
        }
    }

@app.delete("/context")
async def clear_context():
    get_router().conversation_history.clear()
    return {"status": "cleared"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.on_event("shutdown")
def shutdown_event():
    get_router().shutdown()

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
