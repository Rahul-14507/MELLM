from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from orchestrator import LLMRouter

app = FastAPI(title="LLM Router API")

# Initialize router at startup
try:
    router = LLMRouter()
except Exception as e:
    # Error will be caught by FastAPI on startup
    print(f"FAILED TO INITIALIZE ROUTER: {e}")
    router = None

class QueryRequest(BaseModel):
    prompt: str

@app.get("/health")
def health():
    return {"status": "ok", "router_initialized": router is not None}

@app.post("/query")
def process_query(request: QueryRequest):
    if router is None:
        raise HTTPException(status_code=500, detail="Router is not initialized")
    
    try:
        result = router.query(request.prompt)
        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
