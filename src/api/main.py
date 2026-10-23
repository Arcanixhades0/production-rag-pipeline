"""
REST API for the RAG pipeline.

Run:   uvicorn src.api.main:app --reload
Docs:  http://127.0.0.1:8000/docs
"""
import json
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import LLM_PROVIDER

state = {}


@asynccontextmanager
async def lifespan(app):
    # Load models ONCE at startup, not on every request
    from src.pipeline import RAGPipeline
    state["pipeline"] = RAGPipeline()
    yield
    state.clear()


app = FastAPI(title="Production RAG Pipeline", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)
    source_type: Optional[Literal["pdf", "web", "sql"]] = None


@app.get("/health")
def health():
    return {"status": "ok", "llm_provider": LLM_PROVIDER, "ready": "pipeline" in state}


@app.post("/query")
def query(request: QueryRequest):
    # Plain "def" (not async): FastAPI runs it in a thread pool,
    # so a slow LLM call doesn't block other requests
    try:
        return state["pipeline"].answer(request.question, k=request.k, source_type=request.source_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
def query_stream(request: QueryRequest):
    """Server-Sent Events: sources arrive first, then answer tokens as they are generated."""

    def events():
        for event in state["pipeline"].stream(request.question, k=request.k, source_type=request.source_type):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
