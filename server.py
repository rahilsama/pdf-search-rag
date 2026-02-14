"""
FastAPI backend for the RAG service.
Model and tokenizer are loaded once at startup; no reload per request.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from core.llm import get_model, get_tokenizer
from core.rag_pipeline import run_rag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globals set at startup; never re-initialized per request
_model = None
_tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and tokenizer once when the app starts."""
    global _model, _tokenizer
    logger.info("Loading LLM and tokenizer...")
    _model = get_model()
    _tokenizer = get_tokenizer()
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    logger.info("LLM ready.")
    yield
    # Shutdown: nothing to do (process exits)


app = FastAPI(title="RAG API", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def query(request: QueryRequest):
    """Run RAG on the given question. Uses the model loaded at startup."""
    result = run_rag(
        request.question,
        model=_model,
        tokenizer=_tokenizer,
    )
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "latency": result["latency_seconds"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}
