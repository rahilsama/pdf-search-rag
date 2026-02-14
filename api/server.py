from fastapi import FastAPI
from pydantic import BaseModel
import time

from core.rag_pipeline import run_rag
from core.llm import get_model, get_tokenizer

app = FastAPI()

# Load once at startup
model = get_model()
tokenizer = get_tokenizer()


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(request: QuestionRequest):
    start = time.time()

    result = run_rag(
        request.question,
        model=model,
        tokenizer=tokenizer,
    )

    latency = time.time() - start

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "latency": latency,
    }