import time
from typing import Any, Dict, List

from .config import DEFAULT_TOP_K
from .llm import generate_answer
from .retriever import retrieve_documents



def build_prompt(context: str, question: str) -> str:
    """
    Build the RAG prompt, preserving your original instructions and style.
    """
    return f"""
You are a helpful assistant.

Answer the question using ONLY the provided context.
If the answer is not in the context, say so clearly.
Do not mention page numbers.
Write a complete and well-structured answer.

Context:
{context}

Question:
{question}

Answer:
""".strip()


def _extract_context(results: Dict[str, Any]) -> str:
    """
    Join top documents into a single context string, as in the original script.
    """
    documents: List[List[str]] = results.get("documents") or []
    if not documents:
        return ""
    return "\n\n".join(documents[0])


def _extract_sources(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract source metadata from Chroma results for transparency.

    This is additional structure on top of your original behavior; it does not
    change the model output logic.
    """
    sources: List[Dict[str, Any]] = []

    documents: List[List[str]] = results.get("documents") or []
    metadatas: List[List[Dict[str, Any]]] = results.get("metadatas") or []
    ids: List[List[str]] = results.get("ids") or []

    if not documents:
        return sources

    docs = documents[0]
    metas = metadatas[0] if metadatas else [{}] * len(docs)
    id_list = ids[0] if ids else [None] * len(docs)

    for doc, meta, _id in zip(docs, metas, id_list):
        sources.append(
            {
                "id": _id,
                "source": meta.get("source"),
                "chunk_index": meta.get("chunk"),
                "preview": doc[:500] + ("..." if len(doc) > 500 else ""),
            }
        )

    return sources


def run_rag(query: str, model, tokenizer, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
    """
    End-to-end RAG pipeline:

    - Accepts user query
    - Retrieves top-k chunks
    - Builds prompt
    - Calls LLM
    - Returns structured response: {answer, sources, latency_seconds}
    """
    start_time = time.time()

    results = retrieve_documents(query, top_k=top_k)
    # context = _extract_context(results)

    MAX_CONTEXT_CHARS = 2000
    context = "\n\n".join(results["documents"][0])
    context = context[:MAX_CONTEXT_CHARS]

    prompt = build_prompt(context=context, question=query)
    answer = generate_answer(prompt, model, tokenizer)

    latency_seconds = time.time() - start_time
    sources = _extract_sources(results)

    return {
        "answer": answer,
        "sources": sources,
        "latency_seconds": latency_seconds,
        "raw_results": results,  # keep for debugging / future use
    }