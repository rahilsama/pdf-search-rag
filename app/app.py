import sys
from pathlib import Path
import requests
# Ensure project root is on path so "core" resolves when run via Streamlit or from app/
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.rag_pipeline import run_rag
import streamlit as st
from core.llm import get_model, get_tokenizer
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)




@st.cache_resource
def load_llm():
    model = get_model()
    tokenizer = get_tokenizer()
    return model, tokenizer

model, tokenizer = load_llm()


def main() -> None:
    """
    Simple CLI entry point for querying the local RAG system.
    """
    query = input("Enter your question: ")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # result = run_rag(query, model, tokenizer)
    response = requests.post(
        "http://127.0.0.1:8000/ask",
        json={"question": query},
    )
    data = response.json()

    st.write(data["answer"])
    st.write(data["sources"])
    st.write(f"Latency: {data['latency']:.2f}s")

    # answer = result["answer"]
    # sources = result["sources"]
    # latency_seconds = result["latency_seconds"]

    # print("\nAI Answer:\n")
    # print(answer)

    # if sources:
    #     print("\nSources:")
    #     for idx, src in enumerate(sources, start=1):
    #         source_str = src.get("source") or "Unknown source"
    #         chunk_index = src.get("chunk_index")
    #         print(f"{idx}. {source_str} (chunk {chunk_index})")

    # print(f"\nLatency: {latency_seconds:.2f} seconds")


if __name__ == "__main__":
    main()