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
    API_URL = "http://127.0.0.1:8000/ask"

    st.set_page_config(page_title="Local RAG System", layout="wide")

    st.title("🧠 Local RAG Knowledge System")

    st.markdown("Ask a question based on your indexed PDFs.")

    # Input box
    question = st.text_input("Enter your question:")

    # Submit button
    if st.button("Ask") and question.strip():

        with st.spinner("Generating answer..."):
            response = requests.post(
                API_URL,
                json={"question": question}
            )

            data = response.json()

        st.subheader("📌 Answer")
        st.write(data["answer"])

        st.subheader("📚 Sources")
        for source in data["sources"]:
            st.write(f"- {source}")

        st.caption(f"⏱ Latency: {data['latency']:.2f} seconds")


if __name__ == "__main__":
    main()