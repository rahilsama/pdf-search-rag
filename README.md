# Local RAG System

A **local Retrieval-Augmented Generation (RAG)** app that answers questions from your own PDFs. It uses ChromaDB for vector search, SentenceTransformer embeddings, spaCy-based chunking, and a local HuggingFace LLM (Phi-3). No external APIs—everything runs on your machine.

---

## Installation

1. **Clone the repo and go to the project root:**
   ```bash
   cd search-pdf
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the spaCy English model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Run ChromaDB** (e.g. in Docker or as a local server) so it’s available at `localhost:8000`.  
   Adjust `CHROMA_HOST` / `CHROMA_PORT` in `core/config.py` if your setup differs.

6. **Place PDFs** in the folder used for ingestion (default: `~/pdfs`). Change `DOWNLOADS_PATH` in `core/config.py` to use another directory.

---

## Running the Application

**Index your PDFs (once, or when you add new files):**
```bash
python -m ingestion.indexer
```

**Query via the Streamlit UI:**
```bash
streamlit run app/app.py
```

**Or run the FastAPI server:**
```bash
uvicorn api.server:app --reload --host 0.0.0.0 --port 8001
```
Then send `POST` requests to `/ask` with a JSON body like `{"question": "Your question here"}`.

---

## Project Structure

| Path | Purpose |
|------|--------|
| `app/app.py` | Streamlit UI for querying |
| `api/server.py` | FastAPI server for programmatic queries |
| `core/config.py` | Settings (paths, models, Chroma, chunking) |
| `core/embeddings.py` | Embedding model and Chroma embedding function |
| `core/llm.py` | LLM loading and generation |
| `core/retriever.py` | Chroma client and retrieval |
| `core/rag_pipeline.py` | End-to-end RAG (retrieve → prompt → LLM) |
| `ingestion/chunking.py` | Text cleaning and spaCy chunking |
| `ingestion/indexer.py` | PDF ingestion into Chroma |
| `data/` | Optional local data directory |
