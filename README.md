# Local RAG System

A **local Retrieval-Augmented Generation (RAG)** app that answers questions from your own PDFs. It uses ChromaDB for vector search, SentenceTransformer embeddings, spaCy-based chunking, and a local HuggingFace LLM. No external APIs—everything runs on your machine.

---

## Architecture

The app is split into three services that can run together with Docker Compose:

```
┌─────────────┐     HTTP      ┌─────────────┐     ChromaDB client    ┌─────────────┐
│  Frontend  │  POST /query   │  Backend    │  ───────────────────►  │   Chroma     │
│ (Streamlit)│ ────────────►  │  (FastAPI)  │                        │   (vector   │
│  :8501     │                │  :8001      │  ◄── embeddings,      │    store)    │
└─────────────┘                │             │     query             │   :8000      │
                               │  - RAG      │                        └─────────────┘
                               │  - LLM      │
                               │  (load once)│
                               └─────────────┘
```

- **Frontend (Streamlit)** – UI only; calls the backend over HTTP. No model or RAG logic.
- **Backend (FastAPI)** – Loads the LLM once at startup, exposes `POST /query`, runs RAG and returns answer + sources + latency.
- **Chroma** – Vector DB for embeddings and retrieval. Chroma data and the HuggingFace model cache are persisted via Docker volumes.

---

## Docker setup (recommended)

### Prerequisites

- Docker and Docker Compose
- (Optional) Pre-download models by running ingestion once so the `hf_cache` volume is warm

### Run with Docker Compose

From the project root:

```bash
docker compose up --build
```

- **Streamlit UI:** http://localhost:8501  
- **API:** http://localhost:8001 (e.g. `POST /query` with `{"question": "..."}`)  
- **Chroma:** http://localhost:8000 (internal; backend uses `CHROMA_HOST=chroma`)

### Volumes

- `chroma_data` – Chroma vector DB data (persists across restarts).
- `hf_cache` – HuggingFace model cache (embeddings + LLM); persists so models are not re-downloaded.

### First run

1. Start: `docker compose up --build`.
2. Wait for the API to log “LLM ready.” (first run can be slow while models download).
3. Ingest PDFs by running the indexer **inside the api container** (or against Chroma on 8000 with the same embedding model), e.g.  
   `docker compose exec api python -m ingestion.indexer`  
   (Ensure `DOWNLOADS_PATH` or the path you use is mounted into the container if needed.)
4. Open http://localhost:8501 and ask questions.

---

## Installation (local, non-Docker)

1. **Clone and enter the repo:**
   ```bash
   cd search-pdf
   ```

2. **Create and activate a virtual environment:**
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

5. **Run ChromaDB** (e.g. Docker: `docker run -d -p 8000:8000 -e IS_PERSISTENT=TRUE -v chroma_data:/chroma/chroma chromadb/chroma:latest`).  
   Set `CHROMA_HOST` / `CHROMA_PORT` if different from `localhost:8000`.

6. **Place PDFs** in the ingestion path (default `~/pdfs`, or set `DOWNLOADS_PATH` in `core/config.py`).

---

## Running the application

**Index PDFs (once or when you add new files):**
```bash
python -m ingestion.indexer
```

**Backend only (API):**
```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8001
```
- `POST /query` with body `{"question": "Your question"}`  
- Response: `{"answer": "...", "sources": [...], "latency": float}`

**Frontend (needs backend on 8001):**
```bash
export API_URL=http://localhost:8001   # optional if already default
streamlit run app/app.py
```

---

## Project structure

| Path | Purpose |
|------|--------|
| `server.py` | FastAPI app; model loaded once at startup; `POST /query` |
| `app/app.py` | Streamlit UI; calls backend via HTTP only |
| `core/config.py` | Settings; Chroma via `CHROMA_HOST` / `CHROMA_PORT` |
| `core/embeddings.py` | Embedding model and Chroma embedding function |
| `core/llm.py` | LLM load and generation |
| `core/retriever.py` | Chroma client and retrieval |
| `core/rag_pipeline.py` | RAG pipeline (retrieve → prompt → LLM) |
| `ingestion/chunking.py` | Text cleaning and spaCy chunking |
| `ingestion/indexer.py` | PDF ingestion into Chroma |
| `Dockerfile` | Single image for api and frontend |
| `docker-compose.yml` | chroma, api, frontend with volumes |
