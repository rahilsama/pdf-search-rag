## Local RAG System

A local Retrieval-Augmented Generation (RAG) system using:

- **ChromaDB** for vector storage
- **SentenceTransformer** embeddings (`nomic-ai/nomic-embed-text-v1`)
- **spaCy**-based custom chunking
- **Local HuggingFace LLM** (`microsoft/Phi-3-mini-4k-instruct`)
- **No external APIs**

### Project Structure

- `app/app.py`: CLI entry point for querying the RAG pipeline.
- `core/config.py`: Central configuration for paths, models, and parameters.
- `core/embeddings.py`: Embedding model loading and Chroma embedding function.
- `core/llm.py`: LLM loading and text generation.
- `core/retriever.py`: Chroma client, collection, and retrieval helpers.
- `core/rag_pipeline.py`: End-to-end RAG pipeline (query → retrieve → prompt → LLM).
- `ingestion/chunking.py`: Text cleaning and spaCy-based chunking.
- `ingestion/indexer.py`: PDF ingestion into Chroma.
- `data/`: Optional directory for any additional local data you want to store.

### Setup

1. **Install Python dependencies:**

   pip install -r requirements.txt
   