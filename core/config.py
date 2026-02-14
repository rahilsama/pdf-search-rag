from pathlib import Path

# ----------------------------
# General paths
# ----------------------------

DOWNLOADS_PATH: Path = Path.home() / "pdfs"

# ----------------------------
# Chroma / vector store
# ----------------------------

COLLECTION_NAME: str = "my_pdfs"
CHROMA_HOST: str = "localhost"
CHROMA_PORT: int = 8000

# ----------------------------
# Embeddings
# ----------------------------

EMBEDDING_MODEL_NAME: str = "nomic-ai/nomic-embed-text-v1"

# ----------------------------
# LLM
# ----------------------------

LLM_MODEL_NAME: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ----------------------------
# Chunking
# ----------------------------

CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100  # currently not used, preserved from original config
HARD_SPLIT_SIZE: int = 50000  # size for pre-splitting very large documents

# ----------------------------
# Retrieval / RAG
# ----------------------------

DEFAULT_TOP_K: int = 2
MAX_NEW_TOKENS: int = 80
TEMPERATURE: float = 0.2