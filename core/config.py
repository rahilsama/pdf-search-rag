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

LLM_MODEL_NAME: str = "microsoft/Phi-3-mini-4k-instruct"

# ----------------------------
# Chunking
# ----------------------------

CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 100  # currently not used, preserved from original config
HARD_SPLIT_SIZE: int = 50000  # size for pre-splitting very large documents

# ----------------------------
# Retrieval / RAG
# ----------------------------

DEFAULT_TOP_K: int = 5
MAX_NEW_TOKENS: int = 300
TEMPERATURE: float = 0.3