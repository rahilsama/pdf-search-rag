import chromadb
from chromadb.utils import embedding_functions
from typing import Any, cast

# 1. Connect to your local Docker instance
client = chromadb.HttpClient(host='localhost', port=8000)

# 2. Use a lightweight, free embedding model (runs locally)
# This model is open-source and very fast on M1
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = client.get_or_create_collection(name="my_pdfs", embedding_function=cast(Any, emb_fn))

# 3. Add a PDF chunk (Example)
collection.add(
    documents=["This page discusses Go routines and concurrency..."],
    metadatas=[{"source": "'/Users/rahilsama/Downloads/Learning Go An Idiomatic Approach to Real-world Go Programming, 2nd Edition (Jon Bodner) (z-library.sk, 1lib.sk, z-lib.sk).pdf'", "page": 5}],
    ids=["id1"]
)

results = collection.query(
    query_texts=["Explain concurrency in Go"],
    n_results=3
)

print(results["documents"])