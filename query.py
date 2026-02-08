import chromadb
from chromadb.api.types import Documents, Embeddings
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "my_pdfs"

# Connect to Chroma
client = chromadb.HttpClient(host='localhost', port=8000)


class NomicEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self):
        # self.name = "nomic-embed-text-v1"
        self.model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1",
            trust_remote_code=True
        )

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(list(input), normalize_embeddings=True)
        return embeddings.tolist()

emb_fn = NomicEmbeddingFunction()

collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=emb_fn
)

# Ask user for query
query = input("Enter your question: ")

results = collection.query(
    query_texts=[query],
    n_results=5
)

print("\nTop Results:\n")
if results["documents"]:
    for i, doc in enumerate(results["documents"][0]):
        print(f"Result {i+1}:")
        print(doc[:500])
        print("-" * 50)