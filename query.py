import chromadb
from chromadb.api.types import Documents, Embeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


COLLECTION_NAME = "my_pdfs"

# Connect to Chroma
client = chromadb.HttpClient(host='localhost', port=8000)


MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)


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


context = "\n\n".join(results["documents"][0]) if results["documents"] else ""

prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the provided context.
If the answer is not in the context, say so clearly.
Do not mention page numbers.
Write a complete and well-structured answer.

Context:
{context}

Question:
{query}

Answer:
"""


inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.3,
        do_sample=True
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nAI Answer:\n")
print(response.split("Answer:")[-1].strip())