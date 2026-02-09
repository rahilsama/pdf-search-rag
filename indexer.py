import os
import chromadb
from chromadb.api.types import Documents, Embeddings
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import re

import spacy



nlp = spacy.load("en_core_web_sm")

# ----------------------------
# CONFIG
# ----------------------------
DOWNLOADS_PATH = Path.home() / "pdfs"
COLLECTION_NAME = "my_pdfs"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ----------------------------
# CONNECT TO CHROMA
# ----------------------------
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

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=emb_fn
)

# ----------------------------
# PDF TEXT EXTRACTION
# ----------------------------

def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Failed to read {pdf_path}: {e}")
        return ""


# ----------------------------
# SIMPLE TEXT CHUNKING
# ----------------------------
def clean_text(text: str) -> str:
    # Collapse multiple newlines
    text = re.sub(r"\n{2,}", "\n", text)

    # Remove common page number patterns
    text = re.sub(r"Page\s*\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

    # Collapse extra spaces
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()

def chunk_text(text: str):
    # Hard pre-split large documents
    raw_blocks = [
        text[i:i+50000]
        for i in range(0, len(text), 50000)
    ]

    chunks = []

    for block in raw_blocks:
        doc = nlp(block)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        current_chunk = ""

        for sentence in sentences:
            if len(sentence) > CHUNK_SIZE:
                for i in range(0, len(sentence), CHUNK_SIZE):
                    chunks.append(sentence[i:i+CHUNK_SIZE])
                continue

            if len(current_chunk) + len(sentence) + 1 <= CHUNK_SIZE:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

    return chunks

# ----------------------------
# INGEST ALL PDFs
# ----------------------------

pdf_files = list(DOWNLOADS_PATH.rglob("*.pdf"))

if not pdf_files:
    print("No PDFs found in Downloads folder.")
    exit()

print(f"Found {len(pdf_files)} PDFs. Starting ingestion...\n")

id_counter = 0

for pdf_path in pdf_files:
    print(f"Processing: {pdf_path.name}")
    raw_text = extract_text_from_pdf(pdf_path)
    text = clean_text(raw_text)

    if not text.strip():
        print("  -> No extractable text. Skipping.\n")
        continue

    chunks = chunk_text(text)

    documents = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append({
            "source": str(pdf_path),
            "chunk": i
        })
        ids.append(f"doc_{id_counter}")
        id_counter += 1

    print(f"Max chunk length: {max(len(c) for c in chunks)}")
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"  -> Added {len(chunks)} chunks.\n")

print("Ingestion complete.")

# ----------------------------
# TEST QUERY
# ----------------------------

results = collection.query(
    query_texts=["Explain concurrency in Go"],
    n_results=3
)

print("\nTop Results:")
if results["documents"]:
    for doc in results["documents"][0]:
        print("-", doc[:500], "...")