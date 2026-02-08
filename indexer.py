import os
import chromadb
from chromadb.utils import embedding_functions
from typing import Any, cast
from pathlib import Path
from pypdf import PdfReader

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

emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=cast(Any, emb_fn)
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

def chunk_text(text: str):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
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
    text = extract_text_from_pdf(pdf_path)

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