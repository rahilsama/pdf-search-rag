from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader

from core.config import DOWNLOADS_PATH
from core.retriever import get_collection
from ingestion.chunking import clean_text, chunk_text


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract full text from a PDF file.

    This is equivalent to your original extract_text_from_pdf function.
    """
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


def ingest_pdfs(base_path: Path = DOWNLOADS_PATH) -> None:
    """
    Ingest all PDFs under base_path into the Chroma collection.

    - Extract and clean text
    - Chunk with spaCy-based chunking
    - Add documents + metadata + ids to Chroma
    """
    collection = get_collection()

    pdf_files: List[Path] = list(base_path.rglob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in the configured downloads folder.")
        return

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

        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append(
                {
                    "source": str(pdf_path),
                    "chunk": i,
                }
            )
            ids.append(f"doc_{id_counter}")
            id_counter += 1

        print(f"Max chunk length: {max(len(c) for c in chunks)}")
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

        print(f"  -> Added {len(chunks)} chunks.\n")

    print("Ingestion complete.")


if __name__ == "__main__":
    ingest_pdfs()