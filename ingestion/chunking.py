import re
from typing import List

import spacy

from core.config import CHUNK_SIZE, HARD_SPLIT_SIZE

# Load spaCy model once per process
_nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text.

    This is equivalent to your original clean_text function.
    """
    # Collapse multiple newlines
    text = re.sub(r"\n{2,}", "\n", text)

    # Remove common page number patterns
    text = re.sub(r"Page\s*\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

    # Collapse extra spaces
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def chunk_text(text: str) -> List[str]:
    """
    spaCy-based sentence-level chunking, preserving your original logic.

    - Hard pre-split very large documents into blocks of HARD_SPLIT_SIZE
    - Use spaCy to split into sentences
    - Build chunks up to CHUNK_SIZE characters
    """
    # Hard pre-split large documents (unchanged, except using HARD_SPLIT_SIZE constant)
    raw_blocks = [
        text[i : i + HARD_SPLIT_SIZE]  # noqa: E203
        for i in range(0, len(text), HARD_SPLIT_SIZE)
    ]

    chunks: List[str] = []

    for block in raw_blocks:
        doc = _nlp(block)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        current_chunk = ""

        for sentence in sentences:
            if len(sentence) > CHUNK_SIZE:
                # Split very long sentences
                for i in range(0, len(sentence), CHUNK_SIZE):
                    chunks.append(sentence[i : i + CHUNK_SIZE])  # noqa: E203
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