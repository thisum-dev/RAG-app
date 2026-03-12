"""
rag/ingest.py - PDF ingestion and text chunking pipeline.

Responsibility:
  - Load a PDF file from disk.
  - Extract raw text from every page.
  - Split the text into overlapping chunks ready for embedding.
"""

import logging
from pathlib import Path
from pypdf import PdfReader
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def load_pdf(file_path: str | Path) -> str:
    """
    Extract all text from a PDF file.

    Args:
        file_path: Path to the PDF.

    Returns:
        A single string containing the entire document text.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the PDF contains no extractable text.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    logger.info(f"Reading PDF: {file_path.name}")
    reader = PdfReader(str(file_path))

    full_text = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text.append(text)
        else:
            logger.warning(f"Page {page_num + 1} yielded no text (possibly an image page).")

    if not full_text:
        raise ValueError("No extractable text found in the PDF. Is it a scanned / image-only PDF?")

    combined = "\n".join(full_text)
    logger.info(f"Extracted {len(combined)} characters from {len(reader.pages)} pages.")
    return combined


def chunk_text(text: str) -> list[str]:
    """
    Split a long string into overlapping fixed-size chunks.

    Uses a simple character-level sliding-window approach:
      - Each chunk is at most CHUNK_SIZE characters.
      - Consecutive chunks overlap by CHUNK_OVERLAP characters so that
        context at chunk boundaries is not lost.

    Args:
        text: The full document text.

    Returns:
        A list of text chunk strings.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        # Move forward by (CHUNK_SIZE - CHUNK_OVERLAP) to create overlap
        start += CHUNK_SIZE - CHUNK_OVERLAP

    logger.info(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks


def ingest_pdf(file_path: str | Path) -> list[str]:
    """
    Full ingestion pipeline: load PDF → extract text → chunk.

    Args:
        file_path: Path to the uploaded PDF.

    Returns:
        List of text chunks ready to be embedded.
    """
    text = load_pdf(file_path)
    chunks = chunk_text(text)
    return chunks
