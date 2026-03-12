"""
rag/retriever.py - Document retrieval from the FAISS vector store.

Responsibility:
  - Accept a user query string.
  - Embed the query.
  - Fetch the top-k most relevant chunks from the FAISS index.
"""

import logging
from rag.embeddings import embed_query
from rag.vectorstore import search
from config import TOP_K_RESULTS

logger = logging.getLogger(__name__)


def retrieve(query: str, k: int = TOP_K_RESULTS) -> list[str]:
    """
    Retrieve the most relevant document chunks for a given query.

    Steps:
      1. Embed the query using the sentence-transformer model.
      2. Search the FAISS index for the nearest k vectors.
      3. Return the corresponding text chunks.

    Args:
        query: The user's natural-language question.
        k:     Number of chunks to return.

    Returns:
        A list of raw text chunks, most relevant first.

    Raises:
        FileNotFoundError: Propagated from vectorstore if no index exists.
    """
    logger.info(f"Retrieving top-{k} chunks for query: {query[:80]}…")

    # Step 1: Convert the query string to a vector
    query_embedding = embed_query(query)

    # Step 2 & 3: Search FAISS and return matching chunk texts
    chunks = search(query_embedding, k=k)

    logger.info(f"Retrieved {len(chunks)} chunks.")
    return chunks
