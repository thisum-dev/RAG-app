"""
rag/embeddings.py - Handles loading and using the sentence-transformer embedding model.

Responsibility:
  - Load the HuggingFace sentence-transformers model once (singleton pattern).
  - Provide a function to embed a single text or list of texts.
"""

import logging
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

# ─── Singleton: load model once to avoid reloading on every request ──────────
_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """
    Returns the embedding model, loading it the first time it is called.
    Subsequent calls reuse the already-loaded model (singleton pattern).
    """
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded successfully.")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of text strings.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    model = get_embedding_model()
    # encode() returns a numpy array; convert to Python list for compatibility
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.

    Args:
        query: The search query.

    Returns:
        A single embedding vector.
    """
    model = get_embedding_model()
    embedding = model.encode([query], show_progress_bar=False)
    return embedding[0].tolist()
