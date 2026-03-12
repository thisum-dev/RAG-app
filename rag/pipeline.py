"""
rag/pipeline.py - Orchestrator for the full RAG flow.

Responsibility:
  - Coordinate ingest, embedding, vector store, retrieval, generation, and memory.
  - Provide two public methods:
      * ingest_document(file_path)  → build/update the FAISS index.
      * answer(question, session_id) → run the full RAG query pipeline.

app.py calls ONLY this module for RAG operations — keeping Flask routes clean.
"""

import logging
from pathlib import Path

from rag.ingest import ingest_pdf
from rag.embeddings import embed_texts
from rag.vectorstore import build_and_save_index, invalidate_cache
from rag.retriever import retrieve
from rag.generator import generate_answer
from rag import memory

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Top-level orchestrator of the RAG application.

    Usage:
        pipeline = RAGPipeline()
        pipeline.ingest_document("data/uploads/paper.pdf")
        result = pipeline.answer("What is the conclusion?", session_id="abc123")
        # result = {"answer": "...", "sources": ["chunk1...", "chunk2..."]}
    """

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def ingest_document(self, file_path: str | Path) -> dict:
        """
        Run the ingestion pipeline for a newly uploaded PDF.

        Steps:
          1. Parse PDF and extract text.
          2. Split into overlapping chunks.
          3. Generate embeddings for all chunks.
          4. Build and persist a FAISS index.
          5. Invalidate the in-memory cache so the next query loads fresh data.

        Args:
            file_path: Path to the saved PDF file.

        Returns:
            A dict with status and chunk count, e.g.
            {"status": "ok", "chunks_indexed": 42, "file": "paper.pdf"}
        """
        file_path = Path(file_path)
        logger.info(f"Starting ingestion for: {file_path.name}")

        # Step 1 & 2: PDF → chunks
        chunks = ingest_pdf(file_path)

        # Step 3: chunks → embedding vectors
        logger.info(f"Embedding {len(chunks)} chunks…")
        embeddings = embed_texts(chunks)

        # Step 4: build FAISS index and write to disk
        build_and_save_index(chunks, embeddings)

        # Step 5: clear stale cache so next query uses the new index
        invalidate_cache()

        logger.info(f"Ingestion complete: {len(chunks)} chunks indexed.")
        return {
            "status": "ok",
            "chunks_indexed": len(chunks),
            "file": file_path.name,
        }

    # ── Query ──────────────────────────────────────────────────────────────────

    def answer(self, question: str, session_id: str) -> dict:
        """
        Run the full RAG query pipeline.

        Steps:
          1. Load per-session conversation history.
          2. Retrieve the top-k most relevant chunks from FAISS.
          3. Call the LLM (Groq LLaMA 3) with context + history + question.
          4. Store the turn in memory for future context.
          5. Return the answer and source previews.

        Args:
            question:   The user's natural-language question.
            session_id: Unique session identifier (used for memory).

        Returns:
            {
                "answer":  "<model response>",
                "sources": ["<first 200 chars of chunk 1>", ...]
            }
        """
        logger.info(f"[Session {session_id}] Question: {question[:80]}…")

        # Step 1: Fetch prior conversation turns for this session
        chat_history = memory.get_history(session_id)

        # Step 2: Retrieve relevant chunks from FAISS
        context_chunks = retrieve(question)

        # Step 3: Generate an answer via Groq
        answer_text = generate_answer(question, context_chunks, chat_history)

        # Step 4: Persist this turn so future questions have context
        memory.add_turn(session_id, question, answer_text)

        # Step 5: Build source previews (first 200 chars, stripped)
        sources = [chunk[:200].strip() for chunk in context_chunks]

        logger.info(f"[Session {session_id}] Answer generated. Sources: {len(sources)}")
        return {
            "answer": answer_text,
            "sources": sources,
        }
