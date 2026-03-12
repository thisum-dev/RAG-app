"""
config.py - Central configuration for the RAG Flask application.
All tunable parameters live here to keep the rest of the code clean.
"""

import os
from pathlib import Path

# ─── Base paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR      = BASE_DIR / "data"
UPLOAD_DIR    = DATA_DIR / "uploads"
INDEX_DIR     = DATA_DIR / "index"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ─── Text chunking ────────────────────────────────────────────────────────────
CHUNK_SIZE    = 700       # Characters per chunk
CHUNK_OVERLAP = 100       # Characters of overlap between adjacent chunks

# ─── Embedding model ──────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # Lightweight, fast, good quality

# ─── LLM (Groq) ───────────────────────────────────────────────────────────────
LLM_MODEL_NAME = "llama-3.1-8b-instant"     # Fast Groq-hosted LLaMA 3 model

# ─── FAISS index file name ────────────────────────────────────────────────────
FAISS_INDEX_FILE = INDEX_DIR / "faiss.index"
FAISS_DOCSTORE_FILE = INDEX_DIR / "docstore.pkl"

# ─── Retrieval ────────────────────────────────────────────────────────────────
TOP_K_RESULTS = 4         # Number of chunks to retrieve per query

# ─── Allowed upload extensions ────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"pdf"}

# ─── Flask ────────────────────────────────────────────────────────────────────
MAX_CONTENT_LENGTH = 50 * 1024 * 1024   # 50 MB upload limit
