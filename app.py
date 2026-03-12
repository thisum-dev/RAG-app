"""
app.py - Flask application entry point.

Responsibility:
  - Define Flask routes (HTTP API).
  - Delegate ALL business logic to RAGPipeline.
  - Return clean JSON responses.
  - Handle errors gracefully.

This file intentionally contains NO RAG logic — only routing glue code.
"""

import logging
import uuid
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from config import UPLOAD_DIR, ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH
from rag.pipeline import RAGPipeline

# ─── Load .env before anything else ──────────────────────────────────────────
load_dotenv()

# ─── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ─── Singleton pipeline (shared across requests in this process) ──────────────
pipeline = RAGPipeline()


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    """Return True if the filename has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the single-page frontend."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    POST /upload
    Accept a PDF file, save it, and trigger the RAG ingestion pipeline.

    Form field: "file" (multipart/form-data)

    Returns (JSON):
      200: { "message": "...", "chunks_indexed": N, "file": "name.pdf" }
      400: { "error": "..." }
      500: { "error": "..." }
    """
    # ── Validate request ──────────────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request. Use key 'file'."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are supported."}), 400

    # ── Save file ─────────────────────────────────────────────────────────────
    safe_name = Path(file.filename).name          # Strip any directory traversal
    save_path = UPLOAD_DIR / safe_name
    file.save(str(save_path))
    logger.info(f"File saved: {save_path}")

    # ── Ingest ────────────────────────────────────────────────────────────────
    try:
        result = pipeline.ingest_document(save_path)
        return jsonify({
            "message": f"File '{result['file']}' ingested successfully.",
            "chunks_indexed": result["chunks_indexed"],
            "file": result["file"],
        }), 200
    except Exception as e:
        logger.exception("Ingestion failed.")
        return jsonify({"error": f"Ingestion failed: {str(e)}"}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """
    POST /chat
    Accept a user message and return an AI-generated answer with sources.

    JSON body:
      { "message": "...", "session_id": "<optional UUID>" }

    Returns (JSON):
      200: { "answer": "...", "sources": [...], "session_id": "..." }
      400: { "error": "..." }
      500: { "error": "..." }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    question = data.get("message", "").strip()
    if not question:
        return jsonify({"error": "Field 'message' is required and cannot be empty."}), 400

    # Use provided session_id or generate a new one
    session_id = data.get("session_id") or str(uuid.uuid4())

    try:
        result = pipeline.answer(question=question, session_id=session_id)
        return jsonify({
            "answer":     result["answer"],
            "sources":    result["sources"],
            "session_id": session_id,          # Echo back so the client can persist it
        }), 200
    except FileNotFoundError as e:
        # FAISS index doesn't exist yet — user hasn't uploaded a document
        logger.warning(str(e))
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Chat pipeline failed.")
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # debug=True enables auto-reload and detailed tracebacks during development.
    # Set debug=False (or use a production WSGI server like gunicorn) in production.
    app.run(debug=True, host="0.0.0.0", port=5000)
