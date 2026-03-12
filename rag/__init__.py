# rag/__init__.py
# Marks the rag/ folder as a Python package.
# Import convenience: expose key functions at the package level.

from .pipeline import RAGPipeline

__all__ = ["RAGPipeline"]
