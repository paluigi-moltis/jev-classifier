"""jev-classifier: text classification powered by TypeSafe Jev via Vercel AI Gateway."""

from .engine import ClassificationResult, Label, classify, classify_async

__all__ = ["ClassificationResult", "Label", "classify", "classify_async"]
