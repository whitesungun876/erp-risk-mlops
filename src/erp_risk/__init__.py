"""ERP document-level risk scoring, with source data kept outside the repository."""

from .data import Document, read_documents

__all__ = ["Document", "read_documents"]
