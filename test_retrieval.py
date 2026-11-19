"""Temporary script to exercise the ingestion and retrieval pipeline."""
import uuid_shim
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_rag.ingest import ingest_documents
from agentic_rag.retrieval import index_documents, search_documents


def main() -> None:
    """Load, chunk, index, and search the stored sample documents."""
    documents = ingest_documents(ROOT / "data" / "documents")
    print(f"Loaded chunks: {len(documents)}")

    indexed_count = index_documents(documents)
    print(f"Indexed chunks: {indexed_count}")

    results = search_documents("How does RAG work?", k=4)
    print(f"Retrieved results: {len(results)}")
    for index, result in enumerate(results, start=1):
        source = result.metadata.get("source", "unknown")
        print(f"\nResult {index} | source: {source}")
        print(result.page_content)


if __name__ == "__main__":
    main()
