
"""Embedding and session-isolated ChromaDB retrieval."""

import os
import time
import hashlib
import re
from pathlib import Path

import uuid_shim  # Must load before LangChain imports.

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
PERSIST_DIRECTORY = PROJECT_ROOT / "data" / "chroma"

COLLECTION_PREFIX = "agentic_rag_"

_embedding_model = None
_vector_stores: dict[str, Chroma] = {}


def _load_env_file():
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=str(ENV_PATH), override=False)
    else:
        load_dotenv(override=False)


def _validate_session_id(session_id: str) -> str:
    """Validate and normalize a session identifier."""

    if not session_id:
        raise ValueError("A session ID is required.")

    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", session_id)

    if not safe_id or len(safe_id) > 50:
        raise ValueError("Invalid session ID.")

    return safe_id


def get_embedding_model():
    global _embedding_model

    _load_env_file()

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key or not api_key.strip():
        raise ValueError("GOOGLE_API_KEY is missing from .env.")

    if _embedding_model is None:
        _embedding_model = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=api_key,
        )

    return _embedding_model


def get_vector_store(session_id: str = "local") -> Chroma:
    """Return the Chroma collection belonging to one session."""

    safe_id = _validate_session_id(session_id)

    if safe_id not in _vector_stores:
        PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

        _vector_stores[safe_id] = Chroma(
            collection_name=f"{COLLECTION_PREFIX}{safe_id}",
            embedding_function=get_embedding_model(),
            persist_directory=str(PERSIST_DIRECTORY),
        )

    return _vector_stores[safe_id]



def index_documents(
    documents: list[Document],
    session_id: str = "local",
) -> int:
    """Index documents in batches with duplicate-safe IDs."""

    if not documents:
        raise ValueError("No documents were provided to index.")

    valid_documents = [
        document
        for document in documents
        if document.page_content and document.page_content.strip()
    ]

    if not valid_documents:
        raise ValueError("No non-empty document content was provided.")

    vector_store = get_vector_store(session_id)

    batch_size = 10
    indexed_count = 0

    for start in range(0, len(valid_documents), batch_size):
        batch = valid_documents[start:start + batch_size]

        document_ids = []

        for document in batch:
            identity = (
                str(document.metadata.get("source", "unknown"))
                + str(document.metadata.get("page", ""))
                + document.page_content
            )

            document_ids.append(
                hashlib.sha256(identity.encode("utf-8")).hexdigest()
            )

        try:
            vector_store.add_documents(
                documents=batch,
                ids=document_ids,
            )

            indexed_count += len(batch)

        except Exception as error:
            error_text = str(error).lower()

            quota_markers = [
                "429",
                "resource_exhausted",
                "quota exceeded",
                "rate limit",
                "too many requests",
            ]

            if any(marker in error_text for marker in quota_markers):
                raise RuntimeError(
                    "EMBEDDING_QUOTA_EXCEEDED: "
                    "The Gemini embedding service has reached its "
                    "usage limit. Your document was not fully indexed. "
                    "Please wait for the quota to reset before retrying."
                ) from error

            raise

        if start + batch_size < len(valid_documents):
            time.sleep(2)

    return indexed_count





def search_documents(
    query: str,
    k: int = 4,
    session_id: str = "local",
) -> list[Document]:
    """Search only documents that currently exist in this session's library."""

    if not query or not query.strip():
        raise ValueError("A non-empty query is required.")

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    safe_id = _validate_session_id(session_id)
    vector_store = get_vector_store(safe_id)

    # Reconstruct the current session's uploaded documents.
    uploads_directory = PROJECT_ROOT / "data" / "uploads" / safe_id

    supported_suffixes = {".pdf", ".txt", ".md", ".markdown"}

    if not uploads_directory.is_dir():
        return []

    active_sources = [
        str(path)
        for path in uploads_directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in supported_suffixes
        and (path.parent / ".indexed").exists()
    ]

    # Never search stale embeddings from deleted or unindexed files.
    if not active_sources:
        return []

    return vector_store.similarity_search(
        query=query,
        k=k,
        filter={"source": {"$in": active_sources}},
    )

def delete_document(source: str, session_id: str) -> int:
    """Delete all indexed chunks belonging to a document."""

    if not source or not source.strip():
        raise ValueError("Document source is required.")

    vector_store = get_vector_store(session_id)

    collection = vector_store._collection

    results = collection.get(
        where={"source": source},
        include=[],
    )

    document_ids = results.get("ids", [])

    if document_ids:
        collection.delete(ids=document_ids)

    return len(document_ids)
