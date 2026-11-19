"""Load supported documents from disk and split them into text chunks."""

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)

DEFAULT_DOCUMENTS_DIR = Path(__file__).resolve().parents[2] / "data" / "documents"
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


def load_documents(directory: str | Path = DEFAULT_DOCUMENTS_DIR) -> list[Document]:
    """Load PDF, TXT, and Markdown files from ``directory``.

    Files are processed recursively in a stable order. Unsupported files and
    files that contain no non-whitespace text are skipped. A missing directory
    or an individual file that cannot be loaded is reported and does not
    prevent other documents from being ingested.
    """
    documents_dir = Path(directory)
    if not documents_dir.is_dir():
        logger.warning("Documents directory does not exist: %s", documents_dir)
        return []

    documents: list[Document] = []
    for path in sorted(documents_dir.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.casefold()
        if suffix not in SUPPORTED_SUFFIXES:
            logger.info("Skipping unsupported document type: %s", path)
            continue

        if suffix == ".pdf":
            try:
                with path.open("rb") as file:
                    reader = PdfReader(file)
                    for page_number, page in enumerate(reader.pages):
                        page_content = page.extract_text() or ""
                        if page_content.strip():
                            documents.append(
                                Document(
                                    page_content=page_content,
                                    metadata={
                                        "source": str(path),
                                        "page": page_number,
                                    },
                                )
                            )
            except (OSError, ValueError, PdfReadError):
                logger.warning("Could not load PDF: %s", path, exc_info=True)
            continue

        try:
            page_content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            logger.warning("Could not load text document: %s", path, exc_info=True)
            continue

        if page_content.strip():
            documents.append(
                Document(
                    page_content=page_content,
                    metadata={"source": str(path)},
                )
            )

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split non-empty documents into chunks while retaining their metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size"
        )

    non_empty_documents = [
        document for document in documents if document.page_content.strip()
    ]
    if not non_empty_documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(non_empty_documents)

def iter_document_chunks(
    directory: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
):
    """Yield document chunks incrementally to reduce memory usage."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size"
        )

    documents_dir = Path(directory)

    if not documents_dir.is_dir():
        logger.warning("Documents directory does not exist: %s", documents_dir)
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    for path in sorted(documents_dir.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in SUPPORTED_SUFFIXES:
            continue

        if path.suffix.casefold() == ".pdf":
            try:
                with path.open("rb") as file:
                    reader = PdfReader(file)

                    for page_number, page in enumerate(reader.pages):
                        content = page.extract_text() or ""

                        if content.strip():
                            document = Document(
                                page_content=content,
                                metadata={
                                    "source": str(path),
                                    "page": page_number,
                                },
                            )

                            yield from splitter.split_documents([document])

            except (OSError, ValueError, PdfReadError):
                logger.warning("Could not load PDF: %s", path, exc_info=True)

            continue

        try:
            content = path.read_text(encoding="utf-8")

            if content.strip():
                document = Document(
                    page_content=content,
                    metadata={"source": str(path)},
                )

                yield from splitter.split_documents([document])

        except (OSError, UnicodeError):
            logger.warning("Could not load text document: %s", path, exc_info=True)
            
def ingest_documents(
    directory: str | Path = DEFAULT_DOCUMENTS_DIR,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Load supported files from ``directory`` and return their text chunks."""
    return split_documents(
        load_documents(directory),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
