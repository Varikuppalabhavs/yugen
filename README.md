# Agentic RAG — a Tool-Using AI Knowledge Assistant

A 14-day learning project for building a document-grounded assistant with Python, LangChain, LangGraph, Gemini, ChromaDB, and Streamlit.

## Project layout

- `src/agentic_rag/ingest.py` — document loading and chunking.
- `src/agentic_rag/retrieval.py` — embeddings, ChromaDB storage, and retrieval.
- `src/agentic_rag/tools.py` — agent tools, including external web search.
- `src/agentic_rag/graph.py` — LangGraph routing, relevance checks, and retries.
- `src/agentic_rag/app.py` — Streamlit chat interface.
- `data/documents/` — documents to ingest.
- `data/chroma/` — local vector-store data (created when implemented).
- `tests/` — project tests.

The source files are currently scaffolds; application logic has not been implemented.

## Dependencies and configuration

Install the packages listed in `requirements.txt` into the project virtual environment when ready. Copy `.env.example` to `.env` and add your Google Gemini and Tavily API keys. Never commit `.env`.
