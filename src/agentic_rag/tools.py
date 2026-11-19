
"""Web search utilities for the Agentic RAG assistant."""

import uuid_shim  # Must load before LangChain imports.

import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

def get_web_search_tool():
    """Initialize Tavily web search."""

    if not os.getenv("TAVILY_API_KEY"):
        raise ValueError("TAVILY_API_KEY is missing from .env")

    return TavilySearch(
        max_results=5,
        topic="general",
        search_depth="basic",
    )


def search_web(query: str) -> dict:
    """Search the web and return relevant results."""

    if not query or not query.strip():
        raise ValueError("A search query is required.")

    tool = get_web_search_tool()
    return tool.invoke({"query": query})
