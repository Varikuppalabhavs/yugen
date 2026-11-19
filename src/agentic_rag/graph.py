
"""Conversational Agentic RAG using LangGraph."""



import uuid_shim  # Must load before LangChain imports.

from typing import TypedDict, Any, Literal

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

from .generation import (
    get_llm,
    generate_answer_from_documents,
)
from .retrieval import search_documents
from .tools import search_web


MODEL_NAME = "gemini-3.1-flash-lite"
MAX_RETRIES = 1


# --------------------------------------------------
# GRAPH STATE
# --------------------------------------------------

class AgentState(TypedDict):
    question: str
    intent: str
    session_id: str
    search_query: str
    documents: list[Any]
    relevance: str
    retry_count: int
    chat_history: list[dict[str, str]]
    answer: str
    sources: list[dict[str, Any]]


# --------------------------------------------------
# MODEL
# --------------------------------------------------

def get_model():
    return get_llm()


def extract_text(response) -> str:
    """Safely extract text from Gemini responses."""

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
        )

    return str(content)


# --------------------------------------------------
# INTENT ROUTER
# --------------------------------------------------



def classify_intent(
    question: str,
    chat_history: list[dict] | None = None,
) -> Literal["chat", "documents", "web"]:
    """Route questions using explicit intent and recent conversation context."""

    text = question.lower().strip()

    # Explicit web requests.
    web_keywords = [
        "search the web", "search online", "look it up online",
        "browse the web", "latest news", "latest developments",
        "currently", "right now", "today", "this week",
        "this month", "recent updates",
    ]

    if any(keyword in text for keyword in web_keywords):
        return "web"

    # Explicit references to uploaded documents.
    document_keywords = [
        "uploaded file", "file uploaded", "file i uploaded",
        "document uploaded", "document i uploaded",
        "uploaded document", "uploaded pdf", "uploaded txt",
        "pdf file", "txt file", "text file",
        "pdf uploaded", "txt uploaded",
        "my uploaded", "my document", "my documents",
        "my pdf", "my txt", "my files",
        "in the file", "in this file", "inside the file",
        "in the document", "inside the document",
        "in the pdf", "in the txt",
        "according to the document", "according to my notes",
        "from my files", "from the knowledge base",
        "what does the file say", "what is mentioned in",
        "summarise the file", "summarize the file",
        "summarise the pdf", "summarize the pdf",
        "summarise the txt", "summarize the txt",
        "summarise my document", "summarize my document",
        "certifications mentioned", "skills mentioned",
        "education mentioned", "resume", "cv",
    ]

    if any(keyword in text for keyword in document_keywords):
        return "documents"

    # Casual conversation.
    casual_phrases = {
        "hi", "hello", "hey", "hii", "hiii",
        "good morning", "good afternoon", "good evening",
        "how are you", "thank you", "thanks",
        "bye", "goodbye", "lol", "lmao",
    }

    if text.rstrip("!.,?") in casual_phrases:
        return "chat"

    # Continue document-related conversations naturally.
    recent_history = (chat_history or [])[-8:]

    document_context_keywords = [
        "uploaded file", "uploaded document", "uploaded pdf",
        "uploaded txt", "file i uploaded", "document i uploaded",
        "my resume", "my cv", "my files", "my document",
        "my documents", "my pdf", "my txt","file","pdf","txt","document","documents",
    ]

    for message in recent_history:
        content = message.get("content", "").lower()

        if any(keyword in content for keyword in document_context_keywords):
            return "documents"

        sources = message.get("sources", [])
        if sources and any(
            source.get("page") is not None
            for source in sources
        ):
            return "documents"

    # General questions use web research.
    return "web"



def router_node(state: AgentState) -> dict:
    """Determine intent using conversation context."""

    intent = classify_intent(
        state["question"],
        state.get("chat_history", []),
    )

    return {"intent": intent}



def choose_route(
    state: AgentState,
) -> Literal["chat", "documents", "web"]:
    return state["intent"]


# --------------------------------------------------
# CONVERSATIONAL CHAT
# --------------------------------------------------

def chat_node(state: AgentState) -> dict:
    """Respond naturally without retrieval or web search."""

    history = state.get("chat_history", [])

    messages = [
        SystemMessage(
            content="""You are a friendly, intelligent conversational AI assistant.

Your personality:
- Be natural, warm, relaxed, and approachable.
- Understand casual language, slang, humor, and context.
- Respond like a real conversational assistant, not a search engine.
- Be supportive but honest.
- Do not be unnecessarily formal or robotic.

Your behavior:
- Answer the user's actual question directly.
- Keep simple answers short.
- Give detailed explanations when requested.
- Understand follow-up questions using conversation history.
- Never search the web or invent citations in this mode.
- Never expose internal API responses, metadata, or technical signatures.
- If asked about your own architecture, explain your actual technology stack accurately.

You are powered by Gemini and are part of an Agentic RAG application
built using LangGraph, LangChain, ChromaDB, Gemini embeddings,
and Tavily for web research."""
        )
    ]

    # Include recent conversation history.
    for message in history[-10:]:
        role = message.get("role")
        content = message.get("content", "")

        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            from langchain_core.messages import AIMessage
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=state["question"]))

    response = get_model().invoke(messages)

    return {
        "answer": extract_text(response),
        "sources": [],
    }


# --------------------------------------------------
# DOCUMENT RETRIEVAL
# --------------------------------------------------

def retrieve_node(state: AgentState) -> dict:
    """Retrieve relevant document chunks."""

    query = state.get("search_query") or state["question"]

    documents = search_documents(
        query,
        k=4,
        session_id=state["session_id"],
    )

    print("\n========== YUGEN RETRIEVAL DEBUG ==========")
    print("Question:", state["question"])
    print("Retrieved chunks:", len(documents))

    for doc in documents:
        print("SOURCE:", doc.metadata.get("source"))
        print("CONTENT PREVIEW:", doc.page_content[:300])

    print("===========================================\n")

    return {"documents": documents}


def grade_node(state: AgentState) -> dict:
    """Evaluate whether retrieved documents answer the question."""

    documents = state.get("documents", [])

    if not documents:
        return {"relevance": "no"}

    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    messages = [
        SystemMessage(
            content="""You are a document relevance evaluator.

Determine whether the supplied documents contain useful information
to answer the user's question.

Return only:
yes
or
no

Do not use outside knowledge."""
        ),
        HumanMessage(
            content=(
                f"Question: {state['question']}\n\n"
                f"Documents:\n{context}"
            )
        ),
    ]

    response = get_model().invoke(messages)
    result = extract_text(response).strip().lower()

    return {
        "relevance": (
            "yes" if result.startswith("yes") else "no"
        )
    }



def choose_after_grading(
    state: AgentState,
) -> Literal["documents", "rewrite", "not_found"]:

    if state["relevance"] == "yes":
        return "documents"

    if state.get("retry_count", 0) < MAX_RETRIES:
        return "rewrite"

    return "not_found"



def rewrite_query_node(state: AgentState) -> dict:
    """Improve the document retrieval query."""

    messages = [
        SystemMessage(
            content="""Rewrite the question into a concise,
keyword-focused query for searching the uploaded documents.

Preserve the original meaning, dates, and named entities.
Do not invent dates or facts.
Return only the rewritten query."""
        ),
        HumanMessage(content=state["question"]),
    ]

    response = get_model().invoke(messages)
    rewritten = extract_text(response).strip()

    return {
        "search_query": rewritten or state["question"],
        "retry_count": state.get("retry_count", 0) + 1,
    }


def document_search_node(state: AgentState) -> dict:
    """Generate a grounded answer from retrieved documents."""

    result = generate_answer_from_documents(
        question=state["question"],
        documents=state["documents"],
        chat_history=state.get("chat_history", []),
    )

    return {
        "answer": result["answer"],
        "sources": result["sources"],
    }


def not_found_node(state: AgentState) -> dict:
    """Explain when the uploaded documents lack the answer."""

    return {
        "answer": (
            "I couldn't find enough relevant information in your "
            "uploaded documents to answer that question. "
            "You can try rephrasing it or upload a relevant document."
        ),
        "sources": [],
    }


# --------------------------------------------------
# WEB RESEARCH
# --------------------------------------------------

def web_search_node(state: AgentState) -> dict:
    """Search the web when the user requests current information."""

    results = search_web(state["question"])
    web_results = results.get("results", [])

    if not web_results:
        return {
            "answer": "I couldn't find relevant web results for this question.",
            "sources": [],
        }

    context = "\n\n".join(
        f"Title: {item.get('title', 'Untitled')}\n"
        f"URL: {item.get('url', '')}\n"
        f"Content: {item.get('content', '')}"
        for item in web_results
    )

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in state.get("chat_history", [])[-10:]
    )

    messages = [
        SystemMessage(
            content="""You are a helpful research assistant.

Answer using the supplied web search results.

Rules:
- Understand the user's actual question.
- Use conversation history to understand follow-ups.
- Ground factual claims in the supplied results.
- Do not invent information.
- Be conversational and clear.
- Keep the response proportional to the question.
- Mention uncertainty when the sources are insufficient."""
        ),
        HumanMessage(
            content=(
                f"Conversation history:\n{history_text}\n\n"
                f"Question: {state['question']}\n\n"
                f"Web search results:\n{context}"
            )
        ),
    ]

    response = get_model().invoke(messages)

    sources = [
        {
            "source": item.get("title", "Web result"),
            "url": item.get("url", ""),
            "score": item.get("score"),
        }
        for item in web_results
    ]

    return {
        "answer": extract_text(response),
        "sources": sources,
    }


# --------------------------------------------------
# BUILD GRAPH
# --------------------------------------------------


def build_graph():
    workflow = StateGraph(AgentState)

    # Register nodes.
    workflow.add_node("router", router_node)
    workflow.add_node("chat", chat_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("rewrite", rewrite_query_node)
    workflow.add_node("documents", document_search_node)
    workflow.add_node("not_found", not_found_node)
    workflow.add_node("web", web_search_node)

    # Start with intent classification.
    workflow.add_edge(START, "router")

    # Route the question according to its intent.
    workflow.add_conditional_edges(
        "router",
        choose_route,
        {
            "chat": "chat",
            "documents": "retrieve",
            "web": "web",
        },
    )

    # Chat and web answers finish the workflow.
    workflow.add_edge("chat", END)
    workflow.add_edge("web", END)

    # Document retrieval and relevance grading.
    workflow.add_edge("retrieve", "grade")

    workflow.add_conditional_edges(
    "grade",
    choose_after_grading,
    {
        "documents": "documents",
        "rewrite": "rewrite",
        "not_found": "not_found",
    },
)

    # Retry document retrieval with a rewritten query.
    workflow.add_edge("rewrite", "retrieve")

    # Finish after a grounded document answer.
    workflow.add_edge("documents", END)
    workflow.add_edge("not_found", END)

    return workflow.compile()


agent = build_graph()
