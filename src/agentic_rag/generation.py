
"""Generate answers grounded in retrieved document context."""

import uuid_shim  # Must load before LangChain imports.

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .retrieval import search_documents
from functools import lru_cache

MODEL_NAME = "gemini-3.1-flash-lite"


PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful knowledge assistant.

Use the conversation history to understand follow-up questions.
Answer using only the supplied document context.

Rules:
- Do not invent information outside the context.
- If the context does not contain the answer, say so clearly.
- Explain in simple, understandable language.
- Refer to relevant source documents when possible.

Conversation history:
{chat_history}

Document context:
{context}"""
    ),
    ("human", "{question}"),
])


@lru_cache(maxsize=1)
def get_llm():
    """Initialize and reuse the Gemini chat model."""

    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0,
        timeout=60,
    )


def generate_answer_from_documents(
    question: str,
    documents: list,
    chat_history: list[dict] | None = None,
) -> dict:
    """Generate an answer using already-retrieved documents."""

    if not question or not question.strip():
        raise ValueError("Please provide a question.")

    if not documents:
        return {
            "answer": "I couldn't find relevant information in the uploaded documents.",
            "sources": [],
        }

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"Content: {doc.page_content}"
        for doc in documents
    )

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in (chat_history or [])
    )

    chain = PROMPT | get_llm() | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question,
        "chat_history": history_text,
    })

    sources = [
        {
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page"),
            "content": doc.page_content,
        }
        for doc in documents
    ]

    return {
        "answer": answer,
        "sources": sources,
    }


def generate_answer(question: str, k: int = 4) -> dict:
    """Retrieve documents and generate a grounded answer."""

    if not question or not question.strip():
        raise ValueError("Please provide a question.")

    documents = search_documents(question, k=k)

    return generate_answer_from_documents(question, documents)
