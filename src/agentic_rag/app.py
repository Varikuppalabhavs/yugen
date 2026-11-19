"""Yugen — editorial interface for the persistent Agentic RAG assistant."""

import sys
import uuid_shim  # Must load before LangChain imports.

import base64
import hashlib
import re
import secrets
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

import streamlit as st
from streamlit_cookies_controller import CookieController

from src.agentic_rag.generation import get_llm
from src.agentic_rag.graph import agent
from src.agentic_rag.ingest import iter_document_chunks
from src.agentic_rag.retrieval import index_documents, delete_document
from src.agentic_rag.memory import (
    add_message,
    load_chat_history,
    save_chat_history,
    clear_chat_history,
)

# -----------------------------------------------------------------------------
# Configuration and brand assets
# -----------------------------------------------------------------------------

UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
SUPPORTED_TYPES = ["pdf", "txt", "md", "markdown"]
ASSETS_DIR = PROJECT_ROOT / "assets"
MASCOT_PATH = ASSETS_DIR / "yugen_pixel.png"
HERO_PATH = ASSETS_DIR / "yugen_cherry_arch.png"
CSS_PATH = ASSETS_DIR / "yugen_theme.css"

st.set_page_config(
    page_title="Yugen — AI Knowledge Assistant",
    page_icon=str(MASCOT_PATH) if MASCOT_PATH.exists() else "Y",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_theme() -> None:
    """Load Yugen's fixed editorial theme and embed the local hero artwork."""
    if not CSS_PATH.exists():
        st.warning("Yugen's theme file is missing. Please copy the assets folder into your project root.")
        return
    css = CSS_PATH.read_text(encoding="utf-8")
    if HERO_PATH.exists():
        hero_b64 = base64.b64encode(HERO_PATH.read_bytes()).decode("ascii")
        css = css.replace("__HERO_IMAGE__", hero_b64)
    else:
        css = css.replace("url('data:image/png;base64,__HERO_IMAGE__')", "none")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


inject_theme()

# -----------------------------------------------------------------------------
# Persistent browser session — intentionally unchanged in behavior
# -----------------------------------------------------------------------------

controller = CookieController(key="agentic_rag_cookies")
browser_cookies = controller.getAll()


if "cookie_initialized" not in st.session_state:
    time.sleep(1)
    browser_cookies = controller.getAll()
    st.session_state.cookie_initialized = True

if browser_cookies is None:
    print("YUGEN DEBUG: browser_cookies is None")
    st.error("Could not read browser cookies. Please reload the page.")
    st.stop()

session_id = browser_cookies.get("session_id")
if not session_id:
    session_id = secrets.token_urlsafe(32)
    controller.set("session_id", session_id)
    time.sleep(1)
    browser_cookies = controller.getAll()
    session_id = (browser_cookies or {}).get("session_id")

if not session_id:
    st.warning("Your browser session is being created. Please reload once.")
    st.stop()

if not isinstance(session_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{20,100}", session_id):
    st.error("Invalid browser session.")
    st.stop()
print("YUGEN DEBUG: PASSED SESSION VALIDATION")

SESSION_ID = session_id
if st.session_state.get("active_session_id") != SESSION_ID:
    st.session_state.active_session_id = SESSION_ID
    st.session_state.messages = load_chat_history(SESSION_ID)

SESSION_UPLOAD_DIR = UPLOADS_DIR / SESSION_ID
SESSION_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
st.session_state.setdefault("yugen_view", "workspace")
st.session_state.setdefault("messages", load_chat_history(SESSION_ID))

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def persist_history() -> None:
    save_chat_history(SESSION_ID, st.session_state.messages)


def is_casual_message(message: str) -> bool:
    text = message.lower().strip().rstrip("!.,?")
    casual_phrases = {
        "hi", "hello", "hey", "hii", "hiii", "good morning",
        "good afternoon", "good evening", "how are you", "how are you doing",
        "what's up", "whats up", "thank you", "thanks", "thankss",
        "bye", "goodbye", "see you", "who are you", "what are you",
        "are you there", "you there", "and you", "and u", "lol", "lmao",
    }
    if text in casual_phrases:
        return True
    casual_markers = ["hehe", "haha", "hahaha", "lol", "lmao", "omg", "bruh", "bro"]
    return len(text.split()) <= 12 and any(marker in text for marker in casual_markers)


def get_saved_documents() -> list[dict]:
    saved_documents = []
    if not SESSION_UPLOAD_DIR.exists():
        return saved_documents
    for path in sorted(SESSION_UPLOAD_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".pdf", ".txt", ".md", ".markdown"}:
            continue
        saved_documents.append({
            "name": path.name,
            "size": path.stat().st_size,
            "indexed": (path.parent / ".indexed").exists(),
            "path": path,
        })
    return saved_documents


def delete_uploaded_document(document_path: Path) -> int:
    document_path = document_path.resolve()
    session_root = SESSION_UPLOAD_DIR.resolve()
    if session_root not in document_path.parents:
        raise ValueError("Invalid document path.")
    if not document_path.is_file():
        return 0
    deleted_chunks = delete_document(source=str(document_path), session_id=SESSION_ID)
    import shutil
    shutil.rmtree(document_path.parent)
    return deleted_chunks


def show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources · {len(sources)}"):
        for index, source in enumerate(sources, start=1):
            name = source.get("source", "Unknown source")
            url = source.get("url")
            page = source.get("page")
            if url:
                st.markdown(f"<div class='yugen-source'><strong>{index:02d}</strong> &nbsp; <a href='{url}' target='_blank'>{name}</a></div>", unsafe_allow_html=True)
            else:
                page_label = f" · page {page + 1}" if page is not None else ""
                st.markdown(f"<div class='yugen-source'><strong>{index:02d}</strong> &nbsp; {name}{page_label}</div>", unsafe_allow_html=True)


def mascot_avatar() -> str | None:
    return str(MASCOT_PATH) if MASCOT_PATH.exists() else None


def render_brand() -> None:
    if MASCOT_PATH.exists():
        mascot = base64.b64encode(MASCOT_PATH.read_bytes()).decode("ascii")
        img = f"<img src='data:image/png;base64,{mascot}' alt='Yugen pixel mascot' />"
    else:
        img = "<span style='font-size:30px'>Y</span>"
    st.markdown(
        f"<div class='yugen-brand'>{img}<div><div class='word'>yugen</div><div class='sub'>AI KNOWLEDGE ASSISTANT</div></div></div>",
        unsafe_allow_html=True,
    )


def process_uploaded_files(uploaded_files) -> None:
    if not uploaded_files:
        st.warning("Choose at least one document to upload.")
        return
    total_chunks = 0
    with st.spinner("Reading and indexing your documents…"):
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            file_hash = hashlib.sha256(file_bytes).hexdigest()[:12]
            original_name = Path(uploaded_file.name).name
            file_stem = Path(original_name).stem
            upload_dir = SESSION_UPLOAD_DIR / f"{file_stem}_{file_hash}"
            target_path = upload_dir / original_name
            success_marker = upload_dir / ".indexed"
            if success_marker.exists():
                st.info(f"Already indexed: {original_name}")
                continue
            try:
                upload_dir.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(file_bytes)
                chunk_batch = []
                indexed_count = 0

                for chunk in iter_document_chunks(upload_dir):
                    chunk_batch.append(chunk)

                    if len(chunk_batch) == 10:
                        indexed_count += index_documents(
                            chunk_batch,
                            session_id=SESSION_ID,
                        )
                        chunk_batch.clear()

                if chunk_batch:
                    indexed_count += index_documents(
                        chunk_batch,
                        session_id=SESSION_ID,
                    )

                if indexed_count == 0:
                    st.warning(f"No readable text found in {original_name}.")
                    continue

                total_chunks += indexed_count
                success_marker.write_text("success", encoding="utf-8")
                st.success(f"{original_name} · {indexed_count} chunks indexed")
            except Exception as error:
                error_text = str(error).lower()
                if "embedding_quota_exceeded" in error_text:
                    st.error(f"Gemini embedding quota reached while processing {original_name}. Please wait and retry.")
                elif "429" in error_text or "resource_exhausted" in error_text:
                    st.error(f"The AI service is temporarily rate-limited. Please retry {original_name} later.")
                else:
                    st.error(f"Could not process {original_name}: {error}")
    if total_chunks:
        st.success(f"Your knowledge base has been updated · {total_chunks} new chunks")


# -----------------------------------------------------------------------------
# Sidebar — compact identity and navigation
# -----------------------------------------------------------------------------

with st.sidebar:
    render_brand()
    st.markdown("<div class='yugen-nav-label'>YOUR WORKSPACE</div>", unsafe_allow_html=True)
    st.caption("A quieter space for curious minds.")
    st.divider()
    st.markdown("<div class='yugen-nav-label'>SESSION</div>", unsafe_allow_html=True)
    st.caption(f"{len(st.session_state.messages)} messages saved in this browser session")
    st.markdown("<div class='yugen-muted'>Your files and conversation stay separated by browser session.</div>", unsafe_allow_html=True)
    st.divider()
    if st.button("Clear conversation", key="clear_history", use_container_width=True):
        clear_chat_history(SESSION_ID)
        st.session_state.messages = []
        st.session_state.yugen_view = "workspace"
        st.rerun()
    st.caption("Clearing the conversation does not remove your documents.")

# -----------------------------------------------------------------------------
# Top navigation — real view switching via Streamlit session state
# -----------------------------------------------------------------------------

nav1, nav2, nav3, nav_spacer = st.columns([1.15, 1.15, 1.0, 4.7])
with nav1:
    if st.button("WORKSPACE", key="nav_workspace", use_container_width=True):
        st.session_state.yugen_view = "workspace"
        st.rerun()
with nav2:
    if st.button("KNOWLEDGE", key="nav_knowledge", use_container_width=True):
        st.session_state.yugen_view = "knowledge"
        st.rerun()
with nav3:
    if st.button("HISTORY", key="nav_history", use_container_width=True):
        st.session_state.yugen_view = "history"
        st.rerun()

view = st.session_state.yugen_view
st.markdown(
    f"<style>.st-key-nav_{view} button{{border-bottom:2px solid #963F43!important;color:#963F43!important;}}</style>",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Workspace
# -----------------------------------------------------------------------------

if view == "workspace":
    if not st.session_state.messages:
        st.markdown(
            "<section class='yugen-hero'><div class='yugen-hero-copy'>"
            ""
            "<h1 class='yugen-title'>Ask Yugen<span class='dot'>.</span></h1>"
            "<div class='yugen-subtitle'>A quieter space for curious minds.</div>"
            "<div class='yugen-note'>Explore your documents, investigate a question, or follow an idea wherever it leads.</div>"
            "</div></section>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div class='yugen-section-title'>YOUR CONVERSATION</div>", unsafe_allow_html=True)
        for message in st.session_state.messages:
            avatar = mascot_avatar() if message["role"] == "assistant" else None
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
                show_sources(message.get("sources", []))

    question = st.chat_input("Ask Yugen about your documents or beyond…")
    if question:
        previous_history = st.session_state.messages.copy()
        add_message(st.session_state.messages, "user", question)
        persist_history()
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant", avatar=mascot_avatar()):
            try:
                if is_casual_message(question):
                    with st.spinner("Yugen is thinking…"):
                        history_text = "\n".join(
                            f"{message['role']}: {message['content']}"
                            for message in previous_history[-10:]
                        )
                        response = get_llm().invoke(
                            f"""You are Yugen, a warm, natural conversational assistant.
Be thoughtful, clear and conversational. Understand casual language and follow-ups.
Keep casual replies concise. Do not search the web or mention sources for casual greetings.
Previous conversation:\n{history_text}\n\nUser: {question}\n\nYugen:"""
                        )
                        content = response.content
                        if isinstance(content, str):
                            answer = content
                        elif isinstance(content, list):
                            answer = "\n".join(
                                block.get("text", "") for block in content
                                if isinstance(block, dict) and block.get("type") == "text"
                            )
                        else:
                            answer = str(content)
                        sources = []
                else:
                    with st.spinner("Yugen is researching…"):
                        result = agent.invoke({
                            "question": question,
                            "session_id": SESSION_ID,
                            "intent": "",
                            "search_query": "",
                            "documents": [],
                            "relevance": "",
                            "retry_count": 0,
                            "chat_history": previous_history,
                            "answer": "",
                            "sources": [],
                        })
                        answer = result["answer"]
                        sources = result.get("sources", [])
                st.markdown(answer)
                show_sources(sources)
                add_message(st.session_state.messages, "assistant", answer, sources)
                persist_history()
            except Exception as error:
                error_message = "Something went wrong while preparing your answer. Please try again."
                st.error(error_message)
                add_message(st.session_state.messages, "assistant", error_message)
                persist_history()
                print(f"Yugen agent error: {error!r}")

# -----------------------------------------------------------------------------
# Knowledge — uploads, indexing, listing and deletion
# -----------------------------------------------------------------------------

elif view == "knowledge":
    st.markdown("<div class='yugen-section-title'>YOUR PERSONAL LIBRARY</div>", unsafe_allow_html=True)
    st.title("Knowledge base")
    st.write("Bring your documents into Yugen. Upload, index and manage the files used for grounded answers.")
    uploaded_files = st.file_uploader(
        "Add documents", type=SUPPORTED_TYPES, accept_multiple_files=True,
        help="Supported formats: PDF, TXT, MD and Markdown.",
    )
    if st.button("Process documents", key="process_documents", use_container_width=True):
        process_uploaded_files(uploaded_files)

    st.markdown("<div class='yugen-section-title'>INDEXED DOCUMENTS</div>", unsafe_allow_html=True)
    saved_documents = get_saved_documents()
    if not saved_documents:
        st.info("Your library is empty. Upload a PDF, TXT or Markdown file to begin.")
    else:
        for document in saved_documents:
            left, middle, right = st.columns([4.2, 1.2, 1.0])
            with left:
                size_kb = document["size"] / 1024
                st.markdown(
                    f"<div class='yugen-doc-card'><div class='yugen-doc-name'>{document['name']}</div>"
                    f"<div class='yugen-muted'>{size_kb:.1f} KB</div></div>",
                    unsafe_allow_html=True,
                )
            with middle:
                st.caption("INDEXED" if document["indexed"] else "PENDING")
            with right:
                if st.button("Remove", key=f"remove_doc_{document['path'].parent.name}", use_container_width=True):
                    try:
                        delete_uploaded_document(document["path"])
                        st.success("Document removed from your library.")
                        st.rerun()
                    except Exception as error:
                        st.error(f"Could not remove document: {error}")

# -----------------------------------------------------------------------------
# History — chronological view of the one saved conversation for this session
# -----------------------------------------------------------------------------

elif view == "history":
    st.markdown("<div class='yugen-section-title'>A RECORD OF YOUR EXPLORATIONS</div>", unsafe_allow_html=True)
    st.title("Conversation history")
    st.caption("This view shows the saved conversation for this browser session. Multiple separate chat threads are not enabled yet.")
    if not st.session_state.messages:
        st.info("No messages yet. Start a conversation in Workspace and it will appear here.")
    else:
        for index, message in enumerate(st.session_state.messages, start=1):
            role_label = "YOU" if message["role"] == "user" else "YUGEN"
            with st.container(border=True):
                st.markdown(f"<div class='yugen-section-title'>{index:02d} &nbsp; / &nbsp; {role_label}</div>", unsafe_allow_html=True)
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    show_sources(message.get("sources", []))
        if st.button("Return to workspace", key="history_to_workspace"):
            st.session_state.yugen_view = "workspace"
            st.rerun()
