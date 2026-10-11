"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface

API contract with the backend (agree this with Pipeline Engineer
before building anything):

  ingest(file_paths: list[Path]) -> IngestionResult
  list_documents() -> list[dict]
  get_document_chunks(source: str) -> list[DocumentChunk]
  chat(query: str, history: list[dict], filters: dict) -> AgentResponse

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag_agent.config import LLMFactory
from rag_agent.agent.state import AgentResponse
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------
# Use st.cache_resource for objects that should persist across reruns
# and be shared across all user sessions. This prevents re-initialising
# ChromaDB and reloading the embedding model on every button click.


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """
    Return the singleton VectorStoreManager.

    Cached so ChromaDB connection is initialised once per application
    session, not on every Streamlit rerun.
    """
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_llm():
    """Return the cached Groq language model."""
    return LLMFactory().create()


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """
    Initialise all st.session_state keys on first run.

    Must be called at the top of main() before any UI is rendered.
    Without this, state keys referenced in callbacks will raise KeyError.

    Interview talking point: Streamlit reruns the entire script on every
    user interaction. session_state is the mechanism for persisting data
    (chat history, ingestion results) across reruns.
    """
    defaults = {
        "chat_history": [],           # list of {"role": "user"|"assistant", "content": str}
        "ingested_documents": [],     # list of dicts from list_documents()
        "selected_document": None,    # source filename currently in viewer
        "last_ingestion_result": None,
        "thread_id": "default-session",  # LangGraph conversation thread
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel (Sidebar)
# ---------------------------------------------------------------------------


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """
    Render the document ingestion panel in the sidebar.

    Allows multi-file upload of PDF and Markdown files. Displays
    ingestion results (chunks added, duplicates skipped, errors).
    Updates the ingested documents list after successful ingestion.

    Parameters
    ----------
    store : VectorStoreManager
    chunker : DocumentChunker
    """
    st.sidebar.header("📂 Corpus Ingestion")

    uploaded_files = st.sidebar.file_uploader(
        "Upload Markdown documents",
        type=["md"],
        accept_multiple_files=True,
    )

    if st.sidebar.button(
        "Ingest Documents",
        disabled=not uploaded_files,
    ):
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            all_chunks = []

            for uploaded_file in uploaded_files:
                file_path = Path(temp_dir) / uploaded_file.name
                file_path.write_bytes(uploaded_file.getvalue())

                chunks = chunker.chunk_file(file_path)
                all_chunks.extend(chunks)

            result = store.ingest(all_chunks)

            st.sidebar.success(
                f"{result.ingested} chunks added, "
                f"{result.skipped} duplicates skipped."
            )

            if result.errors:
                st.sidebar.error(str(result.errors))

    documents = store.list_documents()

    st.sidebar.subheader("Stored Documents")

    for document in documents:
        st.sidebar.write(
            f"{document['source']} — "
            f"{document['chunk_count']} chunks"
        )


def render_corpus_stats(store: VectorStoreManager) -> None:
    """
    Render a compact corpus health summary in the sidebar.

    Shows total chunks, topics covered, and whether bonus topics
    are present. Used during Hour 3 to demonstrate corpus completeness.

    Parameters
    ----------
    store : VectorStoreManager
    """
    stats = store.get_collection_stats()

    st.sidebar.divider()
    st.sidebar.subheader("Corpus Statistics")
    st.sidebar.metric("Total Chunks", stats["total_chunks"])

    if stats["topics"]:
        st.sidebar.write("Topics:", ", ".join(stats["topics"]))
    else:
        st.sidebar.write("No topics available yet.")

    if stats["bonus_topics_present"]:
        st.sidebar.success("Bonus topics present")


# ---------------------------------------------------------------------------
# Document Viewer Panel (Centre)
# ---------------------------------------------------------------------------


def render_document_viewer(store: VectorStoreManager) -> None:
    """
    Render the document viewer in the main centre column.

    Displays a selectable list of ingested documents. When a document
    is selected, renders its chunk content in a scrollable pane.

    Parameters
    ----------
    store : VectorStoreManager
    """
    st.subheader("📄 Document Viewer")

    documents = store.list_documents()

    if not documents:
        st.info("Ingest documents using the sidebar to view content here.")
        return

    selected_source = st.selectbox(
        "Select document",
        options=[doc["source"] for doc in documents],
    )

    st.session_state["selected_document"] = selected_source

    chunks = store.get_document_chunks(selected_source)

    with st.container(height=400):
        for index, chunk in enumerate(chunks, start=1):
            st.markdown(f"**Chunk {index}**")
            st.caption(
                f"{chunk.metadata.topic} | "
                f"{chunk.metadata.difficulty} | "
                f"{chunk.metadata.type}"
            )
            st.write(chunk.chunk_text)
            st.divider()

    st.caption(f"Total chunks in document: {len(chunks)}")


# ---------------------------------------------------------------------------
# Chat Interface Panel (Right)
# ---------------------------------------------------------------------------


def render_chat_interface(store: VectorStoreManager, llm) -> None:
    """
    Render the chat interface in the right column.

    Supports multi-turn conversation with the LangGraph agent.
    Displays source citations with every response.
    Shows a clear "no relevant context" indicator when the
    hallucination guard fires.

    Parameters
    ----------
    graph : CompiledStateGraph
        The compiled LangGraph agent from get_compiled_graph().
    """
    st.subheader("💬 Interview Prep Chat")

    # Filters
    col_topic, col_diff = st.columns(2)
    with col_topic:
        # TODO: st.selectbox for topic filter
        pass
    with col_diff:
        # TODO: st.selectbox for difficulty filter
        pass

    # Chat history display
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("📎 Sources"):
                        for source in message["sources"]:
                            st.caption(source)
                if message.get("no_context_found"):
                    st.warning("⚠️ No relevant content found in corpus.")

    # Chat input
    st.subheader("RAG Question & Answer")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("Sources", expanded=True):
                    for source in message["sources"]:
                        st.caption(source)

    question = st.chat_input("Ask a question about your documents...")

    if question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                retrieved = store.query(question)

                if not retrieved:
                    answer = "No relevant information found in the uploaded documents."
                    sources = []
                else:
                    context = "\n\n".join(
                        chunk.chunk_text for chunk in retrieved
                    )

                    prompt = (
                        "Answer the question using only the provided context. "
                        "If the context does not contain the answer, say so.\n\n"
                        f"Context:\n{context}\n\n"
                        f"Question: {question}"
                    )

                    response = llm.invoke(prompt)
                    answer = response.content
                    sources = [chunk.to_citation() for chunk in retrieved]

                st.markdown(answer)

                if sources:
                    with st.expander("Sources", expanded=True):
                        for source in sources:
                            st.caption(source)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Application entry point.

    Sets page config, initialises session state, instantiates shared
    resources, and renders all UI panels.

    Run with: uv run streamlit run src/rag_agent/ui/app.py
    """
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption(
        "RAG-powered interview preparation — built with LangChain, LangGraph, and ChromaDB"
    )

    initialise_session_state()

    # Instantiate shared backend resources
    store = get_vector_store()
    chunker = get_chunker()
    llm = get_llm()

    # Sidebar
    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    # Main content area — two columns
    viewer_col, chat_col = st.columns([1, 1], gap="large")

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(store, llm)


if __name__ == "__main__":
    main()
