"""
Run with:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from config import UPLOAD_DIR, TOP_K_RESULTS
from modules.rag_pipeline import RAGPipeline
from utils.helpers import save_uploaded_file, format_citation
from config import EMBEDDING_MODEL, LLM_MODEL


# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="DocAI RAG",
    page_icon="📚",
    layout="wide",
)

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    """
    Create (or fetch a cached) instance of the RAG pipeline.

    Streamlit's cache_resource ensures we do not reinitialize the
    embedding model / vector store connection on every rerun.

    Returns:
        RAGPipeline: A ready-to-use pipeline instance.
    """
    return RAGPipeline()


def render_header() -> None:
    """Render the application title and short description."""
    st.title("📚 Personal Knowledge Base AI")
    st.caption(
        "Upload PDFs, build your own knowledge base, and ask questions "
        "answered directly from your documents — with citations."
    )
    st.divider()


def render_upload_section(pipeline: RAGPipeline) -> None:
    """
    Render the PDF upload section and handle the 'Process PDFs' action.

    Args:
        pipeline: The RAG pipeline used to process and index PDFs.
    """
    st.subheader("1. Upload PDFs")

    uploaded_files = st.file_uploader(
        label="Select one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDFs at once.",
    )

    process_clicked = st.button("Process PDFs", type="primary")

    if process_clicked:
        if not uploaded_files:
            st.warning("Please upload at least one PDF before processing.")
            return

        with st.spinner("Processing PDFs. This may take a moment..."):
            total_chunks = 0
            for uploaded_file in uploaded_files:
                try:
                    saved_path = save_uploaded_file(uploaded_file, UPLOAD_DIR)
                    chunks_added = pipeline.ingest_pdf(saved_path)
                    total_chunks += chunks_added

                except Exception as error:  # noqa: BLE001
                    st.error(
                        f"Failed to process '{uploaded_file.name}': {error}"
                    )

            if total_chunks > 0:
                st.session_state.indexed = True
                st.session_state.doc_count += len(uploaded_files)
                st.session_state.chunk_count += total_chunks

                st.success("Successfully indexed!")

    if st.button("🗑 Clear Knowledge Base"):
        pipeline.clear_database()
        st.success("Knowledge base cleared.")
        st.rerun()

    st.divider()

def render_knowledge_base_info() -> None:
    """Display indexed knowledge base information."""

    if not st.session_state.indexed:
        return

    st.markdown("### 📊 Knowledge Base")

    st.info(
        f"""
📄 **Documents:** {st.session_state.doc_count}

🧩 **Chunks:** {st.session_state.chunk_count}

🧠 **Embedding:** MiniLM-L6-v2

🤖 **LLM:** Llama 3.3 70B
"""
    )


def render_question_section(pipeline: RAGPipeline) -> None:
    """
    Render the question input section and handle the 'Ask' action.

    Args:
        pipeline: The RAG pipeline used to answer questions.
    """
    st.subheader("2. Ask a Question")

    question = st.text_input(
        label="Enter your question",
        placeholder="e.g. What are the main conclusions of the report?",
    )

    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked:
        if not question or not question.strip():
            st.warning("Please enter a question before clicking Ask.")
            return

        with st.spinner("Retrieving relevant sources and generating answer..."):
            try:
                result = pipeline.answer_question(
                    question=question,
                    top_k=TOP_K_RESULTS,
                )
            except Exception as error:  # noqa: BLE001
                st.error(f"Failed to answer question: {error}")
                return

        render_sources(result["sources"])
        render_answer(result["answer"])


def render_sources(sources: list[dict]) -> None:
    """
    Render the list of retrieved source chunks.

    Args:
        sources: List of metadata dictionaries describing each source chunk.
    """
    st.subheader("Retrieved Sources")

    if not sources:
        st.info("No relevant sources were found in the knowledge base.")
        return

    for index, source in enumerate(sources, start=1):
        citation_label = format_citation(source)
        with st.expander(f"Source {index}: {citation_label}"):
            st.write(source.get("text_preview", "No preview available."))


def render_answer(answer: str) -> None:
    """
    Render the final answer generated by the LLM.

    Args:
        answer: The generated answer text, including citation markers.
    """
    st.subheader("Answer")
    st.markdown(answer)


def main():
    render_header()

    pipeline = get_pipeline()

    left, right = st.columns(2)

    with left:
        render_upload_section(pipeline)
        #st.divider()
        render_knowledge_base_info()

    with right:
        render_question_section(pipeline)

if __name__ == "__main__":
    main()