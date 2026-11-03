"""
RAG pipeline module.

Orchestrates the full Retrieval-Augmented Generation workflow:
ingesting PDFs into the vector store, and answering questions using
retrieved context and the LLM.
"""

from pathlib import Path

from modules.chunker import split_pages_into_chunks
from modules.embeddings import embed_texts
from modules.llm import generate_answer
from modules.pdf_loader import load_pdf_pages
from modules.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from modules.retriever import Retriever
from modules.vector_store import VectorStore


class RAGPipeline:
    """
    High-level orchestrator for the Personal Knowledge Base AI system.

    Combines PDF loading, chunking, embedding, vector storage, retrieval,
    prompt construction, and LLM generation into a simple interface.
    """

    def __init__(self) -> None:
        """Initialize the vector store and retriever used by the pipeline."""
        self._vector_store = VectorStore()
        self._retriever = Retriever(self._vector_store)

    def ingest_pdf(self, pdf_path: Path) -> int:
        """
        Process a single PDF file: extract text, chunk it, embed it, and
        store it in the vector database.

        Args:
            pdf_path: Path to the PDF file to ingest.

        Returns:
            The number of chunks added to the vector store.

        Raises:
            ValueError: If the PDF contains no extractable text.
            RuntimeError: If embedding generation fails.
        """
        pages = load_pdf_pages(pdf_path)
        chunks = split_pages_into_chunks(pages, filename=pdf_path.name)

        chunk_texts = [chunk.text for chunk in chunks]
        chunk_embeddings = embed_texts(chunk_texts)

        added_count = self._vector_store.add_chunks(
            chunks=chunks,
            embeddings=chunk_embeddings,
        )

        return added_count

    def get_chunk_count(self) -> int:
        """
        Return total indexed chunks.
        """
        return self._vector_store.chunk_count()

    def get_document_count(self) -> int:
        """
        Return total indexed documents.
        """
        return self._vector_store.document_count()

    def clear_database(self) -> None:
        """Clear all indexed documents and chunks from the vector store."""
        self._vector_store.clear()

    def answer_question(self, question: str, top_k: int) -> dict:
        """
        Answer a user's question using retrieved context from the
        knowledge base.

        Args:
            question: The user's natural language question.
            top_k: The number of relevant chunks to retrieve.

        Returns:
            A dictionary with two keys:
                - "answer": The generated answer text with citations.
                - "sources": A list of source metadata dictionaries used
                  to generate the answer.

        Raises:
            ValueError: If the question is empty.
        """
        retrieved_chunks = self._retriever.retrieve(question=question, top_k=top_k)

        if not retrieved_chunks:
            return {
                "answer": (
                    "I don't have enough information in the knowledge base "
                    "to answer this question. Please upload relevant PDFs first."
                ),
                "sources": [],
            }

        user_prompt = build_user_prompt(question, retrieved_chunks)
        answer = generate_answer(SYSTEM_PROMPT, user_prompt)

        sources = [
            {
                "filename": chunk["filename"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "text_preview": _build_preview(chunk["text"]),
            }
            for chunk in retrieved_chunks
        ]

        return {"answer": answer, "sources": sources}


def _build_preview(text: str, max_length: int = 300) -> str:
    """
    Build a shortened preview of a chunk's text for display purposes.

    Args:
        text: The full chunk text.
        max_length: Maximum number of characters to include in the preview.

    Returns:
        A truncated version of the text, with an ellipsis if shortened.
    """
    cleaned = text.strip().replace("\n", " ")
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[:max_length].rstrip() + "..."