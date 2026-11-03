"""
Retriever module.

Handles converting a user question into an embedding and fetching the
most relevant chunks from the vector store.
"""

from modules.embeddings import embed_query
from modules.vector_store import VectorStore


class Retriever:
    """
    Retrieves relevant document chunks for a given natural language query.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        """
        Initialize the retriever with a vector store instance.

        Args:
            vector_store: The VectorStore instance to query against.
        """
        self._vector_store = vector_store

    def retrieve(self, question: str, top_k: int) -> list[dict]:
        """
        Retrieve the top-k most relevant chunks for a given question.

        Args:
            question: The user's natural language question.
            top_k: The number of chunks to retrieve.

        Returns:
            A list of retrieved chunk dictionaries, ordered by relevance.

        Raises:
            ValueError: If the question is empty or top_k is not positive.
        """
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        query_embedding = embed_query(question)
        return self._vector_store.query(query_embedding=query_embedding, top_k=top_k)