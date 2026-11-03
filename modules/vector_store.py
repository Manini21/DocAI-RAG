"""
Vector store module.

Manages the ChromaDB persistent client and collection used to store
and query document chunk embeddings.
"""

import chromadb
from chromadb.api.models.Collection import Collection

from config import CHROMA_COLLECTION_NAME, CHROMA_DIR
from modules.chunker import TextChunk


class VectorStore:
    """
    Thin wrapper around a persistent ChromaDB collection.

    Handles adding chunks with their embeddings and metadata, and
    querying the collection for the most relevant chunks given a
    query embedding.
    """

    def __init__(self) -> None:
        """Initialize the persistent ChromaDB client and collection."""
        self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._collection: Collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> int:
        """
        Add a batch of text chunks and their embeddings to the collection.

        Args:
            chunks: List of TextChunk objects to store.
            embeddings: List of embedding vectors corresponding to each chunk.

        Returns:
            The number of chunks successfully added.

        Raises:
            ValueError: If chunks and embeddings have mismatched lengths.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        if not chunks:
            return 0

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "filename": chunk.filename,
                "page": chunk.page,
                "chunk_id": chunk.chunk_id,
            }
            for chunk in chunks
        ]

        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict]:
        """
        Retrieve the most relevant chunks for a given query embedding.

        Args:
            query_embedding: The embedding vector of the user's question.
            top_k: The maximum number of results to return.

        Returns:
            A list of dictionaries, each containing 'text', 'filename',
            'page', 'chunk_id', and 'distance' keys, ordered by relevance
            (lowest distance first).
        """
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved_chunks: list[dict] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            retrieved_chunks.append(
                {
                    "text": document,
                    "filename": metadata.get("filename", "unknown"),
                    "page": metadata.get("page", -1),
                    "chunk_id": metadata.get("chunk_id", "unknown"),
                    "distance": distance,
                }
            )

        return retrieved_chunks

    def count(self) -> int:
        """
        Return the total number of chunks currently stored in the collection.

        Returns:
            The number of stored chunks.
        """
        return self._collection.count()

    def clear(self) -> None:
        """Delete and recreate the Chroma collection to clear all data."""
        self._client.delete_collection(CHROMA_COLLECTION_NAME)

        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def chunk_count(self) -> int:
        """
        Return the total number of stored chunks.
        """
        return self._collection.count()

    def document_count(self) -> int:
        """
        Return the number of unique PDF documents.
        """
        results = self._collection.get(include=["metadatas"])

        filenames = {
            metadata["filename"]
            for metadata in results["metadatas"]
            if metadata and "filename" in metadata
        }

        return len(filenames)