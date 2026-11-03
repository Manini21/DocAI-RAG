from sentence_transformers import SentenceTransformer
import streamlit as st


@st.cache_resource
def get_embedding_model():
    """
    Load and cache the embedding model.
    This runs only once.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        raise ValueError("Cannot generate embeddings for an empty list of texts.")

    model = get_embedding_model()

    try:
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    except Exception as error:
        raise RuntimeError(f"Failed to generate embeddings: {error}") from error


def embed_query(query: str) -> list[float]:
    if not query.strip():
        raise ValueError("Cannot generate embedding for an empty query.")

    return embed_texts([query])[0]