"""
Prompt builder module.

Constructs the system and user prompts sent to the LLM, combining the
user's question with retrieved context chunks and instructing the
model to cite its sources.
"""

SYSTEM_PROMPT = """
You are an intelligent AI assistant that answers questions using the user's uploaded documents.

Guidelines:
- Treat the retrieved documents as the primary source of truth.
- When the answer exists in the documents, answer using that information and cite the relevant sources.
- If the user asks for analysis, recommendations, opinions, comparisons, brainstorming, explanations, or advice, you may combine the retrieved document information with your general knowledge.
- Clearly distinguish between facts taken from the documents and your own recommendations or reasoning.
- Never invent or misrepresent information that is not supported by the documents.
- If the documents do not contain enough information to answer a factual question, state that clearly instead of guessing.
- Keep responses accurate, well-structured, and concise unless the user requests more detail.
- When using information from the documents, include citations in the format [filename, page X].

Response Style:
- Answer the user's question directly in the first sentence.
- Avoid repeatedly saying "Based on the provided context."
- Use the uploaded documents as supporting evidence rather than repeating that they are the source.
- When providing general knowledge or recommendations, clearly indicate that they are based on general knowledge and professional reasoning.
- Keep the tone natural, confident, and conversational.
"""


def build_context_block(retrieved_chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a single context string for the prompt.

    Args:
        retrieved_chunks: List of chunk dictionaries containing 'text',
            'filename', and 'page' keys.

    Returns:
        A formatted string containing all context chunks, each labeled
        with its source for easy citation by the LLM.
    """
    context_parts: list[str] = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        filename = chunk.get("filename", "unknown")
        page = chunk.get("page", "unknown")
        text = chunk.get("text", "")

        context_parts.append(
            f"[Context {index} | Source: {filename}, Page {page}]\n{text}"
        )

    return "\n\n".join(context_parts)


def build_user_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Build the full user prompt combining context and the question.

    Args:
        question: The user's natural language question.
        retrieved_chunks: List of retrieved chunk dictionaries.

    Returns:
        A formatted prompt string ready to send to the LLM.
    """
    context_block = build_context_block(retrieved_chunks)

    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context above. "
        "Cite sources in the format [filename, page X]."
    )