"""
Text chunking module.

Splits page-level text into smaller overlapping chunks suitable for
embedding, using LangChain's RecursiveCharacterTextSplitter.
"""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE
from modules.pdf_loader import PageContent


@dataclass
class TextChunk:
    """
    Represents a single chunk of text ready for embedding.

    Attributes:
        chunk_id: A unique identifier for this chunk within the document.
        filename: Name of the source PDF file.
        page: The page number the chunk originated from.
        text: The chunk's text content.
    """
    chunk_id: str
    filename: str
    page: int
    text: str


def split_pages_into_chunks(
    pages: list[PageContent],
    filename: str,
) -> list[TextChunk]:
    """
    Split a list of extracted PDF pages into overlapping text chunks.

    Args:
        pages: List of PageContent objects extracted from a PDF.
        filename: The original filename of the PDF, used for metadata
            and generating unique chunk identifiers.

    Returns:
        A list of TextChunk objects ready to be embedded and stored.

    Raises:
        ValueError: If no chunks could be produced from the given pages.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[TextChunk] = []
    chunk_counter = 0

    for page in pages:
        if not page.text:
            continue

        page_chunks = splitter.split_text(page.text)

        for chunk_text in page_chunks:
            cleaned_text = chunk_text.strip()
            if not cleaned_text:
                continue

            chunk_counter += 1
            chunk_id = f"{filename}::page{page.page_number}::chunk{chunk_counter}"

            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    filename=filename,
                    page=page.page_number,
                    text=cleaned_text,
                )
            )

    if not chunks:
        raise ValueError(
            f"No text chunks could be generated for file '{filename}'. "
            "The PDF may be empty or contain only images."
        )

    return chunks