"""
PDF loading module.

Responsible for reading PDF files from disk and extracting raw text
content on a per-page basis using PyMuPDF (fitz).
"""

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class PageContent:
    """
    Represents the extracted text content of a single PDF page.

    Attributes:
        page_number: The 1-indexed page number within the PDF.
        text: The raw extracted text of the page.
    """
    page_number: int
    text: str


def load_pdf_pages(pdf_path: Path) -> list[PageContent]:
    """
    Extract text content from every page of a PDF file.

    Args:
        pdf_path: Path to the PDF file on disk.

    Returns:
        A list of PageContent objects, one per page, in page order.

    Raises:
        FileNotFoundError: If the given path does not exist.
        ValueError: If the file cannot be opened or parsed as a PDF.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pages: list[PageContent] = []

    try:
        with fitz.open(pdf_path) as document:
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                extracted_text = page.get_text("text")

                pages.append(
                    PageContent(
                        page_number=page_index + 1,
                        text=extracted_text.strip(),
                    )
                )
    except Exception as error:  # noqa: BLE001
        raise ValueError(
            f"Failed to read PDF '{pdf_path.name}': {error}"
        ) from error

    if not pages:
        raise ValueError(f"No pages could be extracted from '{pdf_path.name}'.")

    return pages