"""
Helper utilities module.

Contains small, reusable functions that support the Streamlit UI and
pipeline modules, such as saving uploaded files and formatting citations.
"""

from pathlib import Path

from streamlit.runtime.uploaded_file_manager import UploadedFile


def save_uploaded_file(uploaded_file: UploadedFile, destination_dir: Path) -> Path:
    """
    Save a Streamlit-uploaded file to disk.

    Args:
        uploaded_file: The file object provided by st.file_uploader.
        destination_dir: The directory in which to save the file.

    Returns:
        The full path to the saved file on disk.

    Raises:
        OSError: If the file cannot be written to disk.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / uploaded_file.name

    try:
        with open(destination_path, "wb") as output_file:
            output_file.write(uploaded_file.getbuffer())
    except OSError as error:
        raise OSError(
            f"Failed to save uploaded file '{uploaded_file.name}': {error}"
        ) from error

    return destination_path


def format_citation(source: dict) -> str:
    """
    Format a source metadata dictionary into a human-readable citation label.

    Args:
        source: A dictionary containing at least 'filename' and 'page' keys.

    Returns:
        A formatted citation string, e.g. "report.pdf, page 4".
    """
    filename = source.get("filename", "unknown file")
    page = source.get("page", "unknown")
    return f"{filename}, page {page}"