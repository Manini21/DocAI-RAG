"""
Configuration module for the Personal Knowledge Base AI application.

Centralizes all constants, paths, and environment variable loading so
that other modules do not need to duplicate configuration logic.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from a .env file if present.
load_dotenv()


# --------------------------------------------------------------------------
# API Configuration
# --------------------------------------------------------------------------
# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.2
# --------------------------------------------------------------------------
# Model Configuration
# --------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --------------------------------------------------------------------------
# Chunking Configuration
# --------------------------------------------------------------------------
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 64

# --------------------------------------------------------------------------
# Retrieval Configuration
# --------------------------------------------------------------------------
TOP_K_RESULTS: int = 5

# --------------------------------------------------------------------------
# Path Configuration
# --------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
UPLOAD_DIR: Path = DATA_DIR / "uploaded_pdfs"
CHROMA_DIR: Path = BASE_DIR / "chroma_db"

# Ensure required directories exist at import time.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# ChromaDB Configuration
# --------------------------------------------------------------------------
CHROMA_COLLECTION_NAME: str = "personal_knowledge_base"