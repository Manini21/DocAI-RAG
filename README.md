# Personal Knowledge Base AI

A production-quality Retrieval-Augmented Generation (RAG) application
that lets you upload PDF documents, build a personal searchable
knowledge base, and ask natural language questions. Every answer is
generated strictly from your documents and includes source citations
(filename and page number).

---

## Features

- Upload one or more PDF files
- Automatic text extraction (PyMuPDF)
- Smart text chunking (LangChain RecursiveCharacterTextSplitter)
- Embedding generation using OpenAI's `text-embedding-3-small`
- Persistent vector storage with ChromaDB
- Top-K semantic retrieval
- LLM-generated answers with mandatory source citations
- Clean, minimal Streamlit interface

---

## Folder Structure