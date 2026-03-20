# Learning RAG Pipeline

A project for learning and implementing Retrieval-Augmented Generation (RAG) pipelines using LangChain.

## Overview

This project demonstrates how to build a RAG pipeline that ingests documents, processes them, and uses them for retrieval-based question answering.

## Features

- Document ingestion from multiple sources (text files, PDFs)
- Document processing and chunking
- LangChain integration for RAG workflows

## Tech Stack

- **Python** 3.12+
- **LangChain** - LLM framework
- **LangChain Community** - Additional loaders and tools
- **PyPDF** - PDF processing
- **PyMuPDF** - Alternative PDF processing

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   uv sync
   ```

## Project Structure

```
learning-rag-pipeline/
├── data/              # Data files and documents
├── notebook/          # Jupyter notebooks for experimentation
├── main.py            # Main entry point
├── pyproject.toml     # Project configuration
├── requirements.txt   # Dependencies
└── README.md          # This file
```

## Getting Started

1. Activate the virtual environment (if using venv)
2. Run notebooks in `notebook/` directory to explore RAG concepts
3. Check `main.py` for the main application logic

## Development

- Python version: 3.12 (see `.python-version`)
- Dependencies are locked in `uv.lock` for reproducibility
