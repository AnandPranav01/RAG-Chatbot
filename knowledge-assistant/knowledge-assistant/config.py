"""
config.py
Central configuration for the Enterprise Knowledge Assistant.
Keeping all tunable settings in one place makes the system easier to
maintain and experiment with (e.g., changing chunk size or model names).
"""

import os

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_store")  # persistent vector DB folder

# ---- Embedding model ----
# Small, fast, runs fully on CPU, no API key required.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ---- Local LLM for answer generation ----
# Flan-T5-base is a small instruction-tuned model that runs on CPU without
# needing a GPU or API key. It's not as fluent as GPT-4/Claude, but it is
# good enough to demonstrate the RAG pipeline end-to-end for free.
LLM_MODEL_NAME = "google/flan-t5-base"

# ---- Chunking ----
CHUNK_SIZE_WORDS = 180        # approx words per chunk
CHUNK_OVERLAP_WORDS = 40      # overlap between consecutive chunks

# ---- Retrieval ----
TOP_K = 4                     # number of chunks retrieved per question

# ---- Collection name in ChromaDB ----
COLLECTION_NAME = "enterprise_knowledge_base"
