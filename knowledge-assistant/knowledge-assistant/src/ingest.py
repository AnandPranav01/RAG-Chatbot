

import os
import sys
import re

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_txt_pages(filepath):
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

 
    parts = re.split(r"\[PAGE (\d+)\]", content)
    pages = []
    # parts looks like: ['', '1', ' text...', '2', ' text...', ...]
    for i in range(1, len(parts), 2):
        page_num = int(parts[i])
        page_text = parts[i + 1].strip()
        if page_text:
            pages.append((page_num, page_text))

  
    if not pages:
        pages = [(1, content.strip())]

    return pages


def load_pdf_pages(filepath):
    """
    Loads a real PDF file and returns a list of (page_number, page_text) tuples.
    Uses pdfplumber for text extraction.
    """
    import pdfplumber

    pages = []
    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append((i, text))
    return pages


def chunk_text(text, chunk_size=config.CHUNK_SIZE_WORDS, overlap=config.CHUNK_OVERLAP_WORDS):
    
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap  # step forward, keeping some overlap
    return chunks


def load_all_documents(data_dir=config.DATA_DIR):
    
    records = []

    for filename in sorted(os.listdir(data_dir)):
        filepath = os.path.join(data_dir, filename)
        if not os.path.isfile(filepath):
            continue

        ext = filename.lower().split(".")[-1]
        if ext == "txt":
            pages = load_txt_pages(filepath)
        elif ext == "pdf":
            pages = load_pdf_pages(filepath)
        else:
            print(f"Skipping unsupported file type: {filename}")
            continue

        for page_num, page_text in pages:
            chunks = chunk_text(page_text)
            for idx, chunk in enumerate(chunks):
                records.append({
                    "filename": filename,
                    "page": page_num,
                    "chunk_index": idx,
                    "text": chunk,
                })

    return records


def build_index():
    
    print("Loading documents from:", config.DATA_DIR)
    records = load_all_documents()
    print(f"Loaded {len(records)} chunks from documents.")

    if not records:
        print("No documents found. Add .txt or .pdf files to the data/ folder.")
        return

    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME} (first run will download it)")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    texts = [r["text"] for r in records]
    print("Generating embeddings...")
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    print("Connecting to ChromaDB at:", config.CHROMA_DIR)
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)

   
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(config.COLLECTION_NAME)

    ids = [f"{r['filename']}_p{r['page']}_c{r['chunk_index']}" for r in records]
    metadatas = [{"filename": r["filename"], "page": r["page"]} for r in records]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Indexed {len(records)} chunks into collection '{config.COLLECTION_NAME}'.")
    print("Ingestion complete. You can now run the app.")


if __name__ == "__main__":
    build_index()
