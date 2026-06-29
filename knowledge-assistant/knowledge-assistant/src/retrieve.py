

import os
import sys

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_embedder = None
_collection = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        try:
            _collection = client.get_collection(config.COLLECTION_NAME)
        except Exception as e:
            raise RuntimeError(
                "Vector index not found. Did you run 'python src/ingest.py' first?"
            ) from e
    return _collection


def retrieve(question, top_k=config.TOP_K):
    """
    Embeds the question and retrieves the top_k most relevant chunks.
    Returns a list of dicts: {text, filename, page, distance}
    """
    embedder = _get_embedder()
    collection = _get_collection()

    query_embedding = embedder.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    retrieved = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, meta, distance in zip(docs, metas, distances):
        retrieved.append({
            "text": text,
            "filename": meta.get("filename"),
            "page": meta.get("page"),
            "distance": distance,
        })

    return retrieved


if __name__ == "__main__":
    # Quick manual test
    q = "What is the employee leave policy?"
    results = retrieve(q)
    for r in results:
        print(f"[{r['filename']} p.{r['page']}] (dist={r['distance']:.3f}) {r['text'][:120]}...")
