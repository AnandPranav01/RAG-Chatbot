"""
src/generate.py

Handles the generation half of RAG: takes retrieved chunks + the user's
question, builds a strict "answer only from context" prompt, and calls a
local LLM (Flan-T5-base) to produce the final answer.

Hallucination control strategy:
1. Prompt explicitly instructs the model to answer ONLY from the given
   context and to say "I don't have enough information..." otherwise.
2. A distance threshold check: if the best-matching chunk is too dissimilar
   from the question (i.e., nothing relevant was actually found), we skip
   calling the LLM entirely and return the "not found" response directly.
   This is a second line of defense against hallucination, independent of
   whether the LLM follows instructions perfectly.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from src.retrieve import retrieve

_pipeline = None


DISTANCE_THRESHOLD = 0.9

NOT_FOUND_MESSAGE = (
    "I don't have enough information in the available documents to answer "
    "that question."
)


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        print(f"Loading local LLM: {config.LLM_MODEL_NAME} (first run will download it)")
        _pipeline = pipeline("text2text-generation", model=config.LLM_MODEL_NAME)
    return _pipeline


def _build_prompt(question, chunks):
    context = "\n\n".join(
        f"[Source: {c['filename']}, page {c['page']}]\n{c['text']}" for c in chunks
    )
    prompt = (
        "You are an enterprise knowledge assistant. Answer the question using "
        "ONLY the context provided below. If the context does not contain the "
        "answer, respond exactly with: "
        f"\"{NOT_FOUND_MESSAGE}\"\n"
        "Do not use any outside knowledge. Be concise.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return prompt


def answer_question(question, top_k=config.TOP_K):
    """
    Full RAG answer pipeline: retrieve -> (guard) -> generate.
    Returns a dict matching the API response schema:
      {answer, sources, confidence}
    """
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        return {"answer": NOT_FOUND_MESSAGE, "sources": [], "confidence": 0.0}

    best_distance = chunks[0]["distance"]

    
    if best_distance > DISTANCE_THRESHOLD:
        return {"answer": NOT_FOUND_MESSAGE, "sources": [], "confidence": 0.0}

    prompt = _build_prompt(question, chunks)
    llm = _get_pipeline()
    result = llm(prompt, max_length=200, do_sample=False)
    answer_text = result[0]["generated_text"].strip()

    
    confidence = max(0.0, min(1.0, 1 - best_distance))

    sources = []
    seen = set()
    for c in chunks:
        key = (c["filename"], c["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"document": c["filename"], "page": c["page"]})

    # If the model itself decided it doesn't know, drop the sources to avoid
    # implying we found something we didn't actually use.
    if NOT_FOUND_MESSAGE.lower() in answer_text.lower():
        return {"answer": NOT_FOUND_MESSAGE, "sources": [], "confidence": 0.0}

    return {"answer": answer_text, "sources": sources, "confidence": round(confidence, 2)}


if __name__ == "__main__":
    q = "What is the employee leave policy?"
    print(answer_question(q))
