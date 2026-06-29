# Enterprise Knowledge Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers employee
questions using only the content of a company's internal documents, and
cites the exact source document and page for every answer.

This implementation uses **fully local, free, open-source models** (no API
key required) so it can be set up and run entirely offline.

---

## 1. Architecture Overview

```
                 ┌─────────────────────┐
                 │   data/ (.txt/.pdf)  │
                 └──────────┬───────────┘
                            │ (1) Load + split into pages
                            ▼
                 ┌─────────────────────┐
                 │   Chunking           │  src/ingest.py
                 │ (per-page, overlap)  │
                 └──────────┬───────────┘
                            │ (2) Embed each chunk
                            ▼
                 ┌─────────────────────┐
                 │ Sentence-Transformers│
                 │  (all-MiniLM-L6-v2)  │
                 └──────────┬───────────┘
                            │ (3) Store vectors + metadata
                            ▼
                 ┌─────────────────────┐
                 │     ChromaDB          │  (persisted to disk)
                 └──────────┬───────────┘
                            │
        User question ──────┤ (4) Embed question, similarity search
                            ▼
                 ┌─────────────────────┐
                 │  Top-k relevant      │  src/retrieve.py
                 │  chunks + metadata   │
                 └──────────┬───────────┘
                            │ (5) Build grounded prompt
                            ▼
                 ┌─────────────────────┐
                 │  Local LLM            │  src/generate.py
                 │  (Flan-T5-base)       │
                 └──────────┬───────────┘
                            │ (6) Answer + sources + confidence
                            ▼
                 ┌─────────────────────┐
                 │ Streamlit UI / API    │  src/app.py , api.py
                 └─────────────────────┘
```

**Two-stage pipeline:**
- **Offline (ingestion):** documents → chunks → embeddings → vector store
  (run once, or whenever documents change)
- **Online (query time):** question → embedding → similarity search →
  grounded prompt → LLM answer → response with citations

---

## 2. Technology Choices & Why

| Component        | Choice                     | Why |
|-------------------|-----------------------------|-----|
| Embeddings        | Sentence-Transformers (`all-MiniLM-L6-v2`) | Free, runs on CPU, no API key, fast, good general-purpose semantic quality for short policy text. |
| Vector DB         | ChromaDB                   | No external server needed, simple Python API, persists to disk — ideal for a self-contained assignment project. |
| LLM (generation)  | `google/flan-t5-base` (local, via Hugging Face Transformers) | Free, instruction-tuned, runs on CPU without a GPU or API key, sufficient to demonstrate grounded generation end-to-end. |
| Document parsing  | `pdfplumber` for PDFs, plain text for `.txt` | Reliable text + page-level extraction. |
| UI                | Streamlit                  | Fastest way to build a usable chat interface in pure Python. |
| API               | FastAPI                    | Matches the `/ask` schema requested in the assignment; async-ready and easy to extend. |

**Note on the LLM choice:** Flan-T5-base is intentionally small and free so
the whole project runs without any paid API or internet dependency after
setup. The architecture is provider-agnostic — swapping in Claude or GPT
only requires changing `src/generate.py`'s `_get_pipeline`/`answer_question`
to call an API instead of a local pipeline. This tradeoff (cost/offline-use
vs. answer fluency) is explained further in Limitations below.

---

## 3. Chunking Strategy

- Documents are split **by page first**, then each page's text is chunked
  into ~180-word windows with a 40-word overlap.
- **Why per-page first:** guarantees every chunk can be cited with an
  accurate page number — chunks never span two different pages.
- **Why overlap:** prevents an answer that sits across a chunk boundary
  from becoming unretrievable or fragmented.
- Chunk size (180 words) was chosen because the sample documents are
  short policy sections; this keeps each chunk focused on one topic
  (e.g., one leave type) rather than blending multiple topics together.

---

## 4. Retrieval Strategy

- Pure semantic (vector) similarity search using cosine distance, top-k = 4.
- A **distance threshold** (0.9) acts as a relevance gate: if even the
  best-matching chunk is too dissimilar from the question, the system
  skips the LLM call entirely and returns "I don't have enough
  information..." — this is a deliberate second layer of hallucination
  prevention, independent of whether the LLM follows its instructions.

---

## 5. Prompt Design (Hallucination Prevention)

The prompt sent to the LLM explicitly:
1. Provides only the retrieved chunks as context (with source labels)
2. Instructs the model to answer **only** from that context
3. Gives an exact fallback phrase to use when the answer isn't present
4. Asks for concise answers (reduces drift/rambling)

This is combined with the retrieval-distance guard above, so hallucination
is prevented at two independent points in the pipeline.

---

## 6. Setup Instructions

### Prerequisites
- Python 3.10+
- ~2 GB free disk space (for model downloads on first run)

### Steps

```bash
# 1. Clone/open the project folder in VS Code, then open a terminal in it

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run ingestion (builds the vector index from documents in data/)
python src/ingest.py

# 5a. Launch the chat UI
streamlit run src/app.py

# 5b. OR launch the API instead/also
uvicorn api:app --reload
```

The first run of ingestion/UI will download the embedding model (~80MB)
and the LLM (~250MB) automatically — this requires internet access once,
but no API key and no ongoing cost.

### Adding your own documents
Drop `.txt` files (using `[PAGE n]` markers, see files in `data/` for
the format) or real `.pdf` files into the `data/` folder, then re-run
`python src/ingest.py` to rebuild the index.

### Testing
```bash
python tests/test_questions.py
```

### Example API call
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the employee leave policy?"}'
```

---

## 7. Evaluation Approach

- `tests/test_questions.py` runs 11 manually written test questions covering:
  - Directly answerable questions (8) — checked answer correctness and
    that the cited document/page actually matches the source policy.
  - One ambiguous question ("What is the leave policy?") — checked that
    the system still returns a relevant, non-hallucinated answer (likely
    pulling the annual leave section) rather than failing.
  - Two out-of-scope questions (capital of France, company revenue) —
    checked that the system correctly returns the "I don't have enough
    information" fallback instead of guessing.
- Confidence score is derived from retrieval similarity (`1 - distance`)
  as a simple, explainable proxy for how relevant the retrieved evidence
  was — it is not a calibrated probability of correctness.
- **Improvements attempted during development:**
  - Switched from whole-document chunking to per-page chunking, after
    noticing whole-document chunking caused inaccurate page citations.
  - Added the distance-threshold guard after noticing the small local LLM
    would occasionally try to answer using weakly relevant chunks instead
    of admitting it didn't know.

---

## 8. Limitations

- **LLM fluency:** Flan-T5-base is much weaker than GPT-4/Claude; answers
  are correct but terser and less conversational. Swapping in a hosted
  LLM API would noticeably improve answer quality.
- **No conversation memory:** each question is answered independently;
  follow-up questions referring to "it" or "that" won't have context.
- **No re-ranking or hybrid (keyword) search:** retrieval is purely
  embedding-based, so very rare exact terms/codes might be missed.
- **No authentication or multi-user support:** intended as a
  single-user demo/prototype, not production-hardened.
- **Confidence score is a heuristic**, not a calibrated probability.

---

## 9. Future Improvements

- Plug in a hosted LLM (Claude/GPT) behind a config flag for higher-quality
  answers while keeping the local model as a free fallback.
- Add hybrid search (BM25 keyword + semantic) and a re-ranking step.
- Add conversation memory for multi-turn follow-up questions.
- Add query rewriting for vague/ambiguous questions.
- Add user feedback (thumbs up/down) to collect data for future tuning.
- Containerize with Docker and add basic auth for shared deployment.

---

## 10. Project Structure

```
knowledge-assistant/
├── data/                       # sample source documents
├── src/
│   ├── ingest.py                # document loading, chunking, embedding, indexing
│   ├── retrieve.py               # semantic search over the vector store
│   ├── generate.py               # grounded prompting + local LLM answer generation
│   └── app.py                    # Streamlit chat UI
├── api.py                       # FastAPI POST /ask endpoint
├── tests/
│   └── test_questions.py         # evaluation test questions
├── config.py                    # central configuration
├── requirements.txt
└── README.md
```
