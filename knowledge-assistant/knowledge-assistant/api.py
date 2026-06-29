"""
api.py

Optional REST API for the Enterprise Knowledge Assistant.
Run with:  uvicorn api:app --reload

Exposes:
  POST /ask   {"question": "..."} -> {"answer", "sources", "confidence"}
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.generate import answer_question

app = FastAPI(title="Enterprise Knowledge Assistant API")


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    document: str
    page: int


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        result = answer_question(request.question)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    return result
