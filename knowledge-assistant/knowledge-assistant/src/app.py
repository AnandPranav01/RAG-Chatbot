import os
import sys

import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.generate import answer_question

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="🤖")
st.title("Enterprise Knowledge Assistant")
st.caption("Ask questions about your internal documents and get grounded answers with citations.")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Ask a question", placeholder="Example: What is the employee leave policy?")

if st.button("Ask") or question:
    if question.strip():
        with st.spinner("Searching the knowledge base..."):
            result = answer_question(question)

        st.session_state.history.append((question, result))

        st.subheader("Answer")
        st.write(result["answer"])

        if result.get("sources"):
            st.subheader("Sources")
            for source in result["sources"]:
                st.write(f"- {source['document']}, page {source['page']}")

        st.caption(f"Confidence: {result.get('confidence', 0.0)}")

if st.session_state.history:
    st.subheader("Recent questions")
    for idx, (q, result) in enumerate(reversed(st.session_state.history), start=1):
        with st.expander(f"{idx}. {q}"):
            st.write(result["answer"])
            if result.get("sources"):
                st.write("Sources:")
                for source in result["sources"]:
                    st.write(f"- {source['document']}, page {source['page']}")