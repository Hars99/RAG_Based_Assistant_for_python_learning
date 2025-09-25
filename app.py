# app.py
import os
import streamlit as st
import chromadb
from prompts import BASE_PROMPT, COLLECTION_NAME
from utils import get_embedder
import numpy as np


import google.generativeai as genai

# ---------- Config ----------
PERSIST_DIR = "./chroma_db"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # Set this in your environment
MODEL_NAME = "gemini-1.5-flash"  # Gemini model
K = 3  # top-k retrieval

# ---------- Setup ----------
st.set_page_config(page_title="RAG Chatbot — Python Tutor", page_icon="🐍", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#0f172a,#001219); color: #e2e8f0; }
    .chat-box { background: rgba(255,255,255,0.03); padding: 12px; border-radius: 12px; margin-bottom:8px; }
    .user { color: #7dd3fc; }
    .bot { color: #a78bfa; }
    </style>
    """, unsafe_allow_html=True
)

st.title("🐍 Python Tutor — RAG Chatbot")
st.write("Ask questions about your Python tutorial PDF. Uses ChromaDB + all-MiniLM-L6-v2 + Gemini LLM.")

# Initialize Chroma persistent client
client = chromadb.PersistentClient(path=PERSIST_DIR)
try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception as e:
    st.error("Chroma collection not found. Run `ingest.py` first to ingest your PDF.")
    st.stop()

# Initialize embedder
embedder = get_embedder()

# Initialize Gemini client
genai.configure(api_key=GEMINI_API_KEY)

def call_gemini(prompt, max_tokens=100):
    """Call Gemini 1.5 Flash API and return generated text."""
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt, generation_config={"max_output_tokens": max_tokens})
    return response.text

# UI layout
col1, col2 = st.columns([3, 1])

with col1:
    if "history" not in st.session_state:
        st.session_state.history = []  # list of (user, bot)
    with st.form("query_form", clear_on_submit=True):
        q = st.text_input("Ask a question about the document", placeholder="What is a decorator in Python?")
        submitted = st.form_submit_button("Send")
    if submitted and q:
        # 1) embed query
        q_emb = embedder.encode(q)
        q_emb = q_emb.tolist() if hasattr(q_emb, "tolist") else list(q_emb)

        # 2) query Chroma
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=K,
            include=['documents', 'metadatas', 'distances']
        )

        retrieved_docs = results['documents'][0]  # list of strings
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        # build context
        context = "\n\n---\n\n".join(retrieved_docs)

        # 3) construct prompt
        prompt = BASE_PROMPT.format(context=context, question=q)

        # save retrieved for sidebar transparency
        st.session_state.retrieved = list(zip(retrieved_docs, metadatas, distances))

        # 4) call Gemini LLM
        try:
            generated = call_gemini(prompt, max_tokens=100)  # keep small for 500-token daily quota
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            generated = "Sorry, LLM generation failed."

        # Save to history
        st.session_state.history.append((q, generated))

    # display chat
    for i, (user_q, bot_a) in enumerate(reversed(st.session_state.history)):
        st.markdown(f"<div class='chat-box'><b class='user'>You:</b> {user_q}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-box'><b class='bot'>Tutor:</b> {bot_a}</div>", unsafe_allow_html=True)

with col2:
    st.subheader("Retrieved context")
    if "retrieved" in st.session_state:
        for idx, (doc, meta, dist) in enumerate(st.session_state.retrieved):
            st.markdown(f"**Result {idx+1}** — page {meta.get('page', '?')}, chunk {meta.get('chunk', '?')} — distance {dist:.4f}")
            snippet = doc[:400].replace("\n", " ")
            st.text(snippet + ("..." if len(doc) > 400 else ""))
            st.markdown("---")
    else:
        st.write("No queries yet. Retrieved docs will appear here.")

st.markdown("### Tips\n- Rephrase if answer is missing.\n- Try asking code-specific questions like: 'What does this function do?' or 'Explain variables in Python.'")
