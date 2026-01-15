import os
import streamlit as st
import faiss
import numpy as np
from PyPDF2 import PdfReader
import google.generativeai as genai

# ---------------- CONFIG ---------------- #
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

EMBED_MODEL = "models/embedding-001"
CHAT_MODEL = "gemini-1.5-flash"

chat_model = genai.GenerativeModel(CHAT_MODEL)

# ---------------- PDF LOADING ---------------- #
def load_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# ---------------- CHUNKING ---------------- #
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

# ---------------- EMBEDDINGS ---------------- #
def embed_texts(texts):
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=text
        )
        embeddings.append(result["embedding"])
    return np.array(embeddings).astype("float32")

# ---------------- VECTOR STORE ---------------- #
def build_vector_store(chunks):
    embeddings = embed_texts(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    return index, chunks

def search_vector_store(query, index, chunks, top_k=3):
    query_embedding = embed_texts([query])
    _, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# ======================================================
# ===================== AGENTS =========================
# ======================================================

# -------- PLANNER AGENT -------- #
def planner_agent(question: str) -> str:
    prompt = f"""
You are a planning agent.

Decide the next action.

If the question requires document context, reply ONLY with:
SEARCH

If it can be answered directly, reply ONLY with:
ANSWER

Question:
{question}
"""
    response = chat_model.generate_content(prompt)
    return response.text.strip()

# -------- TOOL AGENT -------- #
def tool_agent_search(question, index, chunks):
    return search_vector_store(question, index, chunks)

# -------- ANSWER AGENT -------- #
def answer_agent(question, context):
    prompt = f"""
You are an answering agent.

Use the context below to answer clearly and concisely.

Context:
{context}

Question:
{question}
"""
    response = chat_model.generate_content(prompt)
    return response.text

# -------- AGENT ORCHESTRATOR -------- #
def run_agentic_ai(question, index, chunks):
    decision = planner_agent(question)

    if decision == "SEARCH":
        retrieved_chunks = tool_agent_search(question, index, chunks)
        context = "\n\n".join(retrieved_chunks)
        return answer_agent(question, context)

    else:
        return answer_agent(question, "")

# ======================================================
# ===================== STREAMLIT ======================
# ======================================================

st.title("📄 Agentic AI PDF Q&A (Google Cloud)")

if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
    st.session_state.chunks = None

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files and st.session_state.vector_index is None:
    with st.spinner("Processing PDFs..."):
        text = load_pdf_text(uploaded_files)
        chunks = chunk_text(text)
        index, chunks = build_vector_store(chunks)

        st.session_state.vector_index = index
        st.session_state.chunks = chunks

        st.success("PDFs processed successfully!")

question = st.chat_input("Ask a question about the PDFs")

if question and st.session_state.vector_index:
    with st.spinner("Agents are thinking..."):
        answer = run_agentic_ai(
            question,
            st.session_state.vector_index,
            st.session_state.chunks
        )
    st.write(answer)
