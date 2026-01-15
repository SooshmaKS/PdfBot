import os
import json
import time
import textwrap
import streamlit as st
import boto3
import faiss
import numpy as np
from PyPDF2 import PdfReader
from typing import List

# =========================
# CONFIG
# =========================
FAISS_INDEX_DIR = "faiss_data.npz"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_AGENT_ITERS = 4
AWS_REGION = "us-east-1"

# =========================
# BEDROCK CLIENT
# =========================
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION
)

# =========================
# EMBEDDINGS (Titan)
# =========================
def get_embedding(text: str) -> np.ndarray:
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        body=json.dumps({"inputText": text}),
        accept="application/json",
        contentType="application/json"
    )
    result = json.loads(response["body"].read())
    return np.array(result["embedding"], dtype="float32")

# =========================
# LLM CALL (Llama 3)
# =========================
def call_model_chat(system: str, user: str, max_tokens=512, temperature=0.2) -> str:
    prompt = f"""<|begin_of_text|>
<|system|>
{system}
<|user|>
{user}
<|assistant|>
"""

    response = bedrock.invoke_model(
        modelId="meta.llama3-8b-instruct-v1",
        body=json.dumps({
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "top_p": 0.9
        }),
        accept="application/json",
        contentType="application/json"
    )

    result = json.loads(response["body"].read())
    return result["generation"].strip()

# =========================
# VECTOR STORE (FAISS)
# =========================
class SimpleVectorStore:
    def __init__(self):
        self.index = None
        self.id_to_text = []

    def build_index(self, texts: List[str]):
        if not texts:
            return
        embeddings = [get_embedding(t) for t in texts]
        dim = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))
        self.id_to_text = texts.copy()

    def save(self, path=FAISS_INDEX_DIR):
        if not self.id_to_text:
            return
        embeddings = [get_embedding(t) for t in self.id_to_text]
        np.savez(path, embs=np.array(embeddings), texts=np.array(self.id_to_text, dtype=object))

    def load(self, path=FAISS_INDEX_DIR):
        if not os.path.exists(path):
            return
        data = np.load(path, allow_pickle=True)
        embs = data["embs"]
        texts = data["texts"].tolist()
        dim = embs.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embs)
        self.id_to_text = texts

    def similarity_search(self, query: str, k=4):
        if not self.index:
            return []
        q_emb = np.array([get_embedding(query)])
        _, idxs = self.index.search(q_emb, k)
        return [self.id_to_text[i] for i in idxs[0] if i < len(self.id_to_text)]

# =========================
# PDF UTILITIES
# =========================
def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n\n"
    return text

def chunk_text(text: str):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# =========================
# TOOLS
# =========================
def tool_search_pdf(query: str):
    vs = st.session_state.vector_store
    hits = vs.similarity_search(query)
    if not hits:
        return "No relevant information found."
    return "\n".join([f"=== CHUNK {i+1} ===\n{h}" for i, h in enumerate(hits)])

# =========================
# AGENT PROMPTS
# =========================
PLANNER_SYSTEM = """You are the Planner agent.
Respond ONLY in JSON.
Actions:
{"action":"research","input":"query"}
{"action":"answer","output":"instruction"}"""

ANSWER_SYSTEM = """You are the Answer agent.
Write a concise final answer using research results.
Use citations like [chunk 1]."""

# =========================
# AGENTS
# =========================
def planner_agent_decide(question, history, research=""):
    user_prompt = f"History:\n{history}\n\nQuestion:\n{question}\n\nResearch:\n{research}"
    raw = call_model_chat(PLANNER_SYSTEM, user_prompt)
    try:
        return json.loads(raw)
    except:
        return {"action": "research", "input": question}

def answer_agent_compose(instr, research, history, question):
    user_prompt = f"""
Instruction: {instr}
Research:
{research}
Question:
{question}
"""
    return call_model_chat(ANSWER_SYSTEM, user_prompt, max_tokens=700)

def run_multi_agent(question):
    history = ""
    research = ""
    for _ in range(MAX_AGENT_ITERS):
        decision = planner_agent_decide(question, history, research)
        if decision["action"] == "research":
            research = tool_search_pdf(decision.get("input", question))
            history += f"Research:\n{research}\n"
        else:
            return answer_agent_compose(decision.get("output",""), research, history, question)
    return answer_agent_compose("", research, history, question)

# =========================
# STREAMLIT UI
# =========================
st.set_page_config("Agentic PDF Chat (AWS Bedrock)", layout="wide")
st.title("📄 Agentic PDF Chat — AWS Bedrock")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = SimpleVectorStore()
    st.session_state.vector_store.load()

with st.sidebar:
    uploaded = st.file_uploader("Upload PDFs", accept_multiple_files=True)
    if st.button("Process PDFs") and uploaded:
        raw = extract_text_from_pdfs(uploaded)
        chunks = chunk_text(raw)
        st.session_state.vector_store.build_index(chunks)
        st.session_state.vector_store.save()
        st.success(f"Indexed {len(chunks)} chunks")

question = st.chat_input("Ask a question")
if question:
    st.chat_message("user").write(question)
    with st.spinner("Agents thinking..."):
        answer = run_multi_agent(question)
    st.chat_message("assistant").write(answer)
