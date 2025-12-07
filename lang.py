import os
import json
import textwrap
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from typing import List
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Set HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN) in your environment or .env")
HF_LLM_MODEL = "deepseek-ai/DeepSeek-R1"
FAISS_INDEX_DIR = "faiss_data.npz"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_AGENT_ITERS = 4
os.environ["AUTOGEN_USE_DOCKER"] = "0"

class SimpleVectorStore:
    def __init__(self, embed_model_name=EMBED_MODEL):
        self.embedder = SentenceTransformer(embed_model_name)
        self.index = None
        self.id_to_text = []

    def build_index(self, texts: List[str]):
        if not texts:
            self.index = None
            self.id_to_text = []
            return
        embs = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        d = embs.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embs)
        self.id_to_text = texts.copy()

    def save(self, path=FAISS_INDEX_DIR):
        if self.index is None:
            np.savez(path, embs=np.array([]), texts=np.array([], dtype=object))
            return
        embs = self.embedder.encode(self.id_to_text, convert_to_numpy=True, show_progress_bar=False)
        np.savez(path, embs=embs, texts=np.array(self.id_to_text, dtype=object))

    def load(self, path=FAISS_INDEX_DIR):
        if not os.path.exists(path):
            self.index = None
            self.id_to_text = []
            return
        data = np.load(path, allow_pickle=True)
        embs = data.get("embs", None)
        texts = data.get("texts", None)
        if embs is None or texts is None or embs.size == 0:
            self.index = None
            self.id_to_text = []
            return
        texts = texts.tolist()
        d = embs.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embs)
        self.id_to_text = texts

    def similarity_search(self, query: str, k=4):
        if self.index is None or not self.id_to_text:
            return []
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        distances, idxs = self.index.search(q_emb, k)
        idxs = idxs[0]
        results = []
        for i in idxs:
            if 0 <= i < len(self.id_to_text):
                results.append(self.id_to_text[i])
        return results

if "vector_store" not in st.session_state:
    st.session_state.vector_store = SimpleVectorStore()
    try:
        st.session_state.vector_store.load(FAISS_INDEX_DIR)
    except:
        pass

def extract_text_from_pdfs(pdf_files) -> str:
    txt = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            p = page.extract_text()
            if p:
                txt += p + "\n\n"
    return txt

def chunk_text(text: str, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end].strip())
        start += chunk_size - chunk_overlap
    return chunks

def tool_search_pdf(query: str, k=4) -> str:
    vs: SimpleVectorStore = st.session_state.vector_store
    hits = vs.similarity_search(query, k=k)
    if not hits:
        return "No relevant information found in the uploaded PDFs."
    out = []
    for i, h in enumerate(hits, start=1):
        out.append(f"=== CHUNK {i} ===\n{h}\n")
    return "\n".join(out)

from autogen import AssistantAgent, UserProxyAgent

# Fixed llm_config - updated to new HuggingFace router endpoint
llm_config = {
    "model": HF_LLM_MODEL,
    "api_key": HF_TOKEN,
    "base_url": "https://router.huggingface.co/v1"
}

PLANNER_SYSTEM = textwrap.dedent("""
You are the Planner agent. The user asks a question about uploaded PDFs.
Return a single JSON object on one line:
{"action":"research","input":"..."} or {"action":"answer","output":"..."}
Only output JSON.
""").strip()

ANSWER_SYSTEM = textwrap.dedent("""
You are the Answer agent. You receive the planner decision and research chunks.
Write a concise final answer. Use inline citations like [chunk 1].
Return only the final answer.
""").strip()

planner_agent = AssistantAgent(
    name="planner",
    system_message=PLANNER_SYSTEM,
    llm_config=llm_config,
    human_input_mode="NEVER"
)

answer_agent = AssistantAgent(
    name="answer",
    system_message=ANSWER_SYSTEM,
    llm_config=llm_config,
    human_input_mode="NEVER"
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    system_message="You act as the user. Provide the user's question as-is.",
    human_input_mode="NEVER"
)

def planner_decide(question: str, convo_history: str = "", research_results: str = "") -> dict:
    prompt = f"Conversation history:\n{convo_history}\n\nUser question:\n{question}\n\n"
    if research_results:
        prompt += f"Previous research results:\n{research_results}\n\n"
    resp = planner_agent.generate_reply(messages=[{"role": "user", "content": prompt}])
    raw = resp if isinstance(resp, str) else str(resp)
    s = raw.strip()
    try:
        first = s.splitlines()[0]
        return json.loads(first)
    except:
        try:
            start = s.find("{")
            end = s.rfind("}") + 1
            return json.loads(s[start:end])
        except:
            return {"action": "research", "input": question}

def answer_compose(planner_instruction: str, research_output: str, convo_history: str, question: str) -> str:
    prompt = (
        f"Planner instruction:\n{planner_instruction}\n\n"
        f"Research output:\n{research_output}\n\n"
        f"Conversation history:\n{convo_history}\n\n"
        f"User question:\n{question}\n\n"
        "Write the final answer.\n"
    )
    resp = answer_agent.generate_reply(messages=[{"role": "user", "content": prompt}])
    return resp.strip() if isinstance(resp, str) else str(resp).strip()

def run_autogen_multi_agent(user_question: str, max_iters: int = MAX_AGENT_ITERS) -> str:
    if "convo_memory" not in st.session_state:
        st.session_state.convo_memory = []
    history = ""
    for msg in st.session_state.convo_memory:
        history += f"{msg['role'].capitalize()}: {msg['content']}\n"
    research = ""
    last_p = None
    for _ in range(max_iters):
        out = planner_decide(user_question, history, research)
        last_p = out
        if out.get("action") == "research":
            q = out.get("input", "").strip() or user_question
            research = tool_search_pdf(q, k=4)
            history += f"Research: {research}\n"
            continue
        if out.get("action") == "answer":
            ins = out.get("output", "")
            ans = answer_compose(ins, research, history, user_question)
            st.session_state.convo_memory.append({"role": "user", "content": user_question})
            st.session_state.convo_memory.append({"role": "assistant", "content": ans})
            return ans
        ins = out.get("output", "") or "Please answer."
        ans = answer_compose(ins, research, history, user_question)
        st.session_state.convo_memory.append({"role": "user", "content": user_question})
        st.session_state.convo_memory.append({"role": "assistant", "content": ans})
        return ans
    ins = last_p.get("output", "") if isinstance(last_p, dict) else "Please answer."
    ans = answer_compose(ins, research, history, user_question)
    st.session_state.convo_memory.append({"role": "user", "content": user_question})
    st.session_state.convo_memory.append({"role": "assistant", "content": ans})
    return ans

st.set_page_config(page_title="Agentic PDF Chat — AutoGen", layout="wide")
st.title("Agentic PDF Chat — AutoGen (Planner/Research/Answer)")

with st.sidebar:
    st.header("Upload & Process PDFs")
    uploaded = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
    if st.button("Process PDFs"):
        if not uploaded:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Extracting text and building vector index..."):
                raw = extract_text_from_pdfs(uploaded)
                if not raw.strip():
                    st.error("No extractable text.")
                else:
                    chunks = chunk_text(raw)
                    st.session_state.vector_store.build_index(chunks)
                    st.session_state.vector_store.save(FAISS_INDEX_DIR)
                    st.success(f"Processed {len(chunks)} chunks.")

st.subheader("Conversation")
if "convo_memory" in st.session_state:
    for msg in st.session_state.convo_memory:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

question = st.chat_input("Ask a question about the PDFs")
if question:
    st.chat_message("user").write(question)
    with st.spinner("Agents working..."):
        ans = run_autogen_multi_agent(question)
    st.chat_message("assistant").write(ans)

st.sidebar.markdown("---")
if st.sidebar.button("Show index size"):
    vs = st.session_state.vector_store
    st.sidebar.text(f"Chunks in index: {len(vs.id_to_text)}")

if st.sidebar.button("Show raw index (first 10 chunks)"):
    vs = st.session_state.vector_store
    if not vs.id_to_text:
        st.sidebar.text("No index built.")
    else:
        txt = "\n\n---\n\n".join(vs.id_to_text[:10])
        st.sidebar.text_area("Raw chunks", txt, height=300)