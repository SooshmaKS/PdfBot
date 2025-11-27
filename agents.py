import os
import json
import time
import textwrap
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from PyPDF2 import PdfReader
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from huggingface_hub import InferenceClient

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
FAISS_INDEX_DIR = "faiss_data.npz"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_AGENT_ITERS = 4
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    raise ValueError("Set HUGGINGFACEHUB_API_TOKEN in your environment or .env")

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
            if i < 0 or i >= len(self.id_to_text):
                continue
            results.append(self.id_to_text[i])
        return results

if "vector_store" not in st.session_state:
    st.session_state.vector_store = SimpleVectorStore()
    try:
        st.session_state.vector_store.load(FAISS_INDEX_DIR)
    except Exception:
        pass

hf_client = InferenceClient(token=HF_TOKEN)

def call_model_chat(system: str, user: str, max_tokens=512, temperature=0.2) -> str:
    """
    Unified chat call wrapper. Tries chat_completion first, falls back to text_generation.
    Returns a string (model output).
    """
    try:
        resp = hf_client.chat_completion(
            model=HF_LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if hasattr(resp, "choices") and len(resp.choices) > 0:
            choice = resp.choices[0]
            if hasattr(choice, "message"):
                return choice.message.get("content", "") or str(choice)
            if isinstance(choice, dict) and "message" in choice:
                return choice["message"].get("content", "") or str(choice)
        if isinstance(resp, dict):
            return resp.get("generated_text") or json.dumps(resp)
        return str(resp)
    except Exception as e:
        try:
            prompt = f"{system}\n\nUser: {user}\nAssistant:"
            resp2 = hf_client.text_generation(
                model=HF_LLM_MODEL,
                inputs=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
            if isinstance(resp2, list) and len(resp2) > 0:
                return resp2[0].get("generated_text", "") or str(resp2[0])
            if isinstance(resp2, dict):
                return resp2.get("generated_text", "") or str(resp2)
            return str(resp2)
        except Exception as e2:
            return f"[model error: {e2}]"

def extract_text_from_pdfs(pdf_files) -> str:
    txt = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                txt += page_text + "\n\n"
    return txt

def chunk_text(text: str, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append(chunk.strip())
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

def tool_read_all() -> str:
    vs: SimpleVectorStore = st.session_state.vector_store
    if not vs.id_to_text:
        return "No PDF text available."
    return "\n\n".join(vs.id_to_text)

PLANNER_SYSTEM = textwrap.dedent("""
You are the Planner agent. The user asks a question and you may ask the Research agent to fetch PDF chunks, or proceed to request the Answer agent to write the final answer.

Rules:
- ALWAYS respond **only** with a single JSON object on one line.
- Valid actions:
  - {"action":"research", "input":"<short query string to send to Research>"}
  - {"action":"answer", "output":"<brief instruction or partial answer for Answer agent>"}
- If you need PDF content to answer, return action "research". If you already can form the answer without more research, return action "answer".
- Keep research queries concise (one sentence). If research results arrive, you may produce an "answer" action that references them.
""").strip()

ANSWER_SYSTEM = textwrap.dedent("""
You are the Answer agent. You will take the Planner's instruction and the research results and produce a concise, helpful, human-readable final answer to the user.
- The input will contain:
  - Planner instruction or preliminary answer
  - Research output (relevant chunks) labeled clearly
- Produce the final answer only (plain text). Use short inline citations like [chunk 1] when quoting or closely paraphrasing research output.
- Be concise and user-friendly.
""").strip()

RESEARCH_SYSTEM = textwrap.dedent("""
You are the Research agent. Your job is simple: given a short query, run the PDF tools and return the most relevant chunks. Return plain text that contains the labeled chunks.
If no relevant content exists, return a short message 'No relevant information found'.
""").strip()

def planner_agent_decide(question: str, convo_history: str, research_results: str = "") -> dict:
    """
    Returns a dict parsed from JSON produced by the Planner agent.
    """
    user_prompt = f"Conversation history:\n{convo_history}\n\nUser question:\n{question}\n\n"
    if research_results:
        user_prompt += f"Research results (if any):\n{research_results}\n\n"
    raw = call_model_chat(PLANNER_SYSTEM, user_prompt, max_tokens=512, temperature=0.2)
    raw_stripped = raw.strip()
    first_line = raw_stripped.splitlines()[0] if raw_stripped else ""
    try:
        parsed = json.loads(first_line)
        if isinstance(parsed, dict) and "action" in parsed:
            return parsed
    except Exception:
        try:
            start = raw_stripped.find("{")
            end = raw_stripped.rfind("}") + 1
            if start != -1 and end != -1:
                parsed = json.loads(raw_stripped[start:end])
                if isinstance(parsed, dict) and "action" in parsed:
                    return parsed
        except Exception:
            pass
    return {"action": "research", "input": question}

def research_agent_run(query: str) -> str:
    """
    Calls the internal PDF tools and returns formatted research text.
    """
    return tool_search_pdf(query, k=4)

def answer_agent_compose(planner_instruction: str, research_output: str, convo_history: str, question: str) -> str:
    """
    Calls the Answer agent (LLM) to produce final answer.
    """
    user_prompt = (
        f"Planner instruction / preliminary answer:\n{planner_instruction}\n\n"
        f"Research output:\n{research_output}\n\n"
        f"Conversation history:\n{convo_history}\n\n"
        f"User question:\n{question}\n\n"
        "Write a concise, user-facing answer. Use citations like [chunk 1] when referencing research chunks.\n"
    )
    raw = call_model_chat(ANSWER_SYSTEM, user_prompt, max_tokens=700, temperature=0.2)
    return raw.strip()

def run_multi_agent(user_question: str, max_iters=MAX_AGENT_ITERS) -> str:
    """
    Orchestrates Planner -> Research -> Planner -> Answer.
    """
    if "convo_memory" not in st.session_state:
        st.session_state.convo_memory = []
    history_text = ""
    for msg in st.session_state.convo_memory:
        history_text += f"{msg['role'].capitalize()}: {msg['content']}\n"

    research_results = ""
    planner_last = None

    for i in range(max_iters):
        planner_out = planner_agent_decide(user_question, history_text, research_results)
        planner_last = planner_out

        action = planner_out.get("action")
        if action == "research":
            query = planner_out.get("input", "").strip() or user_question
            research_results = research_agent_run(query)
            history_text += f"Research: {research_results}\n"
            continue
        elif action == "answer":
            planner_instruction = planner_out.get("output", "")
            final_answer = answer_agent_compose(planner_instruction, research_results, history_text, user_question)
            st.session_state.convo_memory.append({"role": "user", "content": user_question})
            st.session_state.convo_memory.append({"role": "assistant", "content": final_answer})
            return final_answer
        else:
            fallback_instruction = planner_out.get("output", "") or "Please answer using available research results."
            final_answer = answer_agent_compose(fallback_instruction, research_results, history_text, user_question)
            st.session_state.convo_memory.append({"role": "user", "content": user_question})
            st.session_state.convo_memory.append({"role": "assistant", "content": final_answer})
            return final_answer

    fallback_instruction = planner_last.get("output", "") if isinstance(planner_last, dict) else "Please answer using available research results."
    final_answer = answer_agent_compose(fallback_instruction, research_results, history_text, user_question)
    st.session_state.convo_memory.append({"role": "user", "content": user_question})
    st.session_state.convo_memory.append({"role": "assistant", "content": final_answer})
    return final_answer

st.set_page_config(page_title="Agentic PDF Chat — Multi-Agent (Planner, Research, Answer)", layout="wide")
st.title("Agentic PDF Chat — Multi-Agent")

with st.sidebar:
    st.header("Upload & Process PDFs")
    uploaded = st.file_uploader("Upload PDF files (multiple)", accept_multiple_files=True, type=["pdf"])
    if st.button("Process PDFs"):
        if not uploaded:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Extracting text and building vector index..."):
                raw_text = extract_text_from_pdfs(uploaded)
                if not raw_text.strip():
                    st.error("No extractable text. Are these scanned images?")
                else:
                    chunks = chunk_text(raw_text)
                    st.session_state.vector_store.build_index(chunks)
                    st.session_state.vector_store.save(FAISS_INDEX_DIR)
                    st.success(f"Processed {len(chunks)} chunks and built index.")

st.subheader("Conversation")
if "convo_memory" in st.session_state and st.session_state.convo_memory:
    for msg in st.session_state.convo_memory:
        role = msg["role"]
        if role == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

question = st.chat_input("Ask a question about the uploaded PDFs")
if question:
    st.chat_message("user").write(question)
    with st.spinner("Agents are collaborating..."):
        answer = run_multi_agent(question, max_iters=MAX_AGENT_ITERS)
    st.chat_message("assistant").write(answer)

st.sidebar.markdown("---")
if st.sidebar.button("Show index size"):
    vs = st.session_state.vector_store
    count = len(vs.id_to_text) if vs.id_to_text else 0
    st.sidebar.text(f"Chunks in index: {count}")
if st.sidebar.button("Show raw index (first 10 chunks)"):
    vs = st.session_state.vector_store
    if not vs.id_to_text:
        st.sidebar.text("No index built.")
    else:
        txt = "\n\n---\n\n".join(vs.id_to_text[:10])
        st.sidebar.text_area("Raw chunks (preview)", txt, height=300)
