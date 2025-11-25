import os
import json
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    raise ValueError("Set HUGGINGFACEHUB_API_TOKEN in your environment or .env")

import streamlit as st
from PyPDF2 import PdfReader
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from huggingface_hub import InferenceClient
import textwrap
import time

# Config

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"   
FAISS_INDEX_DIR = "faiss_data.npz"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_AGENT_ITER = 3   

#PDF -> text -> chunks
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


# Embeddings + FAISS 
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
            np.savez(path, ids=np.array([]), embs=np.array([]), texts=np.array([]))
            return
        embs = self.embedder.encode(self.id_to_text, convert_to_numpy=True, show_progress_bar=False)
        np.savez(path, embs=embs, texts=np.array(self.id_to_text, dtype=object))

    def load(self, path=FAISS_INDEX_DIR):
        if not os.path.exists(path):
            self.index = None
            self.id_to_text = []
            return
        data = np.load(path, allow_pickle=True)
        embs = data["embs"]
        texts = data["texts"].tolist()
        if embs.size == 0:
            self.index = None
            self.id_to_text = []
            return
        d = embs.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embs)
        self.id_to_text = texts

    def similarity_search(self, query: str, k=4):
        if self.index is None or len(self.id_to_text) == 0:
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

# Hugging Face LLM wrapper
hf_client = InferenceClient(token=HF_TOKEN)

def call_model_chat(system: str, user: str, max_tokens=512, temperature=0.2) -> str:
    """
    Call the HF chat model. Return the assistant content as a string.
    This wrapper attempts chat_completion API shape; adapt if your client version differs.
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
            content = resp.choices[0].message.get("content", "")
            return content

        if isinstance(resp, dict):
            return resp.get("generated_text") or str(resp)
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


def tool_search_pdf(query: str) -> str:
    vs: SimpleVectorStore = st.session_state.vector_store
    hits = vs.similarity_search(query, k=4)
    if not hits:
        return "No relevant information found in the uploaded PDFs."
    out = []
    for i, h in enumerate(hits, start=1):
        out.append(f"=== CHUNK {i} ===\n{h}\n")
    return "\n".join(out)

def tool_read_all(_: str = "") -> str:
    vs: SimpleVectorStore = st.session_state.vector_store
    return "\n\n".join(vs.id_to_text) if vs.id_to_text else "No PDF text available."

TOOLS = {
    "search_pdf": tool_search_pdf,
    "read_all": tool_read_all,
}


SYSTEM_INSTRUCTION = textwrap.dedent("""
You are an assistant that answers user questions using the uploaded PDF documents and tools.
You have two tools you can use:
1) search_pdf - input: a query string. Returns top-matching document chunks.
2) read_all - input: ignored. Returns all document chunks.

When you need to use a tool, output a JSON object on a single line EXACTLY like:
{"action": "tool", "tool": "<tool_name>", "input": "<input string>"}

When you have the final answer, output a JSON object EXACTLY like:
{"action": "answer", "output": "<final answer text>"}

Do not output anything except valid JSON objects in the above formats while you are deciding tool calls or returning the final answer.
Be concise and when quoting from the PDF, include short citations like [chunk 1].
""").strip()

def run_agent_loop(user_question: str, max_iters=MAX_AGENT_ITER) -> str:
    """
    Runs a small agent loop:
    - send system + user question to model
    - model may request tool calls by returning JSON
    - we execute tool calls and send tool outputs back to model
    - repeat until model returns action=answer or max_iters exhausted
    """
    # conversation memory kept per session
    if "convo_memory" not in st.session_state:
        st.session_state.convo_memory = []   

    history_text = ""
    for msg in st.session_state.convo_memory:
        history_text += f"{msg['role'].capitalize()}: {msg['content']}\n"

    model_user_message = f"{history_text}\nUser: {user_question}"

    assistant_response = ""
    tool_results = ""
    for step in range(max_iters):
        prompt_user = model_user_message
        if tool_results:
            prompt_user += f"\n\nTool results:\n{tool_results}"

        raw = call_model_chat(SYSTEM_INSTRUCTION, prompt_user, max_tokens=512, temperature=0.2)
        raw_stripped = raw.strip()
        parsed = None
        try:
            parsed = json.loads(raw_stripped.splitlines()[0])   
        except Exception:
            assistant_response = raw_stripped
            break

        if not isinstance(parsed, dict) or "action" not in parsed:
            assistant_response = raw_stripped
            break

        action = parsed.get("action")
        if action == "tool":
            tool_name = parsed.get("tool")
            tool_input = parsed.get("input", "")
            if tool_name not in TOOLS:
                tool_output = f"[error] unknown tool: {tool_name}"
            else:
                tool_fn = TOOLS[tool_name]
                tool_output = tool_fn(tool_input)
            tool_results += f"\n--- Result of {tool_name} for input: {tool_input} ---\n{tool_output}\n"
           
        elif action == "answer":
            assistant_response = parsed.get("output", "")
            break
        else:
            assistant_response = raw_stripped
            break

    if not assistant_response:
        assistant_response = "answer not available in the PDF or model did not return an answer."

    st.session_state.convo_memory.append({"role": "user", "content": user_question})
    st.session_state.convo_memory.append({"role": "assistant", "content": assistant_response})

    return assistant_response

# Streamlit 

st.set_page_config(page_title="Agentic PDF Chat", layout="wide")
st.title("Agentic PDF ChatBot")

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
    with st.spinner("Agent thinking..."):
        answer = run_agent_loop(question, max_iters=MAX_AGENT_ITER)
    st.chat_message("assistant").write(answer)

st.sidebar.markdown("---")
if st.sidebar.button("Show index size"):
    vs = st.session_state.vector_store
    count = len(vs.id_to_text) if vs.id_to_text else 0
    st.sidebar.text(f"Chunks in index: {count}")
