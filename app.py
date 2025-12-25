import streamlit as st
import os
import time
import json
import base64
import tempfile
import shutil
from dotenv import load_dotenv

# Loaders & Processing
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document

# --- CONFIGURATION ---
st.set_page_config(page_title="Custom RAG Chat", page_icon="📂", layout="wide")
load_dotenv()

# We use a temporary directory for the vector DB so it resets per session/upload
if "temp_db_path" not in st.session_state:
    st.session_state.temp_db_path = tempfile.mkdtemp()

MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# --- SIDEBAR: FILE UPLOAD ---
with st.sidebar:
    st.title(" Your Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
    process_btn = st.button("Process", type="primary")
    st.info("Upload files and click 'Process' to start chatting.")

# --- CACHED MODEL LOADING (Static) ---
@st.cache_resource
def load_models():
    if not os.getenv("GROQ_API_KEY") or not os.getenv("GOOGLE_API_KEY"):
        st.error(" Missing API Keys in .env file!")
        st.stop()
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    chat_model = ChatGroq(model=MODEL_NAME, temperature=0)
    return embeddings, chat_model

embeddings, chat_model = load_models()

# --- DOCUMENT PROCESSING FUNCTION ---
def process_documents(uploaded_files):
    """
    Reads uploaded files, chunks them, and rebuilds the Vector + BM25 DB.
    """
    docs = []
    
    # 1. Read Files
    with st.status(" Processing files...", expanded=True) as status:
        for file in uploaded_files:
            st.write(f" Reading {file.name}...")
            
            # Save to temp file to use standard loaders
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name
            
            # Load based on type
            try:
                if file.name.endswith(".pdf"):
                    loader = PyPDFLoader(tmp_path)
                elif file.name.endswith(".docx"):
                    loader = Docx2txtLoader(tmp_path)
                elif file.name.endswith(".txt"):
                    loader = TextLoader(tmp_path)
                else:
                    continue
                
                # Extract text
                loaded_docs = loader.load()
                docs.extend(loaded_docs)
            except Exception as e:
                st.error(f"Error loading {file.name}: {e}")
            finally:
                os.remove(tmp_path) # Clean up temp file

        # 2. Split Text
        st.write(f" Splitting {len(docs)} documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        # 3. Build Vector DB (Chroma)
        st.write("Building Vector Database...")
        # Clear old DB if exists
        if os.path.exists(st.session_state.temp_db_path):
            shutil.rmtree(st.session_state.temp_db_path)
        
        vector_db = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory=st.session_state.temp_db_path
        )

        # 4. Build BM25 Index
        st.write(" Building Keyword Index (BM25)...")
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = 5

        status.update(label=" Ready to Chat!", state="complete", expanded=False)
        
        return vector_db, bm25_retriever

# --- STATE MANAGEMENT ---
if process_btn and uploaded_files:
    db, bm25 = process_documents(uploaded_files)
    st.session_state.vector_db = db
    st.session_state.bm25_retriever = bm25
    st.session_state.messages = [] # Reset chat on new upload

# --- HELPER FUNCTIONS (Same as before) ---
def reciprocal_rank_fusion(vector_docs, bm25_docs, k=60):
    fused_scores = {}
    doc_map = {}
    for rank, doc in enumerate(vector_docs):
        doc_key = doc.page_content[:200]
        doc_map[doc_key] = doc
        fused_scores[doc_key] = fused_scores.get(doc_key, 0) + (1 / (rank + k))
    
    for rank, doc in enumerate(bm25_docs):
        doc_key = doc.page_content[:200]
        if doc_key not in doc_map: doc_map[doc_key] = doc
        fused_scores[doc_key] = fused_scores.get(doc_key, 0) + (1 / (rank + k))
        
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, score in sorted_docs[:3]]

def rewrite_query(history, question):
    if not history: return question
    msg = [SystemMessage(content="Rewrite as standalone search query."), *history, HumanMessage(content=question)]
    try:
        return chat_model.invoke(msg).content.strip()
    except:
        return question

def generate_answer(docs, history, question):
    context = "\n\n".join([f"[Doc {i+1}]: {d.page_content}" for i, d in enumerate(docs)])
    prompt = f"Answer using this context:\n{context}\n\nQuestion: {question}"
    msg = [SystemMessage(content="You are a helpful assistant."), *history, HumanMessage(content=prompt)]
    try:
        return chat_model.invoke(msg).content.strip()
    except Exception as e:
        return f"Error: {e}"

# --- MAIN CHAT UI ---
st.title(" Chat with Your Documents")

if "vector_db" not in st.session_state:
    st.info("Please upload documents and click 'Process' to begin.")
else:
    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your files..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Prepare history
            history_lc = []
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user": history_lc.append(HumanMessage(content=msg["content"]))
                else: history_lc.append(AIMessage(content=msg["content"]))

            # RAG Pipeline
            with st.status("Thinking...", expanded=False):
                query = rewrite_query(history_lc[-2:], prompt)
                
                # Hybrid Search
                vector_docs = st.session_state.vector_db.similarity_search(query, k=5)
                bm25_docs = st.session_state.bm25_retriever.invoke(query)
                final_docs = reciprocal_rank_fusion(vector_docs, bm25_docs)
            
            # Answer
            response = generate_answer(final_docs, history_lc[-2:], prompt)
            message_placeholder.markdown(response)
            
            # Show Sources
            with st.expander("View Source Context"):
                for i, doc in enumerate(final_docs):
                    st.caption(f"Source {i+1} (from {doc.metadata.get('source', 'unknown')})")
                    st.text(doc.page_content[:500] + "...")

        st.session_state.messages.append({"role": "assistant", "content": response})